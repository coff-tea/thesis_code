"""
Author: Xi Lu
File: synth_optimize.py
Description: Used to determine best synthetic generative parameters based on specified metric.
""" 



#===================================================================================================
import sys
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde as kde
from scipy.special import kl_div as kld
from scipy.linalg import sqrtm
from math import ceil

from Helpers import *

 

#===================================================================================================
#### Argparser####
parser = argparse.ArgumentParser()
# Mandatory arguments
parser.add_argument("mode", type=str, help="Dictate mode for finding synth parameters.", choices=["likeness", "classify", "detect", "bootstrap", "fid"])
parser.add_argument("metric", type=str, help="Assessment metric for optimisation process.")
parser.add_argument("model_name", type=str, help="Which model to use.", choices=["simple", "vgg19bn", "res152", "dense161"])
# Data arguments
parser.add_argument("-max_data", type=int, default=100000, required=False, help="Upper cap on number of samples for each type/subset. [DEFAULT: %(default)s]")
parser.add_argument("-hyp_name", type=str, default="", required=False, \
                    help="If using hyperparameters, provide the full file name. [DEFAULT: %(default)s]")
parser.add_argument("-bh_name", type=str, default="", required=False, \
                    help="If using bellhop masks, provide the full file name. [DEFAULT: %(default)s]")
# Enqueue from previous
parser.add_argument("-enqn", type=int, default=5, required=False, \
                    help="How many to enqueue. [DEFAULT: %(default)s]")
parser.add_argument("-enqs", type=str, default="", required=False, \
                    help="Name of previous file. [DEFAULT: %(default)s]")
parser.add_argument("-enqd", type=str, default="minimize", required=False, \
                    help="Direction of loaded results. [DEFAULT: %(default)s]")
# Misc. arguments
parser.add_argument("-mdl", "--model_paras", type=str, default="", dest="mdl", required=False, help="Name of pre-trained model to load. [DEFAULT: %(default)s]")
parser.add_argument("-y", "--yaml", type=str, default="synth_optimize", dest="y", required=False, help="Name of yaml file storing relevant information. [DEFAULT: %(default)s]")

args = parser.parse_args()
yaml = yaml.safe_load(open(f"Helpers/{args.y}.yaml", "r"))  # Stores relevant information that is less transient than above

 

#===================================================================================================
""" FUNCTION: train_epoch
Train through batches in a dataloader using global parameters of the file. Returns tuple of epoch results (loss, accuracy (%), false alarm (%), missed detection (%)).

Args:
    - dataloader (torch.utils.data.DataLoader object): Dataloader to be trained using.
"""
def train_epoch(dataloader):
    global model
    global opt
    global crit
    global device
    
    ep_loss = 0.
    y_pred = []
    y_true = []
    model.train()
    for _, data in enumerate(dataloader):
        x, y = data[0].to(device), data[1].to(device)
        y = torch.unsqueeze(y, dim=1).float()
        opt.zero_grad()
        output = model(x.float())

        loss = crit(output, y)
        loss.backward()
        opt.step()

        ep_loss += loss.item()
        pred = torch.zeros_like(output)
        pred[torch.sigmoid(output) >= 0.5] = 1.
        y_pred.extend([p.item() for p in pred])
        y_true.extend([true.item() for true in y])

    cf_matrix = confusion_matrix(y_true, y_pred)
    ep_acc = (cf_matrix[0][0] + cf_matrix[1][1])/np.sum(cf_matrix) * 100
    ep_fa = cf_matrix[0][1] / np.sum(cf_matrix[0]) * 100
    ep_md = cf_matrix[1][0] / np.sum(cf_matrix[1]) * 100

    return ep_loss / len(dataloader), ep_acc, ep_fa, ep_md

 

#===================================================================================================
""" FUNCTION: eval_dataloader
Evaluate batches in a dataloader using global parameters of the file. Returns tuple of epoch results (loss, accuracy (%), false alarm (%), missed detection (%)).

Args:
    - dataloader (torch.utils.data.DataLoader object): Dataloader to be evaluated.
"""
def eval_dataloader(dataloader):
    global model
    global crit
    global device

    dl_loss = 0.
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for _, data in enumerate(dataloader):
            x, y = data[0].to(device), data[1].to(device)
            y = torch.unsqueeze(y, dim=1).float()
            
            output = model(x.float())
            loss = crit(output, y)

            dl_loss += loss.item()
            pred = torch.zeros_like(output)
            pred[torch.sigmoid(output) >= 0.5] = 1.
            y_pred.extend([p.item() for p in pred])
            y_true.extend([true.item() for true in y])

    if y_pred == y_true:
        return 0, 100, 0, 0

    cf_matrix = confusion_matrix(y_true, y_pred)
    dl_acc = (cf_matrix[0][0] + cf_matrix[1][1])/np.sum(cf_matrix) * 100
    dl_fa = cf_matrix[0][1] / np.sum(cf_matrix[0]) * 100
    dl_md = cf_matrix[1][0] / np.sum(cf_matrix[1]) * 100 

    return dl_loss / len(dataloader), dl_acc, dl_fa, dl_md


#===================================================================================================
""" FUNCTION: fid
Calculate fid score based on activation maps provided.

Args:
    - x1, x2 (Tensor): Activation maps, inputs for calculating fid score
"""
def fid(x1, x2):
    mu1, sigma1 = x1.mean(axis=0), np.cov(x1, rowvar=False)
    mu2, sigma2 = x2.mean(axis=0), np.cov(x2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


#===================================================================================================
""" FUNCTION: objective
Objective function for optuna hyperparameter tuning.

Args:
    - trial (int): Number
    - prune (bool): Allow early stopping [True]
"""
def objective(trial, prune=True):
    #--------- Relevant global variables
    global args
    global yaml
    global model
    global crit 
    global opt
    global device 
    global bh

    #--------- Suggest parameters
    polyfit = trial.suggest_int("polyfit", yaml["paras"]["polyfit_min"], yaml["paras"]["polyfit_max"], step=2)
    modi = trial.suggest_float("modi", yaml["paras"]["modi_min"], yaml["paras"]["modi_max"])
    width_mean = trial.suggest_float("width_mean", yaml["paras"]["width_mean_min"], yaml["paras"]["width_mean_max"])
    width_sd = trial.suggest_float("width_sd", yaml["paras"]["width_sd_min"], yaml["paras"]["width_sd_max"])
    width_k = trial.suggest_int("width_k", yaml["paras"]["width_k_min"], yaml["paras"]["width_k_max"], step=2)
    if bh is not None:
        intensity_mean = trial.suggest_float("intensity_mean", yaml["paras"]["intensity_mean_min"], yaml["paras"]["intensity_mean_max"])
        intensity_sd = trial.suggest_float("intensity_sd", yaml["paras"]["intensity_sd_min"], yaml["paras"]["intensity_sd_max"])
        intensity_k = trial.suggest_int("intensity_k", yaml["paras"]["intensity_k_min"], yaml["paras"]["intensity_k_max"], step=2)
    else:
        intensity_mean = 0 
        intensity_sd = 0 
        intensity_k = 0
    gauss_filter = trial.suggest_categorical("gauss_filter", [True, False])
    gauss_kernel = trial.suggest_float("gauss_kernel", yaml["paras"]["gauss_kernel_min"], yaml["paras"]["gauss_kernel_max"])
    
    #--------- Set up data
    if yaml['data']['channel'] > 0:
        data_name_prefix = f"{yaml['folders']['data']}/{yaml['data']['prefix']}_X{yaml['data']['channel']}"
    else:
        data_name_prefix = f"{yaml['folders']['data']}/{yaml['data']['prefix']}_X"
    
    synth_dict = np.load(f"{yaml['folders']['data']}/{yaml['data']['synth']}.npy", allow_pickle=True).item()
    syn_data, _ = synth.make_synth(synth_dict, polyfit, [width_mean, width_sd, width_k], [intensity_mean, intensity_sd, intensity_k], max_data=args.max_data, \
                                gauss_filter=gauss_filter, gauss_kernel=gauss_kernel, width_bounds=[yaml["paras"]["width_min"], yaml["paras"]["width_max"]])
    del synth_dict
    non_frames = spec_data.load_data(np.load(f"{data_name_prefix}non.npy")[:args.max_data])
    non_frames = [synth.overlay_synth(non_frames[i]) for i in range(len(non_frames))]
    syn_frames = []
    for i in range(min(len(non_frames), len(syn_data))):
        syn_temp = syn_data[i]
        if bh is not None:
            m = np.random.randint(bh.shape[0])
            syn_temp = synth.mask_synth(syn_temp, bh[m])
        syn_frames.append(synth.overlay_synth(non_frames[i], add=syn_temp, modi=modi))
    del non_frames
    del syn_data

    pos_frames = spec_data.load_data(np.load(f"{data_name_prefix}pos.npy")[:args.max_data])
    pos_frames = [synth.overlay_synth(pos_frames[i]) for i in range(len(pos_frames))] 
    
    #--------- KDE
    if args.mode == "kde":
        x_scale = np.linspace(0, 1, 100)
        total_kld = 0.
        if args.metric == "matchup":
            for s in syn_frames:
                s_kde = kde(s.flatten()).evaluate(x_scale)
                p = np.random.randint(0, high=len(pos_frames))
                p_kde = kde(pos_frames[p].flatten()).evaluate(x_scale)
                total_kld += kld(s_kde, p_kde)
            total_kld /= len(syn_frames)
        elif args.metric == "average":
            s_kde = None
            for s in syn_frames:
                if s_kde:
                    s_kde += kde(s.flatten()).evaluate(x_scale)
                else:
                    s_kde = kde(s.flatten()).evaluate(x_scale)
            s_kde /= len(syn_frames)
            p_kde = None
            for p in pos_frames:
                if p_kde:
                    p_kde += kde(p.flatten()).evaluate(x_scale)
                else:
                    p_kde = kde(p.flatten()).evaluate(x_scale)
            p_kde /= len(pos_frames)
            total_kld = kld(s_kde, p_kde)
        elif args.metric == "total":
            s_kde = None
            syn_frames = torch.Tensor(syn_frames)
            s_kde = kde(syn_frames.flatten()).evaluate(x_scale)
            p_kde = None
            pos_frames = torch.Tensor(pos_frames)
            p_kde = kde(pos_frames.flatten()).evaluate(x_scale)
            total_kld = kld(s_kde, p_kde)

    #--------- KDE and FID
    if args.mode == "fid":
        model = detectors.make_detector(args.model_name, 1, yaml["data"]["dim"])
        model.load_state_dict(torch.load(f"{yaml['folders']['load']}/{args.mdl}.pt", map_location=device))
        
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                if args.model_name == "dense161":
                    activation[name] = torch.nn.functional.adaptive_avg_pool2d(output.detach(), (1,1)).flatten()
                else:
                    activation[name] = output.detach().flatten()
            return hook
        target_layer = None
        target_name = None
        if args.model_name == "simple":
            target_layer = model.flat
            target_name = "flat"
        elif args.model_name == "vgg19bn":
            target_layer = model.avgpool
            target_name = "avgpool"
        elif args.model_name == "res152":
            target_layer = model.avgpool
            target_name = "avgpool"
        elif args.model_name == "dense161":
            target_layer = model.features
            target_name = "features"
        target_layer.register_forward_hook(get_activation(target_name))

        total_fid = 0.
        if args.metric == "inf":
            min_samples = min(len(syn_frames), len(pos_frames))
            batches = int(np.ceil(min_samples/yaml["training"]["batch_size"]))
            _ = model(syn_frames[0][None, :].float())
            min_samples = min(len(syn_frames), len(pos_frames))
            flat_dim = activation[target_name].shape[0]
            s_acts = torch.empty(min_samples, flat_dim)
            for i in range(min_samples):
                _ = model(syn_frames[i][None, :].float())
                s_acts[i] = activation[target_name]
            del syn_frames
            p_acts = torch.empty(min_samples, flat_dim)
            for i in range(min_samples):
                _ = model(pos_frames[i][None, :].float())
                p_acts[i] = activation[target_name]
            del pos_frames
            fids = []
            fid_batches = np.linspace(yaml["training"]["batch_size"], min_samples, ceil(min_samples/yaml["training"]["batch_size"])).astype('int32')
            for fid_b in fid_batches:
                indices = torch.randperm(s_acts.shape[0])
                s_batch = s_acts[indices[:fid_b], :]
                indices = torch.randperm(p_acts.shape[0])
                p_batch = p_acts[indices[:fid_b], :]
                fids.append(fid(np.array(s_batch), np.array(p_batch)))
            fids = np.array(fids).reshape(-1, 1)
            reg = LinearRegression().fit(1/fid_batches.reshape(-1, 1), fids)
            total_fid = max(0, reg.predict(np.array([[0]]))[0,0])
        elif args.metric == "batch":
            min_samples = min(len(syn_frames), len(pos_frames))
            batches = int(np.ceil(min_samples/yaml["training"]["batch_size"]))
            _ = model(syn_frames[0][None, :].float())
            s_acts = torch.empty(yaml["training"]["batch_size"], activation[target_name].shape[0])
            p_acts = torch.empty(yaml["training"]["batch_size"], activation[target_name].shape[0])
            for b in range(batches):
                for i in range(yaml["training"]["batch_size"]):
                    my_idx = b*yaml["training"]["batch_size"]+i
                    if my_idx >= min_samples:
                        s_acts = s_acts[:my_idx]
                        p_acts = p_acts[:my_idx]
                        break
                    _ = model(syn_frames[my_idx][None, :].float())
                    s_acts[i] = activation[target_name]
                    _ = model(pos_frames[my_idx][None, :].float())
                    p_acts[i] = activation[target_name]
                total_fid += fid(np.array(s_acts), np.array(p_acts))
            total_fid /= batches
        elif args.metric == "total":
            _ = model(syn_frames[0][None, :].float())
            s_acts = torch.empty(len(syn_frames), activation[target_name].shape[0])
            for s in range(len(syn_frames)):
                _ = model(syn_frames[s][None, :].float())
                s_acts[s] = activation[target_name]
            p_acts = torch.empty(len(pos_frames), activation[target_name].shape[0])
            for p in range(len(pos_frames)):
                _ = model(pos_frames[s][None, :].float())
                p_acts[p] = activation[target_name]
            total_fid = fid(np.array(s_acts), np.array(p_acts))
        return total_fid if total_fid >= 0 else 1e100

    #--------- CLASSIFY, DETECT, and BOOTSTRAP
    idx_dict = dict()
    trainloader, testloader, valloader = None, None, None
    if args.mode == "detect" or args.mode == "bootstrap":    # Split synth and negative (neg) into training/validation sets, put positive [and negative (unn)] into testing set
        #--------- testloader
        # Note: Using "training samples" from pos and unn subsets
        temp_train, temp_val, temp_test = tuple(spec_data.split_sets(len(pos_frames), [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]))
        test_data = [[np.array(pos_frames[i]), 1] for i in temp_train]
        if args.mode == "detect":
            unn_frames = spec_data.load_data(np.load(f"{data_name_prefix}unn.npy")[:args.max_data])
            temp_train, temp_val, temp_test = tuple(spec_data.split_sets(len(unn_frames), [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]))
            unn_frames = [synth.overlay_synth(unn_frames[i]) for i in temp_train]
            for i in range(len(unn_frames)):
                test_data.append([np.array(unn_frames[i]), 0])
            del unn_frames
        testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=yaml["training"]["batch_size"])
        del pos_frames
        del test_data

        #--------- trainloader and valloader
        neg_frames = spec_data.load_data(np.load(f"{data_name_prefix}neg.npy")[:args.max_data])
        neg_frames = [synth.overlay_synth(neg_frames[i]) for i in range(len(neg_frames))]
        idx_dict["neg"] = tuple(spec_data.split_sets(len(neg_frames), [yaml["training"]["train_split"]], replicable=yaml["replicable"], seed=yaml["seed"]+1))
        idx_dict["syn"] = tuple(spec_data.split_sets(len(syn_frames), [yaml["training"]["train_split"]], replicable=yaml["replicable"], seed=yaml["seed"]+2))
        train_data = [[np.array(neg_frames[i]), 0] for i in idx_dict["neg"][0]]
        for i in idx_dict["syn"][0]:
            train_data.append([np.array(syn_frames[i]), 1]) 
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=yaml["training"]["batch_size"])
        del train_data
        val_data = [[np.array(neg_frames[i]), 0] for i in idx_dict["neg"][1]]
        for i in idx_dict["syn"][1]:
            val_data.append([np.array(syn_frames[i]), 1]) 
        valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=yaml["training"]["batch_size"])
        del val_data
        del neg_frames 
    elif args.mode == "classify":    # Split positive and synthetic into t/t/v sets
        idx_dict["pos"] = tuple(spec_data.split_sets(len(pos_frames), [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]))
        idx_dict["syn"] = tuple(spec_data.split_sets(len(syn_frames), [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]+2))
        #--------- trainloader
        train_data = [[np.array(pos_frames[i]), 1] for i in idx_dict["pos"][0]]
        for i in idx_dict["syn"][0]:
            train_data.append([np.array(syn_frames[i]), 0])
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=yaml["training"]["batch_size"])
        del train_data
        #--------- testloader
        test_data = [[np.array(pos_frames[i]), 1] for i in idx_dict["pos"][1]]
        for i in idx_dict["syn"][1]:
            test_data.append([np.array(syn_frames[i]), 0])
        testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=yaml["training"]["batch_size"])
        del test_data
        #--------- valloader
        val_data = [[np.array(pos_frames[i]), 1] for i in idx_dict["pos"][2]]
        for i in idx_dict["syn"][2]:
            val_data.append([np.array(syn_frames[i]), 0]) 
        valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=yaml["training"]["batch_size"])
        del pos_frames
        del val_data
    del syn_frames

    print(f"Using {len(trainloader.dataset)}, {len(testloader.dataset)}, {len(valloader.dataset)} t/t/v samples")

    #--------- Pre-training setup 
    if args.hyp_name != "":
        load_hyps = np.load(f"{yaml['folders']['load']}/{args.hyp_name}.npy", allow_pickle=True).item()
        best_val = 1000 
        best_key = 0 
        for k, v in load_hyps.items():
            if v["trial_value"] < best_val:
                best_val = v["trial_value"]
                best_key = k 
        hyper = load_hyps[k]
        print(f"Found {args.hyp_name} best key!")
    else:
        hyper = yaml["defaults"]
    model = detectors.make_detector(args.model_name, 1, yaml["data"]["dim"])
    model.to(device)
    class_weights[0] = hyper["pos"]
    crit = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    opt = optim.Adam(model.parameters(), lr=hyper["lr"], betas=(hyper["beta1"], hyper["beta2"]), weight_decay=hyper["decay"])
    
    #--------- Training loop
    no_change = 0
    best_at = 0
    best_vloss, best_vacc, _, _ = eval_dataloader(valloader)
    best_sloss, best_sacc, _, _ = eval_dataloader(testloader)
    stop_after = yaml["training"]["stop_after"]
    if args.model_name == "simple":     # Smaller model, train for longer
        stop_after *= 2
    for epoch in range(yaml["training"]["max_epochs"]):
        ept_loss, ept_acc, _, _ = train_epoch(trainloader)
        epv_loss, epv_acc, _, _ = eval_dataloader(valloader)
        if epv_loss + + yaml["training"]["improve_margin"] < best_vloss:
            best_vloss = epv_loss
            best_vacc = epv_acc
            best_sloss, best_sacc, _, _ = eval_dataloader(testloader)
            best_at = epoch + 1
            no_change = 0
        else:
            no_change += 1
        if args.mode == "detect" or args.mode == "bootstrap":
            if args.metric == "loss":
                if best_sacc - best_vacc > 25:      # Likely a case of all 0/1 predictions
                    trial.report(100, epoch)
                else:
                    trial.report(best_sloss, epoch)
            elif args.metric == "acc":
                if best_sacc - best_vacc > 25:      # Likely a case of all 0/1 predictions
                    trial.report(0, epoch)
                else:
                    trial.report(best_sacc, epoch)
        elif args.mode == "classify":
            if args.metric == "loss":
                trial.report(best_sloss, epoch)
            elif args.metric == "acc":
                trial.report(np.abs(50 - best_sacc), epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if (epoch+1) % yaml["training"]["print_status"] == 0:
            print(f"{args.mode} || Epoch {epoch+1}: training {ept_loss:.4f}/{ept_acc:.4f}, validation {epv_loss:.4f}/{epv_acc:.4f}, and best test {best_sloss:.4f}/{best_sacc:.2f} at {best_at}")
        if no_change >= stop_after:
            break
    if args.mode == "detect" or args.mode == "bootstrap":
        if args.metric == "loss":
            return 100 if (best_sacc - best_vacc > 25) else best_sloss
        elif args.metric == "acc":
            return 0 if (best_sacc - best_vacc > 25) else best_sacc
    elif args.mode == "classify":
        if args.metric == "loss":
            return best_sloss
        elif args.metric == "acc":
            return np.abs(50 - best_sacc)
        elif args.metric == "epoch":
            return best_at

    print("Error in optuna trial!")
    return None



#===================================================================================================
#### Main code ####
matchups = {"likeness": ["matchup", "average", "total"], \
            "classify": ["loss", "acc", "epoch"], \
            "detect": ["loss", "acc"], \
            "bootstrap": ["loss", "acc"], \
            "fid": ["inf", "batch", "total"]}
incl_model = ["classify", "detect", "bootstrap", "fid"]
if args.metric not in matchups[args.mode]:
    print(f"Given invalid mode/metric pairing {args.mode}/{args.metric}!")
    exit()

study = None
model = None 
opt = None 
crit = None 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
class_weights = torch.ones(1)
save_name = f"{yaml['save_name']}_synthopt_{args.mode}_{args.metric}"
if args.mode in incl_model:
    save_name = f"{save_name}_{args.model_name}"
print(f"\n\nWorking on {save_name}!")

try:
    opt_paras = np.load(f"{yaml['folders']['load']}/{save_name}.npy", allow_pickle=True).item()
    if len(opt_paras) >= yaml["training"]["max_trials"]:
        print("Max number of optimization trials reached!")
        exit()
except:
    opt_paras = dict()

#--------- Create study 
if args.mode == "likeness":
    study = optuna.create_study(direction="minimize", study_name=save_name)
elif args.mode == "classify":
    if args.metric == "loss" or args.metric == "epoch":
        study = optuna.create_study(direction="maximize", study_name=save_name)
    elif args.metric == "acc":
        study = optuna.create_study(direction="minimize", study_name=save_name)
elif args.mode == "detect" or args.mode == "bootstrap":
    if args.metric == "loss":
        study = optuna.create_study(direction="minimize", study_name=save_name)
    elif args.metric == "acc":
        study = optuna.create_study(direction="maximize", study_name=save_name)
elif args.mode == "fid":
    study = optuna.create_study(direction="minimize", study_name=save_name)
else:
    print("Mode error")
    exit()

#--------- Setup
distributions = {"polyfit": optuna.distributions.IntDistribution(yaml["paras"]["polyfit_min"], yaml["paras"]["polyfit_max"], step=2), \
                 "modi": optuna.distributions.FloatDistribution(yaml["paras"]["modi_min"], yaml["paras"]["modi_max"]), \
                 "width_mean": optuna.distributions.FloatDistribution(yaml["paras"]["width_mean_min"], yaml["paras"]["width_mean_max"]), \
                 "width_sd": optuna.distributions.FloatDistribution(yaml["paras"]["width_sd_min"], yaml["paras"]["width_sd_max"]), \
                 "width_k": optuna.distributions.IntDistribution(yaml["paras"]["width_k_min"], yaml["paras"]["width_k_max"], step=2), \
                 "intensity_mean": optuna.distributions.FloatDistribution(yaml["paras"]["intensity_mean_min"], yaml["paras"]["intensity_mean_max"]), \
                 "intensity_sd": optuna.distributions.FloatDistribution(yaml["paras"]["intensity_sd_min"], yaml["paras"]["intensity_sd_max"]), \
                 "intensity_k": optuna.distributions.IntDistribution(yaml["paras"]["intensity_k_min"], yaml["paras"]["intensity_k_max"], step=2), \
                 "gauss_filter": optuna.distributions.CategoricalDistribution([False, True]), \
                 "gauss_kernel": optuna.distributions.FloatDistribution(yaml["paras"]["gauss_kernel_min"], yaml["paras"]["gauss_kernel_max"])}
bh = None 
if args.bh_name != "":      # Intensity masks generated through bellhop
    bh = np.load(f"{yaml['folders']['data']}/{args.bh_name}.npy", allow_pickle=True)

#--------- Load study's previous trials, if they exist
for k, v in opt_paras.items():
    params = {name: chosen for name, chosen in v.items() if name != "trial_value"}
    trial = optuna.trial.create_trial(params=params, distributions=distributions, value=v["trial_value"])
    study.add_trial(trial)
print(f"Number of trials done is {len(study.trials)}")

#--------- Enquing possible study values from file, if given
if args.enqs != "":     
    tried = np.load(f"{yaml['folders']['load']}/{args.enqs}.npy", allow_pickle=True).item()
    sorts = pd.DataFrame(columns=["key", "val"])
    for k, v in tried.items():
        row = {"key": k, "val": v["trial_value"]}
        sorts = sorts.append(row, ignore_index=True)
    if args.enqd == "minimize":
        sorts = sorts.sort_values(by=["val"])
    else:
        sorts = sorts.sort_values(by=["val"], ascending=False)
    for r in range(args.enqn):
        if len(study.trials) < r+1:
            enq_trial = dict()
            for k in distributions.keys():
                enq_trial[k] = tried[int(sorts.iloc[r]["key"])][k]
            study.enqueue_trial(enq_trial)
            break

study.optimize(lambda trial:objective(trial), n_trials=yaml["training"]["tune_trials"])

#--------- Save study results
new_paras = dict()
count = 1
for t in study.trials:
    trial_dict = dict()
    trial_dict["trial_value"] = t.value
    for k, v in t.params.items():
        trial_dict[k] = v 
    new_paras[count] = trial_dict 
    count += 1
np.save(f"{yaml['folders']['results']}/{save_name}.npy", new_paras, allow_pickle=True)