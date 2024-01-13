"""
Author: Xi Lu
File: cross_tune_train.py
Description: Used to cross-validate, tune, and train select image classification models for the task of dolphin whistle detection.
""" 



#===================================================================================================
import sys
import random

import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import confusion_matrix

from Helpers import *



#===================================================================================================
#### Argparser and yaml####
parser = argparse.ArgumentParser()
# Mandatory arguments
parser.add_argument("mode", type=str, help="Task to be done.", choices=["cross", "tune", "train"])
parser.add_argument("model_name", type=str, help="Which model to use.", \
                    choices=["simple", "vgg16tf", "vgg16tfd", "vgg16", "vgg16bn", "vgg19", "vgg19bn", "res50", "res101", "res152", "dense161", "dense169", "dense201"])
parser.add_argument("data_format", type=str, help="Data format.", choices=["single", "all", "stk", "avg", "stkwavg"])
# Data-related arguments
parser.add_argument("-c", "--channel", type=int, default=1, dest="c", required=False, help="If using single channel, which channel. [DEFAULT: %(default)s]")
parser.add_argument("-a", action="store_true",  help="Use Add-Pre data pre-processing. [DEFAULT: %(default)s]")
parser.add_argument("-p", "--partial", type=float, default=1.0, dest="p", required=False, help="If using only part of a dataset for training. [DEFAULT: %(default)s]")
parser.add_argument("-l", "--longer", type=int, default=1, dest="l", required=False, help="Use a multiplier for stop_after time if using partial dataset. [DEFAULT: %(default)s]")
# Model-related arguments
parser.add_argument("-g", action="store_true",  help="Use global average pooling rather than linear layer. [DEFAULT: %(default)s]")
parser.add_argument("-f", "--freeze", type=int, default=0, dest="f", required=False, help="If freezing part of pre-trained layer, only valid with certain model types. [DEFAULT: %(default)s]")
parser.add_argument("-o", action="store_true",  help="Do not use pretrained model. [DEFAULT: %(default)s]")
parser.add_argument("-r", "--retrain", type=str, default="", dest="r", required=False, help="If retraining a saved model, provide the model name. [DEFAULT: %(default)s]")
parser.add_argument("-hy", "--hyper", type=str, default="", dest="hy", required=False, help="If using non-standard naming for hyperparameters, provide the file name. [DEFAULT: %(default)s]")
# Misc. arguments
parser.add_argument("-y", "--yaml", type=str, default="cross_tune_train", dest="y", required=False, help="Name of yaml file storing relevant information. [DEFAULT: %(default)s]")

args = parser.parse_args()
yaml = yaml.safe_load(open(f"Helpers/{args.y}.yaml", "r"))  # Stores relevant information that is less transient than above


#===================================================================================================
#### General setup ####
model = None
model_channels = yaml["data"]["mdl_channels"]
opt = None
crit = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_weights = torch.ones(1)
pretrained = not args.o
#--------- Set stopping condition value
stop_after = yaml["training"]["stop_after"] 
if args.model_name == "simple":     # Simple model requires more training to converge
    stop_after *= 2
elif args.mode == "cross":      # Stop earlier for cross mode for time
    stop_after /= 2
stop_after = int(stop_after)
#--------- Make and print saving name for this trial
save_name = f"{yaml['save_name']}"
if args.f > 0:
    save_name = save_name.replace("F#", f"F{args.f}")
if args.p > 0 and args.p < 1:
    replacement = f"{int(args.p*100)}"
    if len(replacement) == 1:
        replacement = f"0{replacement}"
    save_name = save_name.replace("P#", f"P{replacement}")
if yaml["replicable"]:
    save_name += f"_{yaml['seed']}"
if args.a:
    save_name += f"_{args.mode}_{args.model_name}_addpre_{args.data_format}"
else:
    save_name += f"_{args.mode}_{args.model_name}_minpre_{args.data_format}"
print(f"Working on {save_name}!")        # Print save_name
print("===============================================================================================")

#===================================================================================================
#### Load data ####
X_pos = []
X_neg = []
if args.a:      # Using Add-Pre (https://arxiv.org/abs/2211.15406)
    if yaml["data"]["channels"] == 1:
        X_pos.append(spec_data.load_data_tf(np.load(f"{yaml['folders']['data']}/{yaml['data']['prefix']}_Xpos.npy")))
    else:
        for ch in range(yaml["data"]["channels"]):
            X_pos.append(spec_data.load_data_tf(np.load(f"{yaml['folders']['data']}/{yaml['data']['prefix']}_X{ch+1}pos.npy")))
    if yaml["data"]["channels"] == 1:
        X_neg.append(spec_data.load_data_tf(np.load(f"{yaml['folders']['data']}/{yaml['data']['prefix']}_Xneg.npy")))
    else:
        for ch in range(yaml["data"]["channels"]):
            X_neg.append(spec_data.load_data_tf(np.load(f"{yaml['folders']['data']}/{yaml['data']['prefix']}_X{ch+1}neg.npy")))
else:       # Using Min-Pre
    if yaml["data"]["channels"] == 1:
        X_pos.append(spec_data.load_data(np.load(f"{yaml['folders']['data']}/{yaml['data']['prefix']}_Xpos.npy")))
    else:
        for ch in range(yaml["data"]["channels"]):
            X_pos.append(spec_data.load_data(np.load(f"{yaml['folders']['data']}/{yaml['data']['prefix']}_X{ch+1}pos.npy")))
    if yaml["data"]["channels"] == 1:
        X_neg.append(spec_data.load_data(np.load(f"{yaml['folders']['data']}/{yaml['data']['prefix']}_Xneg.npy")))
    else:
        for ch in range(yaml["data"]["channels"]):
            X_neg.append(spec_data.load_data(np.load(f"{yaml['folders']['data']}/{yaml['data']['prefix']}_X{ch+1}neg.npy")))
num_pos = len(X_pos[0])
num_neg = len(X_neg[0])        

#===================================================================================================
#### Dictionary for relevant indices ####            
idx_dict = dict()
if args.mode == "cross":        # Fold boundaries for k-cross validation
    # Positive indices
    fold_size = round(num_pos/yaml["training"]["k_folds"])
    fold_bounds = []
    for i in range(yaml["training"]["k_folds"]):
        fold_bounds.append(i*fold_size)
    fold_bounds.append(num_pos)
    idx_dict["pos"] = fold_bounds.copy()
    # Positive indices
    fold_size = round(num_neg/yaml["training"]["k_folds"])
    fold_bounds = []
    for i in range(yaml["training"]["k_folds"]):
        fold_bounds.append(i*fold_size)
    fold_bounds.append(num_neg)
    idx_dict["neg"] = fold_bounds.copy()
else:       # Which indices to use for which set
    idx_dict["pos"] = tuple(spec_data.split_sets(num_pos, [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]))
    idx_dict["neg"] = tuple(spec_data.split_sets(num_neg, [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]+1))


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
            for s in x:
                if torch.mean(s).isnan():
                    print(s)
                    print(torch.max(s))
                    print(torch.min(s))
            y = torch.unsqueeze(y, dim=1).float()

            output = model(x.float())
            loss = crit(output, y)

            dl_loss += loss.item()
            pred = torch.zeros_like(output)
            pred[torch.sigmoid(output) >= 0.5] = 1.
            y_pred.extend([p.item() for p in pred])
            y_true.extend([true.item() for true in y])

    cf_matrix = confusion_matrix(y_true, y_pred)
    dl_acc = (cf_matrix[0][0] + cf_matrix[1][1])/np.sum(cf_matrix) * 100
    dl_fa = cf_matrix[0][1] / np.sum(cf_matrix[0]) * 100
    dl_md = cf_matrix[1][0] / np.sum(cf_matrix[1]) * 100 

    return dl_loss / len(dataloader), dl_acc, dl_fa, dl_md


#===================================================================================================
#### K-FOLDS CROSSVALIDATION ####
if args.mode == "cross":
    #--------- Dictionaries to retain values, keyed with k# (fold)
    try:        # Try finding in "results_folder"
        cv_perf = np.load(f"{yaml['folders']['load']}/{save_name}.npy", allow_pickle=True).item()
    except:
        cv_perf = dict()
        cv_perf["hyper"] = (yaml["defaults"]["pos"], yaml["defaults"]["dropout"], yaml["defaults"]["lr"], \
                            yaml["defaults"]["decay"], yaml["defaults"]["beta1"], yaml["defaults"]["beta2"])
    folds_done = cv_perf.keys()
    #--------- Prepare to run
    model = detectors.make_detector(args.model_name, model_channels, yaml["data"]["dim"], dropout=yaml["defaults"]["dropout"], freeze=args.f, gap=args.g, pre=pretrained)
    model.to(device)
    torch.save(model.state_dict(), f"{yaml['folders']['temp']}/start.pt")
    pos_idx = [i for i in range(num_pos)]
    neg_idx = [i for i in range(num_neg)]
    if yaml["replicable"]: 
        random.seed(yaml["seed"])
    random.shuffle(pos_idx)
    random.shuffle(neg_idx)
    #--------- Iterate through folds
    for k in range(yaml["training"]["k_folds"]):
        #--------- Make dictionary key
        d_key = f"fold{k+1}"
        if d_key in folds_done:
            print(f"\tSkip {d_key}")
            continue
        print(f"\tTry {d_key}")
        #------------------ Set up model, criterion, optimizer
        model.load_state_dict(torch.load(f"{yaml['folders']['temp']}/start.pt"))
        torch.save(model.state_dict(), f"{yaml['folders']['temp']}/best.pt")
        class_weights[0] = yaml["defaults"]["pos"]
        crit = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        opt = optim.Adam(model.parameters(), lr=yaml["defaults"]["lr"], \
                         betas=(yaml["defaults"]["beta1"], yaml["defaults"]["beta2"]), weight_decay=yaml["defaults"]["decay"])
        #------------------ Make dataloaders
        pos_test_idx = [i for i in pos_idx[idx_dict["pos"][k]:idx_dict["pos"][k+1]]]
        pos_train_idx = [i for i in pos_idx if i not in pos_test_idx]
        test_data = spec_data.process_data(X_pos, pos_test_idx, args.data_format, model_channels, yaml["data"]["dim"], tag=1, which_ch=args.c)
        train_data = spec_data.process_data(X_pos, pos_train_idx, args.data_format, model_channels, yaml["data"]["dim"], tag=1, which_ch=args.c)
        neg_test_idx = [i for i in neg_idx[idx_dict["neg"][k]:idx_dict["neg"][k+1]]]
        neg_train_idx = [i for i in neg_idx if i not in neg_test_idx]
        test_data.extend(spec_data.process_data(X_neg, neg_test_idx, args.data_format, model_channels, yaml["data"]["dim"], tag=0, which_ch=args.c))
        train_data.extend(spec_data.process_data(X_neg, neg_train_idx, args.data_format, model_channels, yaml["data"]["dim"], tag=0, which_ch=args.c))
        trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=yaml["training"]["batch_size"])
        del train_data
        testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=yaml["training"]["batch_size"])
        del test_data
        #------------------ Start training
        train_losses = []
        train_accs = []
        no_change = 0
        best_tloss = 1000.
        best_tacc = 0.
        best_at = 0
        stopped_epoch = yaml["training"]["max_epochs"]
        for epoch in range(yaml["training"]["max_epochs"]):
            ept_loss, ept_acc, _, _ = train_epoch(trainloader)
            if ept_loss + yaml["training"]["improve_margin"] < best_tloss:
                best_tloss = ept_loss
                best_tacc = ept_acc
                best_at = epoch
                torch.save(model.state_dict(), f"{yaml['folders']['temp']}/best.pt")
                no_change = 0
            else:
                no_change += 1
            train_losses.append(ept_loss)
            train_accs.append(ept_acc)
            if no_change >= stop_after:
                stopped_epoch = epoch
                break
            if (epoch+1) % yaml["training"]["print_status"] == 0:
                print(f"\t\t\tOn epoch {epoch+1}: {best_tloss:.4f}")
        #------------------ Get test performance
        model.load_state_dict(torch.load(f"{yaml['folders']['temp']}/best.pt"))
        test_loss, test_acc, test_fa, test_md = eval_dataloader(testloader)
        cv_perf[d_key] = (best_tloss, best_tacc, best_at, test_loss, test_acc, test_fa, test_md)
        #------------------ Save to dictionary
        iter_dict = dict()
        iter_dict["acc"] = train_accs
        iter_dict["loss"] = train_losses
        iter_dict["perf"] = (best_tloss, best_tacc, best_at, test_loss, test_acc, test_fa, test_md)
        cv_perf[d_key] = iter_dict
        np.save(f"{yaml['folders']['results']}/{save_name}.npy", cv_perf, allow_pickle=True)


#===================================================================================================
#### HYPERPARAMETER TUNING ####
if args.mode == "tune":
    distr_ranges = {"dropout": (0.1, 0.9), "pos": (1, 3) , "lr": (1e-7, 1e-1), "decay": (1e-10, 1e-3), "beta1": (0.25, 0.99), "beta2": (0.25, 0.99)}
    distr_dict = {"pos": optuna.distributions.FloatDistribution(distr_ranges["pos"][0], distr_ranges["pos"][1]), \
                 "lr": optuna.distributions.FloatDistribution(distr_ranges["lr"][0], distr_ranges["lr"][1]), \
                 "decay": optuna.distributions.FloatDistribution(distr_ranges["decay"][0], distr_ranges["decay"][1]), \
                 "beta1": optuna.distributions.FloatDistribution(distr_ranges["beta1"][0], distr_ranges["beta1"][1]), \
                 "beta2": optuna.distributions.FloatDistribution(distr_ranges["beta2"][0], distr_ranges["beta2"][1])}   
    if "simple" in args.model_name or "tfd" in args.model_name:
        distr_dict["dropout"] = optuna.distributions.FloatDistribution(distr_ranges["dropout"][0], distr_ranges["dropout"][1])
    #------------------ Dictionaries to retain best hyperparameters, keyed by val_accuracy
    try:        # Try finding in "results_folder" 
        hyp_perf = np.load(f"{yaml['folders']['load']}/{save_name}.npy", allow_pickle=True).item()
    except:
        hyp_perf = dict()
    if len(hyp_perf.keys()) >= yaml["training"]["max_tune"]:
        exit()
    #------------------ Make dataloaders
    train_data = spec_data.process_data(X_pos, idx_dict["pos"][0], args.data_format, model_channels, yaml["data"]["dim"], tag=1, which_ch=args.c)
    train_data.extend(spec_data.process_data(X_neg, idx_dict["neg"][0], args.data_format, model_channels, yaml["data"]["dim"], tag=0, which_ch=args.c))
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=yaml["training"]["batch_size"])
    del train_data
    val_data = spec_data.process_data(X_pos, idx_dict["pos"][2], args.data_format, model_channels, yaml["data"]["dim"], tag=1, which_ch=args.c)
    val_data.extend(spec_data.process_data(X_neg, idx_dict["neg"][2], args.data_format, model_channels, yaml["data"]["dim"], tag=0, which_ch=args.c))
    valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=yaml["training"]["batch_size"])
    del val_data
    #------------------ Objective function
    def objective(trial, model_name, chs_in, spat_dim, max_epochs, prune=True):
        """ FUNCTION: objective
        Objective function for optuna hyperparameter tuning.

        Args:
            - trial (int): Number
            - model_name (str): Detector to be used
            - chs_in (int): Number of channels for input images
            - spat_dim (int): Spatial dimension for square input images
            - max_epochs (int): Allow this many epochs for training
            - prune (bool): Allow early stopping [True]
        """
        global model
        global crit 
        global opt
        global device 
        global trainloader
        global valloader
        #------------------ Create model, criterion, and optimiser
        if "simple" in args.model_name or "tfd" in args.model_name:
            dropout = trial.suggest_float("dropout", distr_ranges["dropout"][0], distr_ranges["dropout"][1], log=True)
            model = detectors.make_detector(args.model_name, chs_in, yaml["data"]["dim"], freeze=args.f, gap=args.g, dropout=dropout, pre=pretrained)
        else:
            model = detectors.make_detector(args.model_name, chs_in, yaml["data"]["dim"], freeze=args.f, gap=args.g, pre=pretrained)
        model.to(device)
        class_weights[0] = trial.suggest_float("pos", distr_ranges["pos"][0], distr_ranges["pos"][1], log=True)
        crit = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
        lr = trial.suggest_float("lr", distr_ranges["lr"][0], distr_ranges["lr"][1], log=True)
        decay = trial.suggest_float("decay", distr_ranges["decay"][0], distr_ranges["decay"][1], log=True)
        beta1 = trial.suggest_float("beta1", distr_ranges["beta1"][0], distr_ranges["beta1"][1], log=True)
        beta2 = trial.suggest_float("beta2", distr_ranges["beta2"][0], distr_ranges["beta2"][1], log=True)
        opt = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=decay)
        #------------------ Start training
        no_change = 0
        best_vloss = 1000.
        for epoch in range(max_epochs):
            _, _, _, _ = train_epoch(trainloader)
            epv_loss, _, _, _ = eval_dataloader(valloader)
            if epv_loss + yaml["training"]["improve_margin"] < best_vloss:
                best_vloss = epv_loss
                torch.save(model.state_dict(), f"{yaml['folders']['temp']}/best.pt")
                no_change = 0
            else:
                trial.report(best_vloss, epoch)
                if prune and trial.should_prune():
                    raise optuna.TrialPruned()
                no_change += 1
            if no_change >= stop_after:
                break
            if (epoch+1) % yaml["training"]["print_status"] == 0:
                print(f"\t\t\tOn epoch {epoch+1}: {best_vloss:.4f}")
        return best_vloss
    #------------------ Tune model
    study = optuna.create_study(direction="minimize", study_name=save_name)
    if len(hyp_perf.keys()) == 0:       # Starting point
        start_point = dict()
        if "simple" in args.model_name or "tfd" in args.model_name:
            start_point["dropout"] = yaml["defaults"]["dropout"]
        start_point["lr"] = yaml["defaults"]["lr"]
        start_point["decay"] = yaml["defaults"]["decay"]
        start_point["beta1"] = yaml["defaults"]["beta1"]
        start_point["beta2"] = yaml["defaults"]["beta2"]
        start_point["pos"] = yaml["defaults"]["pos"]
        study.enqueue_trial(start_point)
    else:       # Queue up best result from last run
        for k, v in hyp_perf.items():
            params = {name: chosen for name, chosen in v.items() if name != "trial_value"}
            trial = optuna.trial.create_trial(params=params, distributions=distr_dict, value=v["trial_value"])
            study.add_trial(trial)
    print(f"Number of trials done is {len(study.trials)}")
    study.optimize(lambda trial:objective(trial, args.model_name, model_channels, yaml["data"]["dim"], int(yaml["training"]["max_epochs"]/2)), \
                                          n_trials=yaml["training"]["tune_trials"], gc_after_trial=True)
    count = 1
    new_hyp = dict()
    for t in study.trials:
        trial_dict = dict()
        trial_dict["trial_value"] = t.value
        for k, v in t.params.items():
            trial_dict[k] = v 
        new_hyp[count] = trial_dict 
        count += 1
    np.save(f"{yaml['folders']['results']}/{save_name}.npy", new_hyp, allow_pickle=True)


#===================================================================================================
#### MODEL TRAINING ####
if args.mode == "train":
    if args.hy == "":
        hyp_name = save_name.replace("train", "tune")
        starter = hyp_name.split("_")[0]
        if "ver" in starter:
            new_starter = starter.split("ver")[0]
            hyp_name = hyp_name.replace(starter, new_starter)
    else:
        hyp_name = args.hy
    try:
        hyp_perf = np.load(f"Results/{hyp_name}.npy", allow_pickle=True).item()
        best_set = 10000
        best_key = 0
        for k, v in hyp_perf.items():
            if v["trial_value"] < best_set:
                best_set = v["trial_value"]
                best_key = k
        hyper = hyp_perf[best_key]
        print(f"\tFound {hyp_name}! Best at {best_key}: {best_set}!")
    except:
        hyper = yaml["defaults"]
        print("\tTraining using default hyperparameters!")
    #------------------ Cut down training set
    if args.p > 0 and args.p < 1:
        pos_train, _ = tuple(spec_data.split_sets(len(idx_dict["pos"][0]), [args.p], seed=yaml["seed"]))
        pos_list = []
        for i in pos_train:
            pos_list.append(idx_dict["pos"][0][i])
        idx_dict["pos"] = (pos_list, idx_dict["pos"][1], idx_dict["pos"][2])
        neg_train, _ = tuple(spec_data.split_sets(len(idx_dict["neg"][0]), [args.p], seed=yaml["seed"]+1))
        neg_list = []
        for i in pos_train:
            neg_list.append(idx_dict["neg"][0][i])
        idx_dict["neg"] = (neg_list, idx_dict["neg"][1], idx_dict["neg"][2])
        stop_after *= int(args.l)
    #------------------ Make dataloaders
    train_data = spec_data.process_data(X_pos, idx_dict["pos"][0], args.data_format, model_channels, yaml["data"]["dim"], tag=1, which_ch=args.c)
    train_data.extend(spec_data.process_data(X_neg, idx_dict["neg"][0], args.data_format, model_channels, yaml["data"]["dim"], tag=0, which_ch=args.c))
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=yaml["training"]["batch_size"])
    del train_data
    test_data = spec_data.process_data(X_pos, idx_dict["pos"][1], args.data_format, model_channels, yaml["data"]["dim"], tag=1, which_ch=args.c)
    test_data.extend(spec_data.process_data(X_neg, idx_dict["neg"][1], args.data_format, model_channels, yaml["data"]["dim"], tag=0, which_ch=args.c))
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=yaml["training"]["batch_size"])
    del test_data
    val_data = spec_data.process_data(X_pos, idx_dict["pos"][2], args.data_format, model_channels, yaml["data"]["dim"], tag=1, which_ch=args.c)
    val_data.extend(spec_data.process_data(X_neg, idx_dict["neg"][2], args.data_format, model_channels, yaml["data"]["dim"], tag=0, which_ch=args.c))
    valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=yaml["training"]["batch_size"])
    del val_data
    #------------------ Start training
    if "simple" in args.model_name or "tfd" in args.model_name:
        model = detectors.make_detector(args.model_name, model_channels, yaml["data"]["dim"], freeze=args.f, gap=args.g, dropout=hyper["dropout"], pre=pretrained)
    else:
        model = detectors.make_detector(args.model_name, model_channels, yaml["data"]["dim"], freeze=args.f, gap=args.g, pre=pretrained)
    model.to(device)
    if args.r != "":
        model.load_state_dict(torch.load(f"{yaml['folders']['load']}/{args.r}.pt"))
    torch.save(model.state_dict(), f"{yaml['folders']['temp']}/best.pt")
    class_weights[0] = hyper["pos"]
    crit = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
    opt = optim.Adam(model.parameters(), lr=hyper["lr"], betas=(hyper["beta1"], hyper["beta2"]), \
                     weight_decay=hyper["decay"])
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    no_change = 0
    best_vloss, best_vacc, _, _ = eval_dataloader(valloader)
    best_at = 0
    stopped_epoch = yaml["training"]["max_epochs"]
    for epoch in range(yaml["training"]["max_epochs"]):
        ept_loss, ept_acc, _, _ = train_epoch(trainloader)
        epv_loss, epv_acc, _, _ = eval_dataloader(valloader)
        if epv_loss + yaml["training"]["improve_margin"] < best_vloss:
            best_vloss = epv_loss
            best_vacc = epv_acc
            best_at = epoch+1
            torch.save(model.state_dict(), f"{yaml['folders']['temp']}/best.pt")
            no_change = 0
        else:
            no_change += 1
        train_losses.append(ept_loss)
        train_accs.append(ept_acc)
        val_losses.append(epv_loss)
        val_accs.append(epv_acc)
        if no_change >= stop_after:
            stopped_epoch = epoch
            break
        if (epoch+1) % yaml["training"]["print_status"] == 0:
            print(f"\t\t\tOn epoch {epoch+1}, best is {best_vloss:.4f} in epoch {best_at}")
    #==================== SAVE RESULTS
    train_perf = dict()
    model.load_state_dict(torch.load(f"{yaml['folders']['temp']}/best.pt"))
    test_loss, test_acc, test_fa, test_md = eval_dataloader(testloader)
    print("\tTest results: ", test_loss, test_acc)
    if args.p > 0 and args.p < 1:
        train_perf["percent"] = args.p
        train_perf["longer"] = args.l
    train_perf["hyper"] = hyper
    train_perf["train_hist"] = (train_losses, train_accs)
    train_perf["val_hist"] = (val_losses, val_accs)
    train_perf["best"] = (best_vloss, best_vacc, best_at, stopped_epoch)
    train_perf["test"] = (test_loss, test_acc, test_fa, test_md)
    if args.r != "":
        train_perf["starting"] = args.r
    np.save(f"{yaml['folders']['results']}/{save_name}.npy", train_perf, allow_pickle=True)
    torch.save(model.state_dict(), f"{yaml['folders']['results']}/{save_name}.pt")