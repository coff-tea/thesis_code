"""
Author: Xi Lu
File: generate_diffusion.py
Description: Used to tune and train DDPMs for purpose of dolphin whistle image generation.
""" 



#===================================================================================================
import numpy as np
import pandas as pd
import optuna
import argparse
import yaml
import torch
import torch.nn as nn
from typing import Dict, Tuple
from torch.utils.data import DataLoader, random_split
from torchvision import models, transforms
from torchvision.datasets import MNIST
from math import floor

from Helpers import *



#===================================================================================================
#### Argparser and yaml ####
parser = argparse.ArgumentParser()
# Mandatory
parser.add_argument("mode", type=str, help="Task to be done.", choices=["tune", "train"])
parser.add_argument("org", type=str, help="Specify structure of diffusion model(s) needed.", choices=["single", "chain"])
# Data-related arguments
parser.add_argument("-c", "--channel", type=int, default=1, dest="c", required=False, help="If using single channel, which channel. [DEFAULT: %(default)s]")
parser.add_argument("-custclass", type=str, default="", required=False, help="Custom .csv file for classes. [DEFAULT: %(default)s]")
parser.add_argument("-rp", "--realpara", type=str, default="", dest="rp", required=False, help="Name for realistic generation target. [DEFAULT: %(default)s]")
# Model-related arguments
parser.add_argument("-r", "--retrain", type=str, default="", dest="r", required=False, help="If retraining a saved model, provide the model name. [DEFAULT: %(default)s]")
parser.add_argument("-hy", "--hyper", type=str, default="", dest="hy", required=False, help="If using non-standard naming for hyperparameters, provide the file name. [DEFAULT: %(default)s]")
# Misc. arguments
parser.add_argument("-save", type=str, default="none", required=False, help="Change how often model outputs when training are saved. [DEFAULT: %(default)s]", choices=["all", "periodic", "best", "none"])
parser.add_argument("-models", action="store_true",  help="Save models according to output saves. [DEFAULT: %(default)s]")
parser.add_argument("-y", "--yaml", type=str, default="generate_diffusion", dest="y", required=False, help="Name of yaml file storing relevant information. [DEFAULT: %(default)s]")

args = parser.parse_args()
yaml = yaml.safe_load(open(f"Helpers/{args.y}.yaml", "r"))  # Stores relevant information that is less transient than above


#===================================================================================================
#### General setup ####
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
hyper = None
synth_para = None 
chain_para = None

# Save name, depending on mode of operation
save_name = f"{yaml['save_name']}_diffusion_{args.mode}"
if args.mode == "train" and args.org == "chain":
    save_name = f"{save_name}_chain{args.rp.split('_')[1]}"
else:
    save_name = f"{save_name}_single"
print(f"Working on {save_name}!")        # Print save_name
print("===============================================================================================")

# Create dataset
mid_shape = floor(floor(yaml["data"]["dim"]/2)/2)
dataset = []
max_class = 1
if yaml["data"]["prefix"] == "mnist":
    print("\tUsing MNIST dataset -- no chain available!")
    image_transforms = transforms.Compose([transforms.ToTensor(), transforms.Resize((yaml["data"]["dim"], yaml["data"]["dim"])), custom_transforms.MinMaxNorm()])
    dataset = MNIST(f"{yaml['folders']['data']}", download=True, train=True, transform=image_transforms)
    if yaml["data"]["class"] >= 10:         # Use all classes
        yaml["data"]["class"] = 10
        max_class = 10
    elif yaml["data"]["class"] < 10:      # Select specified number
        idx = dataset.targets==0
        idx |= dataset.targets==yaml["data"]["class"] 
        dataset.targets = dataset.targets[idx]
        dataset.data = dataset.data[idx]
    if yaml["data"]["max_data"] < len(dataset):
        if yaml["replicable"]:
            generator = torch.Generator().manual_seed(yaml["seed"])
        else:
            generator = torch.Generator().manual_seed(np.randint(0, 2**32-1))
        dataset, _ = random_split(dataset, [yaml["data"]["max_data"], len(dataset)-yaml["data"]["max_data"]], generator=generator)
elif "polyfit" in yaml["data"]["prefix"]:
    synth_dict = np.load(f"{yaml['folders']['data']}/{yaml['data']['prefix']}.npy", allow_pickle=True).item()
    try:
        synth_para = np.load(f"{yaml['folders']['load']}/{yaml['data']['paras']}.npy", allow_pickle=True).item()
    except:
        synth_para = {"polyfit": 3, "width_mean": 2.5, "width_sd": 0, "width_k": 0, \
                        "modi": 1, "intensity_mean": 0, "intensity_sd": 0, "intensity_k": 0, \
                        "gauss_filter": False, "gauss_kernel": 0}
    if "chain" in save_name:
        chain_para = np.load(f"{yaml['folders']['load']}/{args.rp}.npy", allow_pickle=True).item()
    train_idx, _ = spec_data.split_sets(len(synth_dict), [yaml["training"]["train_split"]], replicable=yaml["replicable"], seed=yaml["seed"])
    if args.custclass != "":
        cust_class = pd.read_csv(f"{yaml['folders']['load']}/{args.custclass}.csv")
    count = 0
    for idx in train_idx:
        synth_data, synth_class = synth.make_synth(synth_dict, synth_para["polyfit"], \
                                        [synth_para["width_mean"], synth_para["width_sd"], synth_para["width_k"]], \
                                        [synth_para["intensity_mean"], synth_para["intensity_sd"], synth_para["intensity_k"]], \
                                        ret_class=yaml["data"]["class"], idx_list=[idx])
        if synth_data == []:
            continue
        idx_class = synth_class[0] if args.custclass == "" else cust_class.iloc[idx] 
        if "chain" in save_name:
            chain_data, _ = synth.make_synth(synth_dict, chain_para["polyfit"], \
                                        [chain_para["width_mean"], chain_para["width_sd"], chain_para["width_k"]], \
                                        [chain_para["intensity_mean"], chain_para["intensity_sd"], chain_para["intensity_k"]], \
                                        gauss_filter=chain_para["gauss_filter"], gauss_kernel=chain_para["gauss_kernel"], \
                                        ret_class=yaml["data"]["class"], idx_list=[idx])
            if chain_data == []:
                continue
            dataset.append([synth.overlay_synth(chain_data[0], data_shape=yaml["data"]["dim"]), synth.overlay_synth(synth_data[0], data_shape=yaml["data"]["dim"]), idx_class])
        else:
            dataset.append([synth.overlay_synth(synth_data[0], data_shape=yaml["data"]["dim"]), idx_class])
        max_class = max(max_class, (idx_class+1))
        count += 1
        if count >= yaml["data"]["max_data"]:
            break
dataloader = DataLoader(dataset, batch_size=yaml["training"]["batch_size"], shuffle=True, num_workers=5)
del dataset


#===================================================================================================
#### HYPERPARAMETER TUNING ####
if args.mode == "tune":
    distr_ranges = {"dropout": (0.1, 0.5), "lr": (1e-7, 1e-1), "beta1": (0.000001, 0.01), "beta2": (0.000001, 0.1), "n_T": (100, 10000), "features": (4, 7)}
    distr_dict = {"dropout": optuna.distributions.FloatDistribution(distr_ranges["dropout"][0], distr_ranges["dropout"][1]), \
                 "lr": optuna.distributions.FloatDistribution(distr_ranges["lr"][0], distr_ranges["lr"][1]), \
                 "beta1": optuna.distributions.FloatDistribution(distr_ranges["beta1"][0], distr_ranges["beta1"][1]), \
                 "beta2": optuna.distributions.FloatDistribution(distr_ranges["beta2"][0], distr_ranges["beta2"][1]), \
                 "n_T": optuna.distributions.IntDistribution(distr_ranges["n_T"][0], distr_ranges["n_T"][1]), \
                 "features": optuna.distributions.IntDistribution(distr_ranges["features"][0], distr_ranges["features"][1])}   
    #------------------ Dictionaries to retain best hyperparameters, keyed by val_accuracy
    try:        # Try finding in "results_folder" 
        hyp_perf = np.load(f"{yaml['folders']['load']}/{save_name}.npy", allow_pickle=True).item()
    except:
        hyp_perf = dict()
    if len(hyp_perf.keys()) >= yaml["training"]["max_tune"]:
        exit()
    #------------------ Objective function
    def objective(trial, distr_ranges, max_class, mid_shape, prune=True):
        global dataloader
        global device 
        global yaml
        #------------------ Generate suggestions
        dropout = trial.suggest_float("dropout", distr_ranges["dropout"][0], distr_ranges["dropout"][1], log=True)
        lr = trial.suggest_float("lr", distr_ranges["lr"][0], distr_ranges["lr"][1], log=True)
        beta1 = trial.suggest_float("beta1", distr_ranges["beta1"][0], distr_ranges["beta1"][1], log=True)
        beta2 = trial.suggest_float("beta2", beta1*2, distr_ranges["beta2"][1], log=True)
        n_T = trial.suggest_int("n_T", distr_ranges["n_T"][0], distr_ranges["n_T"][1], step=50)
        features = trial.suggest_int("features", distr_ranges["features"][0], distr_ranges["features"][1], log=True)
        #------------------ Create model, criterion, and optimiser
        ddpm = diffusions.DDPM(nn_model=diffusions.ContextUnet(in_channels=1, out_channels=1, n_feat=2**features, n_classes=max_class, mid_shape=mid_shape), \
                        betas=(beta1, beta2), n_T=n_T, device=device, drop_prob=dropout)
        ddpm.to(device)
        ddpm.train()
        optim = torch.optim.Adam(ddpm.parameters(), lr)
        #------------------ Start training
        best_loss = 1000
        best_at = 0
        no_change = 0
        for epoch in range(yaml["training"]["max_epochs"]):
            optim.param_groups[0]["lr"] = lr * (1 - epoch/yaml["training"]["max_epochs"])     # Linear lrate decay
            loss_ema = None
            for _, data in enumerate(dataloader):
                optim.zero_grad()
                x_in = data[0].to(device, dtype=torch.float)
                c = data[-1].to(device)
                loss = ddpm(x_in, None, c)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = yaml["training"]["ema_coeff"]*loss_ema + (1-yaml["training"]["ema_coeff"])*loss.item()
                optim.step()
            if (epoch+1) % yaml["training"]["print_status"] == 0:
                print(f"EP {epoch+1} (save {best_at}) // EMA Loss: {loss_ema:.4f}")
            if loss_ema + yaml["training"]["improve_margin"] < best_loss:
                best_loss = loss_ema
                best_at = epoch+1
                no_change = 0
            else:
                no_change += 1
            trial.report(best_loss, epoch)
            if no_change >= yaml["training"]["stop_after"]:
                print(f"\n\tEarly stop at epoch {epoch+1}...")
                break
            if prune and trial.should_prune():
                raise optuna.TrialPruned()
        return best_loss
    #------------------ Tune model
    study = optuna.create_study(direction="minimize", study_name=save_name)
    if len(hyp_perf.keys()) == 0:       # Starting point
        start_point = dict()
        for k in distr_ranges.keys():
            start_point[k] = yaml["defaults"][k]
        study.enqueue_trial(start_point)
    else:       # Queue up best result from last run
        for k, v in hyp_perf.items():
            params = {name: chosen for name, chosen in v.items() if name != "trial_value"}
            trial = optuna.trial.create_trial(params=params, distributions=distr_dict, value=v["trial_value"])
            study.add_trial(trial)
    print(f"Number of trials done is {len(study.trials)}")
    study.optimize(lambda trial:objective(trial, distr_ranges, max_class, mid_shape), n_trials=yaml["training"]["tune_trials"])
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
    # Hyperparameters
    if args.hy == "":
        hyp_name = save_name.replace("train", "tune")
        starter = hyp_name.split("_")[0]
        if "ver" in starter:
            new_starter = starter.split("ver")[0]
            hyp_name = hyp_name.replace(starter, new_starter)
    else:
        hyp_name = args.hy
    try:
        hyper = np.load(f"{yaml['folders']['load']}/{hyp_name}.npy", allow_pickle=True).item()
        print(f"\tFound {hyp_name}!")
    except:
        hyper = yaml["defaults"]
        print("\tTraining using default hyperparameters!")
    # Get real data samples for later viewing purposes
    n_cols = yaml["training"]["examples"]
    n_rows = max_class
    classes = []
    for _ in range(n_cols):
        classes.extend([i for i in range(max_class)])
    sample_y = torch.tensor(np.array(classes), dtype=torch.int64, device=device)
    sample_x_real = torch.Tensor(n_rows*n_cols, 1, yaml["data"]["dim"], yaml["data"]["dim"]).to(device)
    if args.org == "chain":
        sample_x_cond = torch.Tensor(n_rows*n_cols, 1, yaml["data"]["dim"], yaml["data"]["dim"]).to(device)
    elif args.org == "single":
        sample_x_cond = None
    idx = 0
    for _, data in enumerate(dataloader):
        for b in range(data[0].shape[0]):
            if data[-1][b] == sample_y[idx]:
                sample_x_real[idx] = data[0][b]
                if args.org == "chain":
                    sample_x_cond[idx] = data[1][b]
                idx += 1
            if idx >= n_cols*n_rows:
                break
        if idx >= n_cols*n_rows:
            break    

    if args.org == "chain":
        ddpm = diffusions.DDPM(nn_model=diffusions.ContextUnet(in_channels=2, out_channels=1, n_feat=2**hyper["features"], n_classes=max_class, mid_shape=mid_shape), \
                    betas=(hyper["beta1"], hyper["beta2"]), n_T=hyper["n_T"], device=device, drop_prob=hyper["dropout"])
    elif args.org == "single":
        ddpm = diffusions.DDPM(nn_model=diffusions.ContextUnet(in_channels=1, out_channels=1, n_feat=2**hyper["features"], n_classes=max_class, mid_shape=mid_shape), \
                    betas=(hyper["beta1"], hyper["beta2"]), n_T=hyper["n_T"], device=device, drop_prob=hyper["dropout"])
    ddpm.to(device)
    if args.r != "":
        ddpm.nn_model.load_state_dict(torch.load(f"{yaml['folders']['load']}/{args.r}.pt"))
    optim = torch.optim.Adam(ddpm.parameters(), lr=hyper["lr"])

    # Training
    best_loss = 1000
    best_at = 0
    no_change = 0
    losses = []
    save_images = {"cond": (sample_x_cond.cpu() if sample_x_cond is not None else None, sample_y.cpu())}
    torch.save(ddpm.nn_model.state_dict(), f"{yaml['folders']['temp']}/best.pt")
    for epoch in range(yaml["training"]["max_epochs"]):
        ddpm.train()
        optim.param_groups[0]["lr"] = hyper["lr"] * (1 - epoch/yaml["training"]["max_epochs"])     # Linear lrate decay
        loss_ema = None
        for _, data in enumerate(dataloader):
            optim.zero_grad()
            x_in = data[0].to(device, dtype=torch.float)
            if "chain" in save_name:
                x_cond = data[1].to(device, dtype=torch.float)
            else:
                x_cond = None
            c = data[-1].to(device)
            loss = ddpm(x_in, x_cond, c)
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = yaml["training"]["ema_coeff"]*loss_ema + (1-yaml["training"]["ema_coeff"])*loss.item()
            optim.step()
        if (epoch+1) % yaml["training"]["print_status"] == 0:
            print(f"EP {epoch+1} (save {best_at}) // EMA Loss: {loss_ema:.4f}")
        losses.append(loss_ema)
        if loss_ema + yaml["training"]["improve_margin"] < best_loss:
            best_loss = loss_ema
            best_at = epoch 
            no_change = 0
            torch.save(ddpm.nn_model.state_dict(), f"{yaml['folders']['temp']}/best.pt")
        else:
            no_change += 1
        if args.save == "all" or \
            (args.save == "periodic" and (epoch+1)%yaml["training"]["save_every"]==0) or \
            (args.save == "best" and no_change == 0):
            ddpm.eval()
            with torch.no_grad():
                x_gen, x_gen_store = ddpm.sample(sample_x_cond, sample_y, (1, yaml["data"]["dim"], yaml["data"]["dim"]), device)
                save_images[epoch] = (x_gen, x_gen_store)
            torch.save(ddpm.nn_model.state_dict(), f"{yaml['folders']['results']}/{save_name}_epoch{epoch+1}.pt")
        if no_change >= yaml["training"]["stop_after"]:
            print(f"\n\tEarly stop at epoch {epoch+1}...")
            break

    best_at += 1
    print(f"\tBest model at {best_at}!\n\n") 
    res_dict = {k: v for k, v in vars(args).items()}
    res_dict["mid_shape"] = mid_shape
    res_dict["datapoints"] = len(dataloader.dataset)
    res_dict["losses"] = losses 
    res_dict["best"] = (best_loss, best_at)
    res_dict["save_images"] = save_images
    np.save(f"{yaml['folders']['results']}/{save_name}.npy", res_dict, allow_pickle=True)
    ddpm.nn_model.load_state_dict(torch.load(f"{yaml['folders']['temp']}/best.pt"))
    torch.save(ddpm.nn_model.state_dict(), f"{yaml['folders']['results']}/{save_name}.pt")