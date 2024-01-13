"""
Author: Xi Lu
File: synth_iterative.py
Description: Used to perform iterative detection starting from a point of negative and synthetic samples.
""" 



#===================================================================================================
import sys
import os
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import confusion_matrix

from Helpers import *



#===================================================================================================
#### Argparser####
parser = argparse.ArgumentParser()
# Mandatory arguments
parser.add_argument("origin", type=str, help="Name of parameter file.")
parser.add_argument("direction", type=str, help="Direction of optimisation.")
parser.add_argument("kth_best", type=int, help="Which key to use from source.")
parser.add_argument("model_name", type=str, help="Model to be used.", choices=["simple", "vgg19bn", "res152", "dense161"])
# Data arguments
parser.add_argument("-syn_name", type=str, default="", required=False, help="If using generated samples, provide name of file. [DEFAULT: %(default)s]")
parser.add_argument("-hyp_name", type=str, default="", required=False, \
                    help="If using hyperparameters, provide the full file name. [DEFAULT: %(default)s]")
parser.add_argument("-bh_name", type=str, default="", required=False, \
                    help="If using bellhop masks, provide the full file name. [DEFAULT: %(default)s]")
parser.add_argument("-skip_unn", action="store_true", help="When incorporating blind data, do not include mislabelled negatives. [DEFAULT: %(default)s]")
parser.add_argument("-threshold", type=float, default=0.5, help="When output exceeeds this value, consider detected. [DEFAULT: %(default)s]")
parser.add_argument("-voting", type=str, default="", required=False, help="If using voting, provide the agreed-upon indices. [DEFAULT: %(default)s]")
# Misc. arguments
parser.add_argument("-y", "--yaml", type=str, default="synth_iterative", dest="y", required=False, help="Name of yaml file storing relevant information. [DEFAULT: %(default)s]")

args = parser.parse_args()
yaml = yaml.safe_load(open(f"Helpers/{args.y}.yaml", "r"))  # Stores relevant information that is less transient than above


#===================================================================================================
""" FUNCTION: train_epoch
Train through batches in a dataloader using global parameters of the file. Returns tuple of epoch results (loss, accuracy (%), false alarm (%), missed detection (%)).

Args:
    - dataloader (torch.utils.data.DataLoader object): Dataloader to be trained using.
    - threshold (float): Above this value, predict 1 for presence of whistle.
"""
def train_epoch(dataloader, threshold):
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
        pred[torch.sigmoid(output) >= threshold] = 1.
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
    - threshold (float): Above this value, predict 1 for presence of whistle.
"""
def eval_dataloader(dataloader, threshold):
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
            pred[torch.sigmoid(output) >= threshold] = 1.
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
""" FUNCTION: find_positives
Evaluate a list of spectrograms and produce the predicted labels.

Args:
    - dataloader (torch.utils.data.DataLoader object): Dataloader to be evaluated.
    - threshold (float): Above this value, predict 1 for presence of whistle.
"""
def find_positives(dataloader, threshold):
    global model
    global device

    y_pred = []
    with torch.no_grad():
        model.eval()
        for _, data in enumerate(dataloader):
            x = data[0].to(device)
            for s in x:
                if torch.mean(s).isnan():
                    print(s)
                    print(torch.max(s))
                    print(torch.min(s))

            output = model(x.float())
            pred = torch.zeros_like(output)
            pred[torch.sigmoid(output) >= threshold] = 1.
            y_pred.extend([p.item() for p in pred])
    return y_pred


#===================================================================================================
#### General setup ####

#--------- Load parameters
opt_paras = None
try:
    opt_paras = np.load(f"{yaml['folders']['load']}/{args.origin}.npy", allow_pickle=True).item()
    print(f"Loading from {args.origin}!")
except:
    print(f"Load file {args.origin} non-existent...")
    exit()
sort_paras = pd.DataFrame(columns=["key", "val"])
for k, v in opt_paras.items():
    row = {"key": k, "val": v["trial_value"]}
    sort_paras = sort_paras.append(row, ignore_index=True)
if args.direction == "minimise":
    sort_paras = sort_paras.sort_values(by=["val"])
else:
    sort_paras = sort_paras.sort_values(by=["val"], ascending=False)
use_paras = opt_paras[int(sort_paras.iloc[args.kth_best-1]["key"])]
print(f"Use score {args.kth_best}: {use_paras['trial_value']}")

#--------- Save name
uniq = 1
save_spec = "synthiter" 
save_name = f"{yaml['save_name']}_{args.origin}_{save_spec}_k{args.kth_best}-{uniq}_{args.model_name}"
found_flag = True 
while found_flag:
    try:
        os.open(f"{yaml['folders']['results']}/{save_name}.npy", os.O_RDONLY)
        uniq += 1
        save_name = f"{yaml['save_name']}_{args.origin}_{save_spec}_k{args.kth_best}-{uniq}_{args.model_name}"
    except:
        found_flag = False
print("\n\nWorking on {}".format(save_name))

#--------- Hyperparameters and model settings
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = detectors.make_detector(args.model_name, 1, yaml["data"]["dim"])
model.to(device)
torch.save(model.state_dict(), f"{yaml['folders']['temp']}/best.pt")
class_weights = torch.ones(1)
class_weights[0] = hyper["pos"]
crit = nn.BCEWithLogitsLoss(pos_weight=class_weights.to(device))
opt = optim.Adam(model.parameters(), lr=hyper["lr"], betas=(hyper["beta1"], hyper["beta2"]), weight_decay=hyper["decay"])


#===================================================================================================
#### Load data ####
if yaml['data']['channel'] > 0:
    data_name_prefix = f"{yaml['folders']['data']}/{yaml['data']['prefix']}_X{yaml['data']['channel']}"
else:
    data_name_prefix = f"{yaml['folders']['data']}/{yaml['data']['prefix']}_X"

#--------- Label pos/unn data and take all labelled as positive
pos_frames = []
pos_samples_used = []
unn_frames = []
unn_samples_used = []
prev_res = None
if uniq > 1:        # Not the first run, load previous model
    prev_name = f"{yaml['save_name']}_{args.origin}_synthiter_k{args.kth_best}-{uniq-1}_{args.model_name}"
    model.load_state_dict(torch.load(f"{yaml['folders']['load']}/{prev_name}.pt"))
    if args.voting == "":
        prev_res = np.load(f"{yaml['folders']['load']}/{prev_name}.npy", allow_pickle=True).item()
    else:
        prev_res = np.load(f"{yaml['folders']['load']}/{args.voting}.npy", allow_pickle=True).item()
    #--------- Positive data
    pos_samples_used = prev_res["pos_samples_found"]["train"]
    X = spec_data.load_data(np.load(f"{data_name_prefix}pos.npy"))
    temp_train, temp_val, temp_test = tuple(spec_data.split_sets(len(X), [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]))
    data = [[np.array(synth.overlay_synth(X[i])), 1] for i in range(len(X))]
    dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=yaml["training"]["batch_size"])
    pred = find_positives(dataloader, args.threshold)
    for p in range(len(pred)):
        if pred[p] == 1 and p not in pos_samples_used and p in prev_res["unseen_splits"]["pos"][0]:
            pos_samples_used.append(p)
    del data 
    del dataloader
    pos_frames = [synth.overlay_synth(X[i]) for i in pos_samples_used]
    del X
    del pred
    #--------- Unseen negative data
    unn_samples_used = prev_res["unn_samples_found"]["train"]
    X = spec_data.load_data(np.load(f"{data_name_prefix}unn.npy"))
    temp_train, temp_val, temp_test = tuple(spec_data.split_sets(len(X), [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]))
    data = [[np.array(synth.overlay_synth(X[i])), 0] for i in range(len(X))]
    dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=yaml["training"]["batch_size"])
    pred = find_positives(dataloader, args.threshold)
    for p in range(len(pred)):
        if pred[p] == 1 and p not in unn_samples_used and p in prev_res["unseen_splits"]["unn"][0]:
            unn_samples_used.append(p)
    del data 
    del dataloader
    if not args.skip_unn:
        unn_frames = [synth.overlay_synth(X[i]) for i in unn_samples_used]
    del X
    del pred
    model.load_state_dict(torch.load(f"{yaml['folders']['temp']}/best.pt"))
print(f"{len(pos_samples_used)} pos and {len(unn_samples_used)} unn samples detected")

#--------- Make synthetic data from synth and negative (non)
syn_frames = []
if args.syn_name == "":
    synth_dict = np.load(f"{yaml['folders']['data']}/{yaml['data']['synth']}.npy", allow_pickle=True).item()
    syn_data, _ = synth.make_synth(synth_dict, use_paras["polyfit"], [use_paras["width_mean"], use_paras["width_sd"], use_paras["width_k"]], \
                                [use_paras["intensity_mean"], use_paras["intensity_sd"], use_paras["intensity_k"]], \
                                gauss_filter=use_paras["gauss_filter"], gauss_kernel=use_paras["gauss_kernel"], \
                                width_bounds=[yaml["paras"]["width_min"], yaml["paras"]["width_max"]])
    del synth_dict
    bh = None 
    if args.bh_name != "":
        bh = np.load(f"{yaml['folders']['data']}/{args.bh_name}.npy", allow_pickle=True)
    non_frames = spec_data.load_data(np.load(f"{data_name_prefix}non.npy"))
    non_frames = [synth.overlay_synth(non_frames[i]) for i in range(len(non_frames))]
    for i in range(min(len(non_frames), len(syn_data))):
        syn_temp = syn_data[i]
        if bh is not None:
            m = np.random.randint(bh.shape[0])
            syn_temp = synth.mask_synth(syn_temp, bh[m])
        syn_frames.append(synth.overlay_synth(non_frames[i], add=syn_temp, modi=use_paras["modi"]))
    del bh
    del non_frames
    del syn_data
else:
    syn_file = np.load(f"{yaml['folders']['data']}/{args.syn_name}.npy", allow_pickle=True)
    syn_data = [synth.overlay_synth(syn_file[i][0]) for i in range(syn_file.shape[0])]
    del syn_file
    non_frames = spec_data.load_data(np.load(f"{data_name_prefix}non.npy"))
    non_frames = [synth.overlay_synth(non_frames[i]) for i in range(len(non_frames))]
    syn_frames = [synth.overlay_synth(non_frames[i], add=syn_data[i], modi=use_paras["modi"]) for i in range(min(len(syn_data), len(non_frames)))]
    del syn_data 
    del non_frames
    
#--------- Get negative (neg) data
neg_frames = spec_data.load_data(np.load(f"{data_name_prefix}neg.npy"))
neg_frames = [synth.overlay_synth(neg_frames[i]) for i in range(len(neg_frames))]

#--------- Take away extra data
min_num_samples = min(len(neg_frames), len(syn_frames)+len(pos_frames)+len(unn_frames))
neg_frames = neg_frames[:min_num_samples]
syn_frames = syn_frames[:(min_num_samples-len(pos_frames)-len(unn_frames))]
print(f"Using {len(neg_frames)}, {len(syn_frames)}, {len(pos_frames)}, {len(unn_frames)} neg/syn/pos/unn samples")

#--------- Split into different sets
idx_dict = dict()
if len(neg_frames) < 10:
    idx_dict["neg"] = ([i for i in range(len(neg_frames))], [], [])
else:
    idx_dict["neg"] = tuple(spec_data.split_sets(len(neg_frames), [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]+1))
if len(pos_frames) < 10:
    idx_dict["pos"] = ([i for i in range(len(pos_frames))], [], [])
else:
    idx_dict["pos"] = tuple(spec_data.split_sets(len(pos_frames), [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]))
if len(unn_frames) < 10:
    idx_dict["unn"] = ([i for i in range(len(unn_frames))], [], [])
else:
    idx_dict["unn"] = tuple(spec_data.split_sets(len(unn_frames), [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]+1))
if len(syn_frames) < 10:
    idx_dict["syn"] = ([i for i in range(len(syn_frames))], [], [])
else:
    idx_dict["syn"] = tuple(spec_data.split_sets(len(syn_frames), [yaml["training"]["train_split"], 0.5], replicable=yaml["replicable"], seed=yaml["seed"]+2))

#--------- Make dataloaders
train_data = [[np.array(neg_frames[i]), 0] for i in idx_dict["neg"][0]]
train_data.extend([[np.array(pos_frames[i]), 1] for i in idx_dict["pos"][0]])
train_data.extend([[np.array(unn_frames[i]), 1] for i in idx_dict["unn"][0]])
train_data.extend([[np.array(syn_frames[i]), 1] for i in idx_dict["syn"][0]])
trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=yaml["training"]["batch_size"])
del train_data
test_data = [[np.array(neg_frames[i]), 0] for i in idx_dict["neg"][1]]
test_data.extend([[np.array(pos_frames[i]), 1] for i in idx_dict["pos"][1]])
test_data.extend([[np.array(unn_frames[i]), 1] for i in idx_dict["unn"][1]])
test_data.extend([[np.array(syn_frames[i]), 1] for i in idx_dict["syn"][1]])
testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=yaml["training"]["batch_size"])
del test_data
val_data = [[np.array(neg_frames[i]), 0] for i in idx_dict["neg"][2]]
val_data.extend([[np.array(pos_frames[i]), 1] for i in idx_dict["pos"][2]])
val_data.extend([[np.array(unn_frames[i]), 1] for i in idx_dict["unn"][2]])
val_data.extend([[np.array(syn_frames[i]), 1] for i in idx_dict["syn"][2]])
valloader = torch.utils.data.DataLoader(val_data, shuffle=False, batch_size=yaml["training"]["batch_size"])
del val_data
print(f"Using {len(trainloader.dataset)}, {len(testloader.dataset)}, {len(valloader.dataset)} t/t/v samples")
del syn_frames 
del pos_frames 
del unn_frames 
del neg_frames


#===================================================================================================
#### Main code ####
train_losses = []
train_accs = []
val_losses = []
val_accs = []
no_change = 0
stop_after = yaml["training"]["stop_after"]
if args.model_name == "simple":     # Smaller model, train for longer
    stop_after *= 2
stopped_epoch = yaml["training"]["max_epochs"]
best_vloss, best_vacc, _, _ = eval_dataloader(valloader, 0.5)
best_at = 0
for epoch in range(yaml["training"]["max_epochs"]):
    ept_loss, ept_acc, _, _ = train_epoch(trainloader, 0.5)
    epv_loss, epv_acc, _, _ = eval_dataloader(valloader, 0.5)
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
    if (epoch+1) % yaml["training"]["print_status"] == 0:    
        print(f"\t\t\tOn epoch {epoch+1}, current is training {ept_loss:.4f}/{ept_acc:.2f} and validation {epv_loss:.4f}/{epv_acc:.2f}. Best is {best_vloss:.4f}/{best_vacc:.2f} in epoch {best_at}")
    if no_change >= stop_after:
        stopped_epoch = epoch
        break


#===================================================================================================
#### Save results ####
train_perf = {k: v for k, v in vars(args).items()}
model.load_state_dict(torch.load(f"{yaml['folders']['temp']}/best.pt"))
test_loss, test_acc, test_fa, test_md = eval_dataloader(testloader, 0.5)
print("\n-----Test results: ", test_loss, test_acc)
train_perf["hyper"] = hyper
train_perf["ttv"] = (len(trainloader.dataset), len(testloader.dataset), len(valloader.dataset))
train_perf["pos_samples_used"] = pos_samples_used
train_perf["unn_samples_used"] = unn_samples_used
train_perf["train_hist"] = (train_losses, train_accs)
train_perf["val_hist"] = (val_losses, val_accs)
train_perf["best"] = (best_vloss, best_vacc, best_at, stopped_epoch)
train_perf["test"] = (test_loss, test_acc, test_fa, test_md)
#--------- Check pos/unn data again
del trainloader 
del testloader
del valloader
pos_samples_found = {"train": [], "test": []}
unn_samples_found = {"train": [], "test": []}
#--------- Positive data
X = spec_data.load_data(np.load(f"{data_name_prefix}pos.npy"))
if uniq == 1:
    temp_train, temp_test = tuple(spec_data.split_sets(len(X), [yaml["training"]["train_split"]], replicable=yaml["replicable"], seed=yaml["seed"]))
    train_perf["unseen_splits"] = {"pos": (temp_train, temp_test)}
else:
    train_perf["unseen_splits"] = prev_res["unseen_splits"]
data = [[np.array(synth.overlay_synth(X[i])), 1] for i in range(len(X))]
del X
dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=yaml["training"]["batch_size"])
del data
pred = find_positives(dataloader, args.threshold)
for p in range(len(pred)):
    if pred[p] == 1:
        if p in train_perf["unseen_splits"]["pos"][0]:
            pos_samples_found["train"].append(p)
        else:
            pos_samples_found["test"].append(p)
for p in pos_samples_used:
    if p not in pos_samples_found["train"]:
        pos_samples_found["train"].append(p)
del dataloader
del pred
#--------- Unseen negative data
X = spec_data.load_data(np.load(f"{data_name_prefix}unn.npy"))
if uniq == 1: 
    temp_train, temp_test = tuple(spec_data.split_sets(len(X), [yaml["training"]["train_split"]], replicable=yaml["replicable"]+1, seed=yaml["seed"]))
    train_perf["unseen_splits"]["unn"] = (temp_train, temp_test)
data = [[np.array(synth.overlay_synth(X[i])), 0] for i in range(len(X))]
del X
dataloader = torch.utils.data.DataLoader(data, shuffle=False, batch_size=yaml["training"]["batch_size"])
del data
pred = find_positives(dataloader, args.threshold)
for p in range(len(pred)):
    if pred[p] == 1:
        if p in train_perf["unseen_splits"]["unn"][0]:
            unn_samples_found["train"].append(p)
        else:
            unn_samples_found["test"].append(p)
for p in unn_samples_used:
    if p not in unn_samples_found["train"]:
        unn_samples_found["train"].append(p)
del dataloader
del pred
print(f"\n-----After training, {len(pos_samples_found['test'])} pos and {len(unn_samples_found['test'])} unn test samples detected")
train_perf["pos_samples_found"] = pos_samples_found
train_perf["unn_samples_found"] = unn_samples_found
np.save(f"{yaml['folders']['results']}/{save_name}.npy", train_perf, allow_pickle=True)
torch.save(model.state_dict(), f"{yaml['folders']['results']}/{save_name}.pt")