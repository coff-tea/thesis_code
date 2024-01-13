"""
Author: Xi Lu
File: generate_gan.py
Description: Used to train GANs for purpose of dolphin whistle image generation. Code based on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
""" 



#===================================================================================================
import sys
import os
import random
import gc
import argparse
import yaml
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from mnist import MNIST

from Helpers import *



#===================================================================================================
#### Argparser and json####
parser = argparse.ArgumentParser()
# Mandatory arguments
parser.add_argument("model_size", type=str, help="Which size of model to use.", choices=["s", "m", "l"])
parser.add_argument("model_type", type=str, help="Which type of model to use.", choices=["dense", "dcgan"])
parser.add_argument("final_layer", type=str, help="What type of final layer for model to have.", choices=["tanh", "sigmoid", "relu"])
parser.add_argument("hyper", type=int, help="Choose hyperparameter set.")
parser.add_argument("pens", type=int, help="Choose penalty set.")
# Data-related arguments
parser.add_argument("-max_data", type=int, default=10000, required=False, help="Limit to this many samples of each type. [DEFAULT: %(default)s]")
# Model-related arguments
parser.add_argument("-l", "--latent", type=int, default=100, dest="l", required=False, help="Number of latent parameter inputs to generator. [DEFAULT: %(default)s]")
parser.add_argument("-r", "--recover", type=str, dest="r", required=False, help="File name for starting point. [DEFAULT: %(default)s]")
parser.add_argument("-i", "--index", type=int, dest="i", required=False, help="Accompanying args.r. [DEFAULT: %(default)s]")
# Training-related arguments
parser.add_argument("-p", "--penalties", type=str, nargs="+", default=[], dest="p", required=False, help="Additional penalty terms to add. [DEFAULT: %(default)s]")
parser.add_argument("-s", "--save", dest="s", action="store_true",  help="Save checkpointed gen/dis models. [DEFAULT: %(default)s]")
parser.add_argument("-dg", "--disgen", type=float, default=1, dest="dg", required=False, help="Multiplier to learning rate/decay of generator compared to discriminator. [DEFAULT: %(default)s]")
# Misc. arguments
parser.add_argument("-y", "--yaml", type=str, default="generate_gan", dest="y", required=False, help="Name of yaml file storing relevant information. [DEFAULT: %(default)s]")

args = parser.parse_args()
yaml = yaml.safe_load(open(f"Helpers/{args.y}.yaml", "r"))  # Stores relevant information that is less transient than above


#===================================================================================================
""" FUNCTION: get_hyps
Obtain a set of hyperparameter values based on an "indexing" value.

Args:
    - hyp_num (int): Indicate which set of hyperparameters to use.
"""
def get_hyps(hyp_num):
    lrs = [0.0005, 0.001]
    decay = [0, 0.001]
    b1s = [0.5]
    b2s = [0.99]
    smooths = [0, 0.2]
    count = 1
    for l in lrs:
        for s in smooths:
            for b1 in b1s:
                for b2 in b2s:
                    for d in decay:
                        if count == hyp_num:
                            return l, s, b1, b2, l*d
                    count += 1
    return 0.0005, 0., 0.5, 0.99, 0.
lr, smooth, beta1, beta2, decay = get_hyps(args.hyper)


#===================================================================================================
""" FUNCTION: get_pens
Obtain a set of penalty modifiers based on an "indexing" value.

Args:
    - pen_num (int): Indicate which set of penalty values to use.
"""
def get_pens(pen_num):
    sums = [0.01, 0.1, 1.]
    noisy = [0.01, 0.1, 1.]
    empty = [0.01, 0.1, 1.]
    count = 1
    for s in sums:
        for a in noisy:
            for e in empty:
                if count == pen_num:
                    return {"sum": s, "noisy": a, "empty": e}
                count += 1
    return {"sum": 0.1, "noisy": 0.1, "empty": 0.1}
pen_dict = get_pens(args.pens)


#===================================================================================================
""" FUNCTION: sum_penalty
Penalize based on the sum of non-minimum values.

Args:
    - frames (Tensor): Data
    - min_val (float): Minimum possible value in frames
    - tolerance (float): +/- this for minimum value [1e-3] 
"""
def sum_penalty(frames, min_val, tolerance=1e-3):
    pen = 0.
    for b in range(frames.shape[0]):
        pen += torch.sum(frames[b] + torch.min(frames[b])).item()
    return pen


#===================================================================================================
""" FUNCTION: noisy_penalty
Penalize based on the number of data points not equal to minimum value (penalize noisy).

Args:
    - frames (Tensor): Data
    - min_val (float): Minimum possible value in frames
    - tolerance (float): +/- this for minimum value [1e-3] 
"""
def noisy_penalty(frames, min_val, tolerance=1e-3):
    pen = 0.
    for b in range(frames.shape[0]):
        pen += torch.numel(frames[b][frames[b]>(min_val+tolerance)])
    return pen 


#===================================================================================================
""" FUNCTION: empty_penalty
Penalize based on the number of data points equal to the minimum value (penalize empty).

Args:
    - frames (Tensor): Data
    - min_val (float): Minimum possible value in frames
    - tolerance (float): +/- this for minimum value [1e-3] 
"""
def empty_penalty(frames, min_val, tolerance=1e-3):
    pen = 0.
    for b in range(frames.shape[0]):
        pen += torch.numel(frames[b][frames[b]<=(min_val+tolerance)])
    return pen 


#===================================================================================================
#### General setup ####
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#--------- Models
dis, gen = gans.make_models(args.model_size, args.model_type, args.final_layer, args.l)
dis.to(device)
gen.to(device)
#--------- Load model state_dicts() if given
if args.r is not None and args.i is not None:
    try:
        saved_models = np.load("{}/{}.npy".format(yaml["folders"]["models"], args.r), allow_pickle=True).item()
        dis.load_state_dict(saved_models["d_models"][args.i])
        gen.load_state_dict(saved_models["g_models"][args.i])
        print("\tLoaded from {}".format(args.r))
    except:
        print("\tFailed to load from {}".format(args.r))
#--------- Criterion
crit = nn.BCELoss()
#--------- Optimizers
optD = optim.Adam(dis.parameters(), lr=lr, betas=(beta1, beta2), weight_decay=decay)
optG = optim.Adam(gen.parameters(), lr=lr*args.dg, betas=(beta1, beta2), weight_decay=decay*args.dg)
#--------- Penalty modifiers
pen_mod = {k: pen_dict[k] for k in args.p}
#... Save name
uniq = 1
save_name = f"{yaml['save_name']}-{uniq}_gans_{args.model_type}_{args.final_layer}"
found_flag = True 
while found_flag:
    try:
        os.open(f"{yaml['folders']['results']}/{save_name}.npy", os.O_RDONLY)
        uniq += 1
        save_name = f"{yaml['save_name']}-{uniq}_gans_{args.model_type}_{args.final_layer}"
    except:
        found_flag = False
print("Working on {}".format(save_name))


#===================================================================================================
#### Load data ####
spat_dim = 28   # "s"
if args.model_size == "m":
    spat_dim = 112
elif args.model_size == "l":
    spat_dim = 224
X_data = []
max_classes = 10
min_val = 0.0
max_val = 1.0
if args.final_layer == "tanh":
    min_val = -1.0
if yaml["data"]["mode"] == "mnist":
    mndata = MNIST(f"{yaml['folders']['data']}/MNIST")
    mndata.gz = True    # Load from .gz files
    X_pos, y_pos = mndata.load_training()
    del mndata
    X_pos = X_pos[:min(args.max_data, len(X_pos))]
    y_pos = y_pos[:min(args.max_data, len(y_pos))]
    old_dim = int(np.sqrt(len(X_pos[0])))   # Stored as list, needs to be reshaped
    X_pos = np.array(X_pos).reshape(len(X_pos), old_dim, old_dim) 
    for i in range(len(X_pos)):
        X_data.append([synth.overlay_synth(X_pos[i].astype("float64"), data_shape=spat_dim, min_val=min_val, max_val=max_val), y_pos[i]])
    del X_pos
    del y_pos
elif yaml["data"]["mode"] == "synth":
    synth_dict = np.load(f"{yaml['folders']['data']}/{yaml['data']['synth']}.npy", allow_pickle=True).item() 
    synth_data, synth_class = synth.make_synth(synth_dict, yaml["data"]["polyfit"], \
                                        [yaml["data"]["w_mean"], yaml["data"]["w_sd"], yaml["data"]["w_k"]], \
                                        [yaml["data"]["i_mean"], yaml["data"]["i_sd"], yaml["data"]["i_k"]], \
                                        ret_class=yaml["data"]["class"])
    del synth_dict
    for i in range(len(synth_data)):
        if yaml["data"]["class"] > 0:
            class_i = synth_class[i]
        else:
            class_i = 0
        X_data.append([synth.overlay_synth(synth_data[i], data_shape=spat_dim, min_val=min_val, max_val=max_val), class_i])
    del synth_data
    del synth_class

            
#===================================================================================================
#### Dataloaders ####
pos_idx = spec_data.split_sets(len(X_data), [yaml["training"]["train_perc"]], seed=yaml["seed"])
train_data = []
for idx in pos_idx[0]:
    train_data.append(X_data[idx])
trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=yaml["training"]["batch_size"])
del train_data
test_data = []
for idx in pos_idx[1]:
    test_data.append(X_data[idx])
testloader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=yaml["training"]["batch_size"])
del test_data
del X_data
num_train = len(trainloader.dataset)
num_test = len(testloader.dataset)   


#===================================================================================================
#### Train ####   
# Save checkpoints of models and generator output at time
g_models = []
d_models = [] 
iso_list = []
pos_list = []
# Losses of models
g_losses = []
d_losses = []
# Percent of generated samples considered real per epoch (random) and for fixed latent vectors 
d_tricked = []
f_tricked = []
noise_shape = (yaml["training"]["batch_size"], args.l) if args.model_type == "dense" else \
              (yaml["training"]["batch_size"], args.l, 1, 1)
f_noise = torch.randn(*noise_shape, device=device)
#------------------ Start training
print(f"Training with {len(trainloader.dataset)} training and {len(testloader.dataset)} testing samples")
for epoch in range(yaml["training"]["max_epochs"]):   # For each epoch
    g_loss = 0.
    d_loss = 0.
    d_trick = 0
    f_trick = 0
    for i, data in enumerate(trainloader, 0):    # For each batch
        if pos_list == []:
            pos_list = data[0].cpu()
            plt.imshow(data[0][0][0].cpu(), cmap="Greys_r")
            plt.axis("off")
            plt.savefig(f"Temp/showmepos.png")
        ############################
        # (1) Update D network
        ###########################
        dis.zero_grad()
        #------ Train with all-real batch
        real_batch = data[0].to(device).float()
        b_size = real_batch.size(0)
        # Forward pass real batch through D
        output = dis(real_batch).view(-1)
        label = torch.full((b_size,), (1.0-smooth), dtype=torch.float, device=device)
        # Calculate gradients for D in backward pass
        errD_real = crit(output, label)
        errD_real.backward()
        #------ Train with all-fake batch
        noise_shape = (b_size, args.l) if args.model_type == "dense" else \
                      (b_size, args.l, 1, 1)
        noise = torch.randn(*noise_shape, device=device)
        # Generate fake image batch with G
        gen_batch = gen(noise)
        label.fill_(smooth)    
        # Classify all fake batch with D
        output = dis(gen_batch.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = crit(output, label)
        errD_fake.backward()
        optD.step()
        d_loss += (errD_real.mean().item() + errD_fake.mean().item())
        ############################
        # (2) Update G network
        ###########################
        gen.zero_grad()
        label.fill_(1.0-smooth)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = dis(gen_batch).view(-1)
        pred = torch.zeros_like(output)
        pred[output >= 0.5] = 1.
        d_trick += torch.sum(pred).item()
        errG = crit(output, label)
        if "sum" in args.p:
            pen = sum_penalty(gen_batch, min_val)
            errG += pen_mod["sum"] * pen / (spat_dim*spat_dim)
        if "noisy" in args.p:
            pen = noisy_penalty(gen_batch, min_val)
            errG += pen_mod["noisy"] * pen / (spat_dim*spat_dim)
        if "empty" in args.p:
            pen = empty_penalty(gen_batch, min_val)
            errG += pen_mod["empty"] * pen / (spat_dim*spat_dim)
        errG.backward()
        optG.step()  
        # Save Losses for plotting later
        g_loss += errG.mean().item()
    d_loss /= len(trainloader)
    g_loss /= len(trainloader)
    # Output training stats
    if (epoch+1) % yaml["training"]["print_status"] == 0:
        perf_string = f"Epoch {epoch+1}: Tricked {int(d_trick)}/{num_train} \n\t D {d_loss:.4f} and G {g_loss:.4f}"
        print(perf_string)
    # Save to lists as necessary
    d_losses.append(d_loss)
    g_losses.append(g_loss)
    d_tricked.append(d_trick/num_train*100)
    # Check how the generator is doing by saving G's output on f_noise
    with torch.no_grad():
        ## Train with all-fake batch
        gen_batch = gen(f_noise)
        output = dis(gen_batch).view(-1)
    f_trick = np.nan_to_num(torch.sum(torch.round(output)).item())  
    f_tricked.append(f_trick/yaml["training"]["batch_size"]*100)
    if (epoch+1) % yaml["training"]["checkpoint"] == 0:
        iso_list.append(gen_batch.cpu())
        # plt.imshow(gen_batch[0][0].cpu(), cmap="Greys_r")
        # plt.axis("off")
        # plt.savefig(f"Temp/showme{epoch+1}.png")
        # plt.imshow(gen_batch[1][0].cpu(), cmap="Greys_r")
        # plt.axis("off")
        # plt.savefig(f"Temp/showmee{epoch+1}.png")
        if args.s:
            d_models.append(copy.deepcopy(dis.state_dict()))
            g_models.append(copy.deepcopy(gen.state_dict()))


#===================================================================================================
#### Save ####    
save_dict = {k: v for k, v in vars(args).items()}
save_dict["g_losses"] = g_losses
save_dict["d_losses"] = d_losses
save_dict["d_tricked"] = d_tricked
save_dict["f_tricked"] = f_tricked
save_dict["iso_list"] = iso_list
save_dict["pos_list"] = pos_list
save_dict["f_noise"] = f_noise.cpu()
save_dict["hyper"] = (lr, smooth, beta1, beta2, decay)
save_dict["penalties"] = pen_mod
np.save("{}/{}.npy".format(yaml["folders"]["results"], save_name), save_dict, allow_pickle=True)     
if args.s:
    model_dict = dict()
    model_dict["d_models"] = d_models
    model_dict["g_models"] = g_models
    np.save("{}/{}.npy".format(yaml["folders"]["models"], save_name.replace("gans", "gansmodels")), model_dict, allow_pickle=True)