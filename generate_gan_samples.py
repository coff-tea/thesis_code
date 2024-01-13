"""
Author: Xi Lu
File: generate_diffusion_samples.py
Description: Used to generate image samples from a trained GAN.
""" 



#===================================================================================================
import sys
import os
import random
import argparse
import yaml
import numpy as np
import copy
import torch
import torch.nn as nn
from mnist import MNIST

from Helpers import *



#===================================================================================================
#### Argparser and json####
parser = argparse.ArgumentParser()
# Mandatory arguments
parser.add_argument("file_name", type=str, help="Which model file to use.")
parser.add_argument("num_sample", type=int, help="Number of samples to generate.")
# Model-related arguments
parser.add_argument("-model_idx", type=int, nargs="+", default=[-1], required=False, help="Choose this index in list of models. [DEFAULT: %(default)s]")
# Misc. arguments
parser.add_argument("-y", "--yaml", type=str, default="generate_gan", dest="y", required=False, help="Name of yaml file storing relevant information. [DEFAULT: %(default)s]")

args = parser.parse_args()
yaml = yaml.safe_load(open(f"Helpers/{args.y}.yaml", "r"))  # Stores relevant information that is less transient than above


#===================================================================================================
#### General setup ####
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
res_file = np.load(f"{yaml['folders']['load']}/{args.file_name}.npy", allow_pickle=True).item()
#--------- Models
dis, gen = gans.make_models(res_file["model_size"], res_file["model_type"], res_file["final_layer"], res_file["l"])
dis.to(device)
gen.to(device)
#--------- Load pre-trained model parameters
mdl_file = np.load(f"{yaml['folders']['load']}/{args.file_name.replace('gans', 'gansmodels')}.npy", allow_pickle=True).item()
#--------- Criterion
crit = nn.BCELoss()


#===================================================================================================
#### Load data ####
spat_dim = 28   # "s"
if res_file["model_size"] == "m":
    spat_dim = 112
elif res_file["model_size"] == "l":
    spat_dim = 224
X_data = []
min_val = 0.0
max_val = 1.0


#===================================================================================================
#### Generate samples ####   
fxd_images = dict()
fxd_output = dict()
gen_images = dict()
gen_output = dict()
noise_shape = (args.num_sample, res_file["l"]) if res_file["model_type"] == "dense" else \
              (args.num_sample, res_file["l"], 1, 1)
f_noise = torch.randn(*noise_shape, device=device)
#------------------ Start training
for i in args.model_idx:   # For each epoch
    dis.load_state_dict(mdl_file["d_models"][i])
    gen.load_state_dict(mdl_file["g_models"][i])
    with torch.no_grad():
        gen_batch = gen(f_noise)
        fxd_images[i] = gen_batch.cpu()
        output = dis(gen_batch).view(-1)
        pred = torch.zeros_like(output)
        pred[output >= 0.5] = 1.
        fxd_output[i] = pred.cpu()
        new_noise = torch.randn(*noise_shape, device=device)
        gen_batch = gen(new_noise)
        gen_images[i] = gen_batch.cpu()
        output = dis(gen_batch).view(-1)
        pred = torch.zeros_like(output)
        pred[output >= 0.5] = 1.
        gen_output[i] = pred.cpu()


#===================================================================================================
#### Save ####    
save_dict = {}
save_dict["fxd_images"] = fxd_images
save_dict["fxd_output"] = fxd_output
save_dict["gen_images"] = gen_images
save_dict["gen_output"] = gen_output
np.save(f"{yaml['folders']['results']}/{args.file_name}_sample{args.num_sample}.npy", save_dict, allow_pickle=True) 