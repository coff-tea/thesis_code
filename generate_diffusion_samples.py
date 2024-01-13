"""
Author: Xi Lu
File: generate_diffusion_samples.py
Description: Used to generate image samples from a trained DDPM.
""" 



#===================================================================================================
import sys
import os
import random
import argparse
import yaml
import numpy as np
import torch
from math import floor

from Helpers import *



#===============================================================================================
#### Argparser and json####
parser = argparse.ArgumentParser()
# Mandatory arguments
parser.add_argument("file_name", type=str, help="Which model file to use.")
parser.add_argument("hyper", type=str, help="Which hyperparameter file to use.")
parser.add_argument("org", type=str, help="Specify structure of diffusion model(s) needed.", choices=["single", "chain"])
parser.add_argument("num_sample", type=int, help="Number of samples to generate per class type.")
# Data arguments
parser.add_argument("-c", "--classes", type=int, default=1, dest="c", required=False, help="Number of classes total. [DEFAULT: %(default)s]")
parser.add_argument("-cd", "--cond", type=str, default="", dest="cd", required=False, help="If provided for chain type, name of previously generated that can be used. [DEFAULT: %(default)s]")
# Misc. arguments
parser.add_argument("-y", "--yaml", type=str, default="generate_diffusion", dest="y", required=False, help="Name of yaml file storing relevant information. [DEFAULT: %(default)s]")

args = parser.parse_args()
yaml = yaml.safe_load(open(f"Helpers/{args.y}.yaml", "r"))  # Stores relevant information that is less transient than above


#===============================================================================================
#### General setup ####
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#--------- Load pre-trained model parameters
# Hyperparameters
try:
    hyper = np.load(f"{yaml['folders']['load']}/{args.hyper}.npy", allow_pickle=True).item()
    print(f"\tFound {args.hyper}!")
except:
    hyper = yaml["defaults"]
    print("\tTraining using default hyperparameters!")
#--------- Model
mid_shape = floor(floor(yaml["data"]["dim"]/2)/2)
if args.org == "chain":
    ddpm = diffusions.DDPM(nn_model=diffusions.ContextUnet(in_channels=2, out_channels=1, n_feat=2**hyper["features"], n_classes=args.c, mid_shape=mid_shape), \
                betas=(hyper["beta1"], hyper["beta2"]), n_T=hyper["n_T"], device=device, drop_prob=hyper["dropout"])
elif args.org == "single":
    ddpm = diffusions.DDPM(nn_model=diffusions.ContextUnet(in_channels=1, out_channels=1, n_feat=2**hyper["features"], n_classes=args.c, mid_shape=mid_shape), \
                betas=(hyper["beta1"], hyper["beta2"]), n_T=hyper["n_T"], device=device, drop_prob=hyper["dropout"])
ddpm.nn_model.load_state_dict(torch.load(f"{yaml['folders']['load']}/{args.file_name}.pt"))


#===============================================================================================
#### Load data ####
X_cond = None
if args.org == "chain":
    X_cond = np.load(f"{yaml['folders']['load']}/{args.cd}.npy", allow_pickle=True).item()


#===============================================================================================
#### Generate samples ####   
save_dict = {c: [] for c in range(args.c)}
actual_sample = args.num_sample
#------------------ Start training
for c in range(args.c):
    sample_y = torch.tensor(np.array([c]), dtype=torch.int64, device=device)
    num_sample = args.num_sample if X_cond is None else min(args.num_sample, X_cond.shape[0])
    for n in range(num_sample):
        print(f"Sample {c} # {n}")
        sample_x_cond = None
        if args.org == "chain":
            if args.num_sample*c+n >= len(X_cond[c]):
                actual_sample = len(X_cond[c])
                continue
            sample_x_cond = X_cond[c][args.num_sample*c+n]
        ddpm.eval()
        with torch.no_grad():
            x_gen, _ = ddpm.sample(sample_x_cond, sample_y, (1, yaml["data"]["dim"], yaml["data"]["dim"]), device, store=False)
        save_dict[c].append(x_gen.cpu())


#===============================================================================================
#### Save ####    
np.save(f"{yaml['folders']['results']}/{args.file_name}_sample{actual_sample}.npy", save_dict, allow_pickle=True) 