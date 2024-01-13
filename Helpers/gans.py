"""
Author: Xi Lu
File: gan.py
Description: Image generation model using GAN methodology.
""" 



#===================================================================================================
import math
import torch
import torch.nn as nn
import torchvision.models as models



#===================================================================================================
""" FUNCTION: make_models
Create and return a GAN 

Args:
    - model_size (str): General size/architecture of model, "s"(mall)/"m"(edium)/"l"(arge)
    - model_type (str): Type of model, "dense"/"dcgan"
    - final_layer (str): Final layer, "tanh"/"relu"/"sigmoid"
    - latent (int): Size of noise vector for generator
    - leaky (float): Hyperparameter for leakyReLU [0.2]
    - drop_prob (float): Probability for dropout layers [0.3]
    - dis_sig (bool): Include final sigmoid layer for discriminator portion [True]
"""
def make_models(model_size, model_type, final_layer, latent, leaky=0.2, drop_prob=0.3, dis_sig=True):
    dis = None
    gen = None
    spat_dim = 28   # "s"
    if model_size == "m":
        spat_dim = 112
    elif model_size == "l":
        spat_dim = 224

    ####### DENSE GANS (all dense/linear layers)    
    if model_type == "dense":
        #........ Discriminator code
        class Discriminator(nn.Module):
            def __init__(self, spat_dim, nodes, leaky, drop, dis_sig):
                super(Discriminator, self).__init__()
                self.spat_dim = spat_dim
                layers = []
                x_in = spat_dim * spat_dim
                for i in range(3):
                    layers.append(nn.Linear(x_in, nodes[i]))
                    x_in = nodes[i]
                    layers.append(nn.LeakyReLU(leaky))
                    layers.append(nn.Dropout(drop))
                layers.append(nn.Linear(x_in, 1))
                if dis_sig:
                    layers.append(nn.Sigmoid())
                self.main = nn.Sequential(*layers)
            def forward(self, input):
                input = input.view(-1, self.spat_dim * self.spat_dim)
                return torch.nan_to_num(self.main(input))
        #........ Generator code
        class Generator(nn.Module):
            def __init__(self, spat_dim, nodes, latent, final_layer, leaky):
                super(Generator, self).__init__()
                self.latent = latent
                self.spat_dim = spat_dim
                layers = []
                x_in = self.latent
                for i in range(3):
                    layers.append(nn.Linear(x_in, nodes[i]))
                    x_in = nodes[i]
                    layers.append(nn.LeakyReLU(leaky))
                layers.append(nn.Linear(x_in, self.spat_dim * self.spat_dim))
                if final_layer == "tanh":
                    layers.append(nn.Tanh())
                elif final_layer == "relu":
                    layers.append(nn.ReLU())
                elif final_layer == "sigmoid":
                    layers.append(nn.Sigmoid())
                self.main = nn.Sequential(*layers)
            def forward(self, input):
                temp = self.main(input).view(-1, 1, self.spat_dim, self.spat_dim)
                return torch.nan_to_num(temp)
        #........ Make model
        model_paras = {"s": [2**10, 2**9, 2**8], "m": [2**13, 2**10, 2**7], "l": [2**13, 2**11, 2**9, 2**7]}
        # s -- 784 => 1024 => 512 => 256 
        # m -- 12544 => 8192 => 1024 => 128 
        # l -- 50176 => 8192 => 2048 => 512 => 128 
        dis = Discriminator(spat_dim, model_paras[model_size], leaky, drop_prob, dis_sig)
        gen = Generator(spat_dim, list(reversed(model_paras[model_size])), latent, final_layer, leaky)
    
    ####### DCGANS (all convolutional layers)
    elif model_type == "dcgan":
        #........ Discriminator code
        class Discriminator(nn.Module):
            def __init__(self, layers, leaky, dis_sig):
                super(Discriminator, self).__init__()
                clayers = []
                for i in range(len(layers)):
                    clayers.append(nn.Conv2d(*layers[i]))
                    if i < (len(layers)-1):
                        if i > 0:
                            clayers.append(nn.BatchNorm2d(layers[i][1]))
                        clayers.append(nn.LeakyReLU(negative_slope=leaky, inplace=False))
                if dis_sig:
                    clayers.append(nn.Sigmoid())
                self.conv = nn.Sequential(*clayers)
            def forward(self, input):
                return torch.nan_to_num(self.conv(input))
        #........ Generator code
        class Generator(nn.Module):
            def __init__(self, layers, leaky, latent, final_layer):   
                super(Generator, self).__init__() 
                clayers = []
                for i in range(len(layers)):
                    clayers.append(nn.ConvTranspose2d(layers[i][1], layers[i][0], *layers[i][2:]))
                    if i < (len(layers)-1):
                        if i > 0:
                            clayers.append(nn.BatchNorm2d(layers[i][0]))
                        clayers.append(nn.ReLU(inplace=False))
                if final_layer == "tanh":
                    clayers.append(nn.Tanh())
                elif final_layer == "relu":
                    clayers.append(nn.ReLU())
                elif final_layer == "sigmoid":
                    clayers.append(nn.Sigmoid())
                self.conv = nn.Sequential(*clayers)
            def forward(self, input):
                wh = self.conv(input)
                return torch.nan_to_num(wh)
        #........ Make model
        model_paras = {"s": [[1, 16, 5, 1, 0], [16, 32, 5, 1, 0], [32, 64, 5, 1, 0],\
                             [64, 128, 5, 1, 0], [128, 256, 5, 1, 0], [256, 1, 8, 1, 0]], \
                       "m": [[1, 16, 6, 2, 0], [16, 32, 4, 2, 0], [32, 64, 4, 2, 0],\
                             [64, 128, 5, 1, 0], [128, 256, 5, 1, 0], [256, 1, 4, 1, 0]], \
                       "l": [[1, 16, 6, 2, 0], [16, 64, 4, 2, 0], [64, 128, 4, 2, 0], \
                             [128, 256, 4, 2, 0], [256, 512, 4, 2, 0], [512, 1, 5, 1, 0]]}
        # s -- 28 => 24 => 20 => 16 => 12 => 8
        # m -- 112 => 54 => 26 => 12 => 8 => 4
        # l -- 224 => 110 => 54 => 26 => 12 => 5 
        dis = Discriminator(model_paras[model_size], leaky, dis_sig)
        model_paras[model_size][-1][1] = latent
        gen = Generator(list(reversed(model_paras[model_size])), leaky, latent, final_layer)
    return dis, gen

