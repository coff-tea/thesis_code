"""
Author: Xi Lu
File: diffusions.py
Description: Denoising Diffusion Probablistic Model (DDPM) for generating dolphin whistles.

!! PRIMARILY FROM https://github.com/TeaPearce/Conditional_Diffusion_MNIST/blob/main/script.py
[Comment retained from original source]
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487
"""



#===================================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
from typing import Dict, Tuple



#===================================================================================================
class ResidualConvBlock(nn.Module):
    """
    Convolutional block, also incorporates residuals if necessary.

    Args:
        - in_channels, out_channels (int): For convolutional layers, as indicated
        - is_res (bool): Use residuals [False]
    """

    def __init__(self, in_channels: int, out_channels: int, is_res: bool=False) -> None:
        super().__init__()
        self.same_channels = in_channels==out_channels
        self.is_res = is_res
        # Conv blocks don't change non-channel dimensions
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:  # Adds residual in case channels have increased
                out = x + x2
            else:
                out = x1 + x2 
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class UnetDown(nn.Module):
    """
    Used in Unet for downscaling image feature maps.

    Args:
        - in_channels, out_channels (int): For convolutional layers, as indicated
    """

    def __init__(self, in_channels, out_channels):
        super(UnetDown, self).__init__()
        layers = [ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)]    # Convolution, downscale by 1/2
        self.model = nn.Sequential(*[ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2)])

    def forward(self, x):
        return self.model(x)


class UnetUp(nn.Module):
    """
    Used in Unet for upscaling image feature maps.

    Args:
        - in_channels, out_channels (int): For convolutional layers, as indicated
    """
    
    def __init__(self, in_channels, out_channels):
        super(UnetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualConvBlock(out_channels, out_channels),
            ResidualConvBlock(out_channels, out_channels),
        ]   # Similar to convolution block, upscale by 2
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1) 
        x = self.model(x)
        return x


class EmbedFC(nn.Module):
    """
    Used to incorporate embeddings -- timestep, in diffusion network.

    Args:
        - input_dim, emb_dim (int): For dense layers, total nodes in and out of component
    """
    
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    """
    Full UNet model

    Args:
        - in_channels, out_channels (int): For convolutional layers, as indicated
        - n_feat (int): Channel-multiples throughout network [256]
        - n_classes (int): Conditional labelling [10, default for MNIST handwritten]
        - mid_shape (int): Downsized to this shape in middle, variable image dimensions allowed [7, default for MNIST handwritten]
    """
    
    def __init__(self, in_channels, out_channels, n_feat=256, n_classes=10, mid_shape=7):
        super(ContextUnet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = ResidualConvBlock(self.in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2*n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(mid_shape), nn.GELU())

        self.timeembed1 = EmbedFC(1, 2*n_feat)
        self.timeembed2 = EmbedFC(1, 1*n_feat)
        self.contextembed1 = EmbedFC(n_classes, 2*n_feat)
        self.contextembed2 = EmbedFC(n_classes, 1*n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2*n_feat, 2*n_feat, mid_shape, mid_shape), 
            nn.GroupNorm(8, 2*n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4*n_feat, n_feat)
        self.up2 = UnetUp(2*n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2*n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.out_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        """
        Args:
            - x (Tensor): "Noisy" image, input
            - c (Tensor): Context label
            - t (float): Timestep, scaled 0 to 1
            - context_mask (Tensor): Which samples to block context on 
        """
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        
        # mask out context if context_mask == 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1
        c = c * context_mask
        
        # embed context, time step
        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1*up1+ temb1, down2)  # add and multiply embeddings
        up3 = self.up2(cemb2*up2+ temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


""" FUNCTION: ddpm_schedules
Create and return a dictionary with values in noise schedule, linear schedule

Args:
    - beta_1, beta_2 (float): Controls strength of noise at beginning and end
    - T (int): Total number of timesteps
"""
def ddpm_schedules(beta1, beta2, T):
    assert 0 < beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)" 

    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t, 
        "oneover_sqrta": oneover_sqrta,  
        "sqrt_beta_t": sqrt_beta_t, 
        "alphabar_t": alphabar_t,  
        "sqrtab": sqrtab,  
        "sqrtmab": sqrtmab,  
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  
    }


class DDPM(nn.Module):
    """
    Module used to run DDPM training, using ContextUnet

    Args:
        - nn_model (pytorch model): Model used
        - betas (list of float): Noise schedule
        - n_T (int): Number of timesteps for image generation
        - device (pytorch device): Used in run
        - drop_prob (float): Probability for dropout layers [0.1]
    """

    def __init__(self, nn_model, betas, n_T, device, drop_prob=0.1):
        super(DDPM, self).__init__()
        self.nn_model = nn_model.to(device)

        # register_buffer allows accessing dictionary produced by ddpm_schedules
        # e.g. can access self.sqrtab later
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_mse = nn.MSELoss()

    def forward(self, x, cond, c):
        """
        Args:
            - x (Tensor): "Noisy" image, input
            - cond (Tensor): Used for cascaded DDPM, conditional image to be stacked
            - c (Tensor): Context label
        """
        # Timestep and noise generated at random in training process 
        _ts = torch.randint(1, self.n_T+1, (x.shape[0],)).to(self.device)  # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * noise
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        # dropout context with some probability
        context_mask = torch.bernoulli(torch.zeros_like(c)+self.drop_prob).to(self.device)
        
        # return MSE between added noise, and our predicted noise
        if cond is None:
            return self.loss_mse(noise, self.nn_model(x_t, c, _ts / self.n_T, context_mask))
        else:
            return self.loss_mse(noise, self.nn_model(torch.cat([x_t, cond], dim=1), c, _ts / self.n_T, context_mask))

    def sample(self, cond_x, cond_y, size, device, save_every=20, store=True):
        """
        Args:
            - cond_x (Tensor): Used for cascaded DDPM, conditional image to be stacked
            - cond_y (Tensor): Classes of desired image
            - size (tuple of int): Dimension of desired x
            - device (pytorch device): Used in run
            - save_every (int): How often x is saved to track progress through timesteps [20]
            - store (bool): Save throughout instead of just final product [True]
        """
        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1), sample initial noise

        # don't drop context at test time
        context_mask = torch.zeros_like(cond_y).to(device)

        x_i_store = [] # keep track of generated steps in case want to plot something 
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample,1,1,1)
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            if cond_x is None:
                eps = self.nn_model(x_i, cond_y, t_is, context_mask)
            else:
                eps = self.nn_model(torch.cat([x_i, cond_x], dim=1), cond_y, t_is, context_mask)
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            del eps
            if store and (i%save_every==0 or i==self.n_T):
                x_i_store.append(x_i.detach().cpu().numpy())
        
        x_i_store = np.array(x_i_store)
        return x_i, x_i_store