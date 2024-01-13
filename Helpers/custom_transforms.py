"""
Author: Xi Lu
File: custom_transforms.py
Description: Transforms that could be used in image generation.
""" 



#===================================================================================================
import torch



#===================================================================================================
# Default values used throughout for desired min/max of masking
mask_min = 0.25
mask_max = 1.


#===================================================================================================
class GaussianMask(object):
    """
    Sample from Gaussian distribution (min 0, max 1) and multiply with input

    Args:
        mean (float): For Gaussian distribution [0.5]
        sd (float): For Gaussian distribution [0.5]
        min_val (float): Clamping mask minimum [mask_min]
        max_val (float): Clamping mask maximum [mask_max]
    """

    def __init__(self, mean=0.5, sd=0.5, min_val=mask_min, max_val=mask_max):
        self.mean = mean
        self.sd = sd
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample):
        mask = torch.randn_like(sample)
        mask = mask * self.sd + self.mean
        mask = torch.clamp(mask, self.min_val, self.max_val)
        return mask * sample


#===================================================================================================
class UniformMask(object):
    """
    Sample from Uniform distribution and multiply with input

    Args:
        min_val (float): Clamping mask minimum [mask_min]
        max_val (float): Clamping mask maximum [mask_max]
        method (str): How the min/max is enforced ["clamp"]
    """
    def __init__(self, min_val=mask_min, max_val=mask_max, method="clamp"):
        self.min_val = min_val
        self.max_val = max_val
        self.method = method

    def __call__(self, sample):
        mask = torch.rand(*sample.shape)
        if self.method == "clamp":
            mask = torch.clamp(mask, self.min_val, self.max_val)
        elif self.method == "rescale":
            mask = mask * (self.max_val - self.min_val) + self.min_val
        return mask * sample


#===================================================================================================
class MinMaxNorm(object):
    """
    Perform min-max normalisation

    Args:
        min_val (float): New minimum [0.]
        max_val (float): New maximum [1.]
        mode (str): How the min/max is enforced ["scale"]
    """
    def __init__(self, min_val=0., max_val=1., mode="scale"):
        self.min_val = min_val
        self.max_val = max_val
        self.mode = mode

    def __call__(self, sample):
        if self.mode == "scale":
            sample -= torch.min(sample)
            sample /= (torch.max(sample) - torch.min(sample))
            sample *= (self.max_val - self.min_val)
            sample += self.min_val
            return torch.nan_to_num(sample)
        elif self.mode == "clamp":
            return torch.clamp(torch.nan_to_num(sample), min=self.min_val, max=self.max_val)
        else:
            return None
