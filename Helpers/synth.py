"""
Author: Xi Lu
File: synth.py
Description: Functions used in the process of generating synthetic dolphin whistle images.
""" 



#===================================================================================================
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from matplotlib.collections import LineCollection
from scipy.ndimage.filters import gaussian_filter

from . import custom_transforms



#===================================================================================================
""" FUNCTION: make_synth
Return a list of numpy arrays containing the images/contours of the polynomial-fitted points

Args:
    - synth_dict (dict): Time-freq points for contours, irrelevant keys for this function
    - pf_degree (int): Polynomial degree to fit contour to points
    - width_specs (list): Values indicating mean, sd, and smoothing kernel for generated widths
    - intensity_specs (list): Values indicating mean, sd, and smoothing kernel for generated intensities
    - width_bounds (list): Min/max for width values [ [0,5] ]
    - intensity_bounds (list): Min/max for intensity values [ [0,1] ]
    - gauss_filter (bool): Use a Gaussian filter after generating [False]
    - gauss_kernel (float): Strength of Gaussian filter (2.5)
    - freq_flip (bool): Flip y-axis [True]
    - freq_bounds (list): Values indicating min and max of frequency in kHz [ [0,25] ]
    - time_bounds (list): Values indicating min and max of time in s [ [0,1] ]
    - zoom (float): Scaled zoom, none to only start/end of whistle [0]
    - max_data (int): Maximum number of synthetic contours to be returned [100000]
    - idx_list (list of int): If provided, which entries in dictionary to look at, indexed by counting [None]
    - ret_class (int): Which dictionary class to use [0, unconditional]
"""
def make_synth(synth_dict, pf_degree, width_specs, intensity_specs, \
                        width_bounds=[0,5], intensity_bounds=[0,1], \
                        gauss_filter=False, gauss_kernel=2.5, \
                        freq_flip=True, freq_bounds=[0,25], time_bounds=[0,1], zoom=0, \
                        max_data=100000, idx_list=None, ret_class=0):
    
    X_synth = []
    X_classes = []
    idx = 0
    count = 0
    for v in synth_dict.values():
        if idx_list is not None:    # Look at specific "indices" only
            if idx not in idx_list:
                idx += 1
                continue
        t = v[0][:, 0]      # time points
        f = v[0][:, 1]      # freq points
        fine_points = 5000
        coarse_points = 250
        tp = np.linspace(t[0], t[-1], num=fine_points)
        fp = np.poly1d(np.polyfit(t, f, pf_degree))(tp)
        if freq_flip:       # flip vertically
            fp = 25 - fp                
        if np.max(fp) > freq_bounds[1] or np.min(fp) < freq_bounds[0]:      # contour exceeds bounds
            continue
        #------------ Create the contour
        # Varying widths
        plot_widths = np.zeros(len(tp))     
        if width_specs[2] > 0:
            t_list = []
            w_list = []        
            for i in range(0, plot_widths.shape[0], coarse_points):
                temp = -1
                while temp < width_bounds[0] or temp > width_bounds[1]:
                    temp = np.random.normal(loc=width_specs[0], scale=width_specs[1])
                w_list.append(temp)
                t_list.append(tp[i])
            plot_widths = np.interp(tp, t_list, w_list)
            plot_widths[0:5] = 0
            plot_widths[-5:-1] = 0
            kwidths = len(tp) if width_specs[2] >= len(tp) else width_specs[2]
            smoother = np.array([1/kwidths for _ in range(kwidths)])
            plot_widths = np.convolve(plot_widths, smoother, mode="same")
        else:
            plot_widths.fill(width_specs[0])
        # Varying intensities
        plot_intens = np.zeros(len(tp))     
        if intensity_specs[2] > 0:
            # Horizontal
            t_list = []
            i_list = []
            for i in range(0, plot_intens.shape[0], coarse_points):
                temp = -1
                while temp < intensity_bounds[0] or temp > intensity_bounds[1]:
                    if True:#intensity_change_sudden:
                        temp = np.random.normal(loc=intensity_specs[0], scale=intensity_specs[1])
                    else:
                        temp = i_list[-1] + np.random.normal(loc=intensity_specs[0], scale=intensity_specs[1])
                i_list.append(temp)
                t_list.append(tp[i])
            plot_intens = np.interp(tp, t_list, i_list)
            kintens = len(tp) if intensity_specs[2] >= len(tp) else intensity_specs[2]
            smoother = np.array([1/kintens for _ in range(kintens)])
            plot_intens = np.convolve(plot_intens, smoother, mode="same")      
        else:
            plot_intens.fill(intensity_specs[0])
        # Draw contour
        points = np.array([tp, fp]).T.reshape(-1, 1, 2);
        segments = np.concatenate([points[:-1], points[1:]], axis=1);
        colors = [(i,i,i,1) for i in plot_intens]
        lc = LineCollection(segments, linewidths=plot_widths, colors=colors);
        fig, a = plt.subplots();
        a.add_collection(lc);
        x_min = max(zoom * min(tp), time_bounds[0])
        x_max = min(time_bounds[1] - zoom * (time_bounds[1]-max(tp)), time_bounds[1])
        y_min = max(zoom * min(fp), freq_bounds[0])
        y_max = min(freq_bounds[1] - zoom * (freq_bounds[1]-max(fp)), freq_bounds[1])
        a.set_xlim(x_min, x_max)
        a.set_ylim(y_min, y_max)
        plt.axis("off");
        fig.canvas.draw();
        contour = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8);
        contour = contour.reshape(fig.canvas.get_width_height()[::-1] + (3,));
        contour = np.amax(contour, axis=2);
        contour = 255 - contour         # Reverse colouring, set 255 where whistle is
        if np.max(contour) <= np.min(contour):      # If all zeros
            idx += 1
            continue
        if gauss_filter:
            contour = gaussian_filter(contour, sigma=gauss_kernel)
        X_synth.append(contour)
        idx += 1
        count += 1
        if ret_class > 0 and ret_class < len(v):    # Provided and valid
            X_classes.append(v[ret_class]) 
        else:
            X_classes.append(0)
        if count >= max_data:
            break
        elif count % 500 == 0:
            print("\t", count)
    return X_synth, X_classes


#===================================================================================================
""" FUNCTION: overlay_synth
Apply the standard transforms and potentially masking to the given tensor. Overlay an additional tensor (add) onto the baseline at a modified strength if provided

Args:
    - x (Tensor, np.array): Baseline tensor to normalise and return
    - base_norm (str): How to normalise x ["scale"]
    - add (Tensor, np.array): If provided, overlay onto x [None]
    - add_norm (str): How to normalise add ["scale"]
    - combo_norm (str): How to normalise combined Tensor ["scale"]
    - modi (float): Relative strength of add to x [1.]
    - data_shape (int): Desired output shape of square image [224]
    - min_val, max_val (float): Min/max of image pixels
"""
def overlay_synth(x, base_norm="scale", add=None, add_norm="scale", combo_norm="scale", modi=1., data_shape=224, min_val=0.0, max_val=1.0):
    if add is not None:     # Provided overlay
        if type(x) == torch.Tensor:
            tfs = transforms.Compose([transforms.Resize((data_shape, data_shape)), custom_transforms.MinMaxNorm(mode=base_norm)])
        else:
            tfs = transforms.Compose([transforms.ToTensor(), transforms.Resize((data_shape, data_shape)), custom_transforms.MinMaxNorm(mode=base_norm)])
        if type(add) == torch.Tensor:
            s_tfs = transforms.Compose([transforms.Resize((data_shape, data_shape)), custom_transforms.MinMaxNorm(max_val=modi, mode=add_norm)])
        else:
            s_tfs = transforms.Compose([transforms.ToTensor(), transforms.Resize((data_shape, data_shape)), custom_transforms.MinMaxNorm(max_val=modi, mode=add_norm)])
        return custom_transforms.MinMaxNorm(mode=combo_norm, min_val=min_val, max_val=max_val)(tfs(x) + s_tfs(add))
    else:       # No overlay
        if type(x) == torch.Tensor:
            tfs = transforms.Compose([transforms.Resize((data_shape, data_shape)), custom_transforms.MinMaxNorm(mode=base_norm, min_val=min_val, max_val=max_val)])
        else:
            tfs = transforms.Compose([transforms.ToTensor(), transforms.Resize((data_shape, data_shape)), custom_transforms.MinMaxNorm(mode=base_norm, min_val=min_val, max_val=max_val)])
        return tfs(x)


#===================================================================================================
""" FUNCTION: mask_synth
Multiply a mask with the synthetic frame

Args:
    - x (Tensor, np.array): Synthetic frame
    - m (Tensor, np.array): Mask
    - data_shape (int): Desired output shape of square image [224]
"""
def mask_synth(x, m, data_shape=224):
    if type(x) == torch.Tensor:
        x_tfs = transforms.Compose([transforms.Resize((data_shape, data_shape))])
    else:
        x_tfs = transforms.Compose([transforms.ToTensor(), transforms.Resize((data_shape, data_shape))])
    if type(m) == torch.Tensor:
        m_tfs = transforms.Compose([transforms.Resize((data_shape, data_shape))])
    else:
        m_tfs = transforms.Compose([transforms.ToTensor(), transforms.Resize((data_shape, data_shape))])
    return torch.multiply(x_tfs(x), m_tfs(m))