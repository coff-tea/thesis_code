"""
Author: Xi Lu
File: spec_data.py
Description: Functions used in the creation and processing spectrogram data for dolphin whistles. 
""" 



#===================================================================================================
import random
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from scipy.signal import spectrogram as sspec
from scipy.signal import butter, lfilter



#===================================================================================================
""" FUNCTION: split_sets
Split set of indices as indicated by arguments and provide list of discrete lists

Args:
    - num_idx (int): Number of total indices to split
    - splits (list of float): Each float represents the next subdivision, so 80/10/10 split is given by [0.8,0.5]
    - replicable (bool): Seed the randomiser [True]
    - seed (int): What seed value to use [42]
"""
def split_sets(num_idx, splits, replicable=True, seed=42):
    idx = [i for i in range(num_idx)]
    if replicable:
        random.seed(seed)
    idx_splits = []
    for i in range(len(splits)):
        random_state = random.randint(0, 2**32-1)
        retain, idx = train_test_split(idx, train_size=splits[i], random_state=random_state)
        idx_splits.append(retain)
    idx_splits.append(idx)
    return(idx_splits)


#===================================================================================================
""" FUNCTION: load_data
Creates list of spectrograms (Min-Pre method) from list of signals, uses scipy.signal.spectrogram

Args:
    - X (list): Audio clips to convert
    - sr (int): Sampling rate [50000]
    - recale (bool): Use logarithmic scaling [True]
    - window (str): What window type for spectrogram conversion ["hamming"]
    - nperseg (int): Number of samples per windowing segment [1024]
    - noverlap (int): Number of samples overlap [None]
    - scaling (str): What scaling method to use when generating spectrogram ["spectrum"]
"""
def load_data(X, sr=50000, rescale=True, window="hamming", nperseg=1024, noverlap=None, scaling="spectrum"):
    X_spec = []
    for i in range(len(X)):
        _, _, ssx = sspec(X[i], fs=sr, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling)
        if rescale:
            ssx = 10*np.log10(ssx)
        ssx = (ssx-np.min(ssx)) / (np.max(ssx)-np.min(ssx))
        ssx = np.nan_to_num(ssx)
        X_spec.append(ssx)
    return X_spec


#===================================================================================================
""" FUNCTION: load_data_tf
As function load_data with specified parameters from Add-Pre (https://arxiv.org/abs/2211.15406)
"""
def load_data_tf(X, sr=50000, rescale=True, window="blackman", nperseg=2048, noverlap=410, scaling="spectrum"):
    X_spec = []
    for i in range(len(X)):
        windowed = butter_bandpass_filter(X[i], 5000, 20000, sr)
        windowed = windowed[int(sr*0.1):int(sr*0.9)]
        _, _, ssx = sspec(windowed, fs=sr, window=window, nperseg=nperseg, noverlap=noverlap, scaling=scaling)
        ssx = ssx[int(3/25*ssx.shape[0]):int(20/25*ssx.shape[0]), :]
        if rescale:
            ssx = 10*np.log10(ssx)
        ssx = (ssx-np.min(ssx)) / (np.max(ssx)-np.min(ssx))
        ssx = np.nan_to_num(ssx)
        X_spec.append(ssx)
    return X_spec


#===================================================================================================
""" FUNCTION: process_data
Creates list of data samples with given specifications. Assumes that the parameters given make sense for the mode specified.
    - single/avg/all mode: mdl_chs=1
    - stk mode: mdl_chs=chs
    - stkwavg mode: mdl_chs>chs

Args:
    - X (list): Spectrogram images
    - idx (list of int): Which indices to look at
    - mode (str): What processing type to use
    - mdl_chs (int): Number of desired channels
    - spat_dim (int): Spatial dimension of desired square image
    - tag (int): Class of image, 0/1 for detector [None]
    - which_ch (int): Which channel to look at if using only one [None]
"""
def process_data(X, idx, mode, mdl_chs, spat_dim, tag=None, which_ch=None):
    data = []
    chs = len(X)
    if mode == "all":   # If multiple channels exist, put each in as an individual sample
        for i in idx:
            for c in range(chs):
                sample = np.expand_dims(resize(X[ch][i], (spat_dim, spat_dim)), axis=0)
                if tag is not None:
                    data.append([sample, tag])
                else:
                    data.append([sample])
    else:
        for i in idx:
            sample = None
            if "stk" in mode:   # Audio channels stacked as image channel input
                frames = []
                for ch in range(chs):
                    frames.append(resize(X[ch][i], (spat_dim, spat_dim)))
                if mode == "stkwavg":   # Include average of individual channels 
                    avg = np.mean(np.array(frames), axis=0)
                    avg = np.nan_to_num((avg-np.min(avg)) / (np.max(avg)-np.min(avg)))
                    for extra in range(mdl_chs-chs):        # Repeat if necessary
                        frames.append(avg)
                sample = np.dstack(tuple(frames)).transpose((2,0,1))
            elif mode == "single" and which_ch is not None:     # Use just a single channel
                sample = np.expand_dims(resize(X[which_ch-1][i], (spat_dim, spat_dim)), axis=0)
                sample = np.nan_to_num((sample-np.min(sample)) / (np.max(sample)-np.min(sample)))
            elif mode == "avg":     # Use the average of existing channel(s)
                frames = []
                for ch in range(chs):
                    frames.append(resize(X[ch][i], (spat_dim, spat_dim)))
                avg = np.mean(np.array(frames), axis=0)
                avg = np.nan_to_num((avg-np.min(avg)) / (np.max(avg)-np.min(avg)))
                sample = np.expand_dims(avg, axis=0)
            if sample is not None:
                if tag is not None:
                    data.append([sample, tag])
                else:
                    data.append([sample])
    return data