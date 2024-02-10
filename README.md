# thesis_code
Dolphin whistle detection and generation using spectrogram images. Work and references used for this project can be found at: https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0438552

## How to run
.py files are designed to be run through command line with appropriate arguments. Less ephemeral arguments can be found in the "Helpers" folder with the corresponding .yaml file. Details about the arguments specified in command line can be found by running `python FILE.py -h`. 

## Folders
- Data: Dolphin whistles or information used for synthetic whistle generation, format dependent on usage.
- Helpers: Functions used in various main code files.
- Load: Location of results/models that will be used. For example: use a pretrained model, save here and specify as appropriate.
- Results: Location for runtime results to be saved. For example: trained a detector, find .pt file here.
- Temp: Temporary files that are not in use but are momentarily saved.

## Code files
More specifics can be found within the code files. Sparse details about functionality and how to run can be found below. All will use the folders as explained above unless a custom/changed .yaml file is specified.

### Detector
`cross_tune_train.py` <br>
For a whistle detector (output [0,1] for probability that spectrogram contains whistle), performs one of three specified tasks. Baseline requires: `python cross_tune_train.py MODE MODEL DATA`
- MODE specifies if running k-cross validation, hyperparameter tuning, or model training
- MODEL selects the type of model being used from a set list
- DATA specifies the format that data should be composed in from a set list

### Synthetic Optimization
`synth_det.py` <br>
For a given set of parameters, generate synthetic whistle contours overlaid onto negative backgrounds. Perform whistle detection by training a model with this synthetic set and a disjoint set of negative spectrograms. Baseline requires: `python synth_det.py ORIGIN DIRECTION KTH MODEL`
- ORIGIN specifies parameter file for synthetic whistle generation containing a dictionary of parameters associated with some score
- DIRECTION specifies how the score is optimized (i.e. maximal or minimal)
- KTH specifies the kth best set of parameters that should be used
- MODEL selects the type of model being used from a set list

`synth_dis.py` <br>
For a given set of parameters, generate synthetic whistle contours overlaid onto negative backgrounds. Perform whistle discrimination by training a model between this and a set of positive spectrograms containing real recorded whistles. Baseline requires: `python synth_dis.py ORIGIN DIRECTION KTH MODEL`
- ORIGIN specifies parameter file for synthetic whistle generation containing a dictionary of parameters associated with some score
- DIRECTION specifies how the score is optimized (i.e. maximal or minimal)
- KTH specifies the kth best set of parameters that should be used
- MODEL selects the type of model being used from a set list
  
`synth_iterative.py` <br>
For a given set of parameters, generate synthetic whistle contours overlaid onto negative backgrounds. Perform whistle discrimination by training a model between this and a set of positive spectrograms containing real recorded whistles. As iterations progress, integrate samples labelled as positive into the "positive" (i.e. synthetic) dataset and replace synthetic whistles. Baseline requires: `python synth_iterative.py ORIGIN DIRECTION KTH MODEL`
- ORIGIN specifies parameter file for synthetic whistle generation containing a dictionary of parameters associated with some score
- DIRECTION specifies how the score is optimized (i.e. maximal or minimal)
- KTH specifies the kth best set of parameters that should be used
- MODEL selects the type of model being used from a set list
  
`synth_optimize.py` <br>
Use a metric to determine an optimal set of synthetic whistle generation parameters. Baseline requires: `python synth_optimize.py MODE METRIC MODEL`
- MODE determines the general metric that is being used from a set list
- METRIC determines the specific way the MODE is being used and is specified within the code as a dictionary
- MODEL selects the type of model being used from a set list
  
### Generator
`generate_diffusion.py` <br>
Tune or train a denoised diffusion probabilistic model (DDPM) for synthetic whistle contour generation. Baseline requires: `python generate_diffusion.py MODE ORG`
- MODE specifies if running hyperparameter tuning or model training
- ORG specifies if the DDPM is being used as one entity or in a "chained" mode
  
`generate_diffusion_samples.py` <br>
Generate a number of samples from a trained DDPM. Baseline requires: `python generate_diffusion_samples.py FILE HYPER ORG NUM`
- FILE specifies the name of the trained model file
- HYPER specifies the name of the hyperparameter file
- ORG specifies if the DDPM is being used as one entity or in a "chained" mode
- NUM specifies the desired number of generated samples per class
  
`generate_gan.py` <br>
Tune or train a generative adversarial network (GAN) for synthetic whistle contour generation. Baseline requires: `python generate_gan.py SIZE TYPE FINAL HYPER PENALTY`
- SIZE specifies the spatial dimensions of the image data and thus size/structure of model
- TYPE specifies the type of GAN to use from a specified list
- FINAL specifies the final layer and min/max range of image values
- HYPER specifies the hyperparameter set to use, found within the code
- PENALTY specifies the penalties to use, found within the code
  
`generate_gan_samples.py` <br>
Generate a number of samples from a trained GAN. Baseline requires: `python generate_diffusion_samples.py FILE NUM`
- FILE specifies the name of the trained model file
- NUM specifies the desired number of generated samples per class
