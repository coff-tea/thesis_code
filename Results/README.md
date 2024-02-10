Files created during runtime will be saved here.

# Model files
Models are saved as `state_dict` objects in .pt files. They are named in accordance to the prefix from their respective .yaml file and the task they were generated during.

# Result dictionaries
Saves the relevant optimization or runtime results, depending on the task. Note:
- TR: training set
- TE: test set
- VA: validation set
- acc: accuracy
- FA: false alarm rate
- MD: missed detection rate

### OPTUNA_DICT: Anything generated using optuna
Each entry in the dictionary is an optimization trial. When running, if the file name to be generated exists, the existing trials are loaded as historical information. Optimization trials often take long periods of time and this allows many more trials to be run than can be accomplished in one session.
- KEY: Integer indicates that this is the nth run trial  <br>
  VAL: Dictionary as follows... 
  - KEY: `trial_value` <br>
    VAL: Optimization value, depending on the task
  - KEY: Name of optimized parameter, as many as exist <br>
    VAL: Value that used to generate this trial's value

### `cross_tune_train.py`
For K-cross validation:
- KEY: Indicates which fold is used as test set (e.g. "fold1") <br>
  VAL: Dictionary as follows...
  - KEY: "acc" <br>
    VAL: List of TR acc at every epoch
  - KEY: "loss" <br>
    VAL: List of TR loss at every epoch
  - KEY: "perf" <br>
    VAL: Tuple of model's final performance when saved at best TR loss and associated statistics `(TR loss, TR acc, epoch where best was found, TE loss, TE acc, TE FA, TE MD)` <br>
    
For hyperparameter tuning:
- OPTUNA_DICT with best validation loss as optimization value  <br>
  
For model training:
- KEY: "hyper" <br>
  VAL: Dictionary of hyperparameters used keyed by name
- KEY: "train_hist" <br>
  VAL: Tuple of `(list of TR loss at every epoch, list of TR acc at every epoch)`
- KEY: "val_hist" <br>
  VAL: Tuple of `(list of VA losses at every epoch, list of VA acc at every epoch)`
- KEY: "best" <br>
  VAL: Tuple of model's final performance when saved at best VA loss and associated statistics `(VA loss, VA acc, epoch where best was found, epoch when training stopped)`
- KEY: "test" <br>
  VAL: Tuple of model's final performance when saved at best VA loss and associated statistics `(TE loss, TE acc, TE FA, TE MD)` <br>
_optional keys, depending on runtime arguments_
- KEY: "percent" <br>
  VAL: Portion of training set used, value in (0, 1)
- KEY: "longer" <br>
  VAL: Multiplier for number of consecutive epochs needed with no change before training stops
- KEY: "starting" <br>
  VAL: Name of mode file where training starts at

### `generate_diffusion.py` 
For hyperparameter tuning:
- OPTUNA_DICT with best validation loss as optimization value <br>

For model training:
- KEY: Argument name <br>
  VAL: Argument value (all possible command line arguments and their values, default or specified)
- KEY: "mid_shape" <br>
  VAL: Integer for spatial dimension in centre of DDPM
- KEY: "data_points" <br>
  VAL: Number of images used for training
- KEY: "losses" <br>
  VAL: List of TR loss at every epoch
- KEY: "best" <br>
  VAL: Tuple of `(best TR loss, epoch this was found at)`
- KEY: "save_images" <br>
  VAL: Dictionary as follows...
  - KEY: "cond" <br>
    VAL: Tuple of `(conditional images or None, classes to be generated)`
  - KEY: Epoch <br>
    VAL: Tuple of `(generated image, list of generated images over timesteps)` saved as specific epochs
    
### `generate_diffusion_samples.py` 
- KEY: Class value <br>
  VAL: List of generated images corresponding to the class
  
### `generate_gan.py` 
Results dictionary:
- KEY: Argument name <br>
  VAL: Argument value (all possible command line arguments and their values, default or specified)
- KEY: "g_losses" <br>
  VAL: List of generator TR loss at every epoch
- KEY: "d_losses" <br>
  VAL: List of discriminator TR loss at every epoch
- KEY: "d_tricked" <br>
  VAL: Portion (0 to 1) of discriminator classification inaccuracy at every epoch
- KEY: "f_tricked" <br>
  VAL: Portion (0 to 1) of discriminator classification inaccuracy at every epoch for a fixed set of images
- KEY: "iso_list" <br>
  VAL: List of generated images saved at every checkpoint
- KEY: "f_noise" <br>
  VAL: Tensor of randomly generated noise used to generate fixed set of images
- KEY: "hyper" <br>
  VAL: Tuple of hyperparameters `(learning rate, smoothing, beta1, beta2, decay)`
- KEY: "penalties" <br>
  VAL: Dictionary as follows:
  - KEY: "sum" <br>
    VAL: Float for sum penalty
  - KEY: "noisy" <br>
    VAL: Float for noisy penalty
  - KEY: "empty" <br>
    VAL: Float for empty penalty

Models dictionary, if elected to save:
- KEY: "d_models" <br>
  VAL: List of discriminator `state_dict` objects
- KEY: "g_models" <br>
  VAL: List of generator `state_dict` objects
  
### `generate_gan_samples.py`
- KEY: "fxd_images" <br>
  VAL: Dictionary as follows...
  - KEY: Integer for index of which model in a list is used <br>
    VAL: Generated batch of images using fixed noise
- KEY: "fxd_output" <br>
  VAL: Dictionary as follows...
  - KEY: Integer for index of which model in a list is used <br>
    VAL: Discriminator outputs for generated fixed images
- KEY: "gen_images" <br>
  VAL: Dictionary as follows...
  - KEY: Integer for index of which model in a list is used <br>
    VAL: Generated batch of images using random noise
- KEY: "gen_output" <br>
  VAL: Dictionary as follows...
  - KEY: Integer for index of which model in a list is used <br>
    VAL: Discriminator outputs for generated random images

### `synth_det.py` 
- KEY: Argument name <br>
  VAL: Argument value (all possible command line arguments and their values, default or specified)
- KEY: "hyper" <br>
  VAL: Dictionary of hyperparameters used keyed by name
- KEY: "ttv" <br>
  VAL: Tuple of number of samples from `(TR, TE, VA)`
- KEY: "train_hist" <br>
  VAL: Tuple of `(list of TR loss at every epoch, list of TR acc at every epoch)`- KEY: "val_hist"
- KEY: "val_hist" <br>
  VAL: Tuple of `(list of VA losses at every epoch, list of VA acc at every epoch)`
- KEY: "best" <br>
  VAL: Tuple of model's final performance when saved at best VA loss and associated statistics `(VA loss, VA acc, epoch where best was found, epoch when training stopped)`
- KEY: "test" <br>
  VAL: Tuple of model's final performance when saved at best VA loss and associated statistics `(TE loss, TE acc, TE FA, TE MD)` <br>
- KEY: "unseen_splits"
  VAL: Dictionary as follows...
  - KEY: "pos" <br>
    VAL: Tuple of indices in unseen positive samples put in `(TR, TE)`
  - KEY: "pos" <br>
    VAL: Tuple of indices in unseen negative samples put in `(TR, TE)`
  - KEY: "unn"
- KEY: "pos_samples_used" <br>
  VAL: Dictionary as follows...
  - KEY: "train" <br>
    VAL: List of indices of unseen positive samples from unshuffled TR labelled as positive
  - KEY: "test" <br>
    VAL: List of indices of unseen positive samples from unshuffled TE labelled as positive
- KEY: "unn_samples_used" <br>
  VAL: Dictionary as follows...
  - KEY: "train" <br>
    VAL: List of indices of unseen negative samples from unshuffled TR labelled as positive
  - KEY: "test" <br>
    VAL: List of indices of unseen negative samples from unshuffled TE labelled as positive

### `synth_dis.py`
- KEY: Argument name <br>
  VAL: Argument value (all possible command line arguments and their values, default or specified)
- KEY: "hyper" <br>
  VAL: Dictionary of hyperparameters used keyed by name
- KEY: "train_hist" <br>
  VAL: Tuple of `(list of TR loss at every epoch, list of TR acc at every epoch)`
- KEY: "val_hist" <br>
  VAL: Tuple of `(list of VA losses at every epoch, list of VA acc at every epoch)`
- KEY: "best" <br>
  VAL: Tuple of model's final performance when saved at best VA loss and associated statistics `(VA loss, VA acc, epoch where best was found, epoch when training stopped)`
- KEY: "test" <br>
  VAL: Tuple of model's final performance when saved at best VA loss and associated statistics `(TE loss, TE acc, TE FA, TE MD)` <br>
_optional keys, depending on runtime arguments_
- KEY: "percent" <br>
  VAL: Portion of training set used, value in (0, 1)
- KEY: "longer" <br>
  VAL: Multiplier for number of consecutive epochs needed with no change before training stops
- KEY: "starting" <br>
  VAL: Name of mode file where training starts at
  
### `synth_iterative.py`
  
### `synth_optimize.py` 
