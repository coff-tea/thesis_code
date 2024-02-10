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

## OPTUNA_DICT: Anything generated using optuna
Each entry in the dictionary is an optimization trial. When running, if the file name to be generated exists, the existing trials are loaded as historical information. Optimization trials often take long periods of time and this allows many more trials to be run than can be accomplished in one session.
- KEY: Integer indicates that this is the nth run trial  <br>
  VALUE: Dictionary as follows... 
  - KEY: `trial_value` <br>
    VAL: Optimization value, depending on the task
  - KEY: Name of optimized parameter, as many as exist <br>
    VAL: Value that used to generate this trial's value

## `cross_tune_train.py`
For K-cross validation:
- KEY: Indicates which fold is used as test set (e.g. "fold1") <br>
  VALUE: Dictionary as follows...
  - KEY: "acc" <br>
    VALUE: List of TR acc at every epoch
  - KEY: "loss" <br>
    VALUE: List of TR loss at every epoch
  - KEY: "perf" <br>
    VALUE: Tuple of model's final performance when saved at best TR loss and associated statistics `(TR loss, TR acc, epoch where best was found, TE loss, TE acc, TE FA, TE MD)` <br>
    
For hyperparameter tuning:
  - OPTUNA_DICT with best validation loss as optimization value  <br>
  
For model training:
- KEY: "hyper" <br>
  VALUE: Dictionary of hyperparameters used keyed by name
- KEY: "train_hist" <br>
  VALUE: Tuple of `(list of TR loss at every epoch, list of TR acc at every epoch)`
- KEY: "val_hist" <br>
  VALUE: Tuple of `(list of VA losses at every epoch, list of VA acc at every epoch)`
- KEY: "best" <br>
  VALUE: Tuple of model's final performance when saved at best VA loss and associated statistics `(VA loss, VA acc, epoch where best was found, epoch when training stopped)`
- KEY: "test" <br>
  VALUE: Tuple of model's final performance when saved at best VA loss and associated statistics `(TE loss, TE acc, TE FA, TE MD)` <br>
_optional keys, depending on runtime arguments_
- KEY: "percent" <br>
  VAL: Portion of training set used, value in (0, 1)
- KEY: "longer" <br>
  VAL: Multiplier for number of consecutive epochs needed with no change before training stops
- KEY: "starting" <br>
  VAL: Name of mode file where training starts at

## `generate_diffusion.py` 
  
## `generate_diffusion_samples.py` 
  
## `generate_gan.py` 
  
## `generate_gan_samples.py`

## `synth_det.py` 

## `synth_dis.py`
  
## `synth_iterative.py`
  
## `synth_optimize.py` 
