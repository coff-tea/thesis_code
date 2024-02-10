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

## `cross_tune_train.py`
For K-cross validation:
  - KEY: Indicates which fold is used as test set (e.g. "fold1") <br>
    VALUE: dictionary as follows...
    - KEY: "acc" <br>
      VALUE: list of training accuracies at every epoch
    - KEY: "loss" <br>
      VALUE: list of training losses at every epoch
    - KEY: "perf" <br>
      VALUE: tuple of model's final performance when saved at best training loss and associated statistics `(TR loss, TR acc, epoch where best was found, TE loss, TE acc, TE FA, TE MD)`
