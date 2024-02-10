Helper files used within main code files. 

# `*.yaml`
All .yaml files are associated with a counterpart .py and provide less ephemeral information for runtime. Code files default to a .yaml file that correspond to their name, but another custom file can be made and selected when running using the appropriate command line switch.

# `custom_transforms.py`
Torchvision transforms created for our use, primarily for min-max normalisation.

# `detectors.py`, `diffusions.py`, `gans.py`
Creating models.

# `spec_data.py`
Functions used to convert audio data from an expected data structure to usable image data through spectrogram conversion and other potential operations (e.g. averaging, channel stacking).

# `synth.py`
Functions used to generate synthetic contours/whistles from an expected data structure.
