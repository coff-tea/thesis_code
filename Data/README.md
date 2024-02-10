# Whistles
Stored as audio clips saved into a numpy array. Expected data shape is `(number of samples, length of each sample)`. By default, a sampling rate of 50kHz and audio clips of length 1.0s are used so each sample has a length of 50,000.

# Synthetic whistle information
Stored as a dictionary where each item can be used to generate a whistle contour based on real tagged whistles. Has the with the following structure:
- KEY: Indicates the file and start time of the whistle (e.g. "filename.flag_starttime") <br>
  VAL: a tuple of `(numpy array TF, class 1, class 2)`
    - `numpy array TF`: list of time-frequency points of real tagged whistles, expected shape `(number of points, 2)`
    - `class 1`: indicates if whistle is going up (0) or down (1)
    - `class 2`: indicates if whistle is long (0), medium (1), or short (2) in length
