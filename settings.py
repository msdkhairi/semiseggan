import numpy as np
from torch import Tensor


# Data settings
DATA_ROOT = '/path/to/VOC'
DATA_LIST = None
IMG_MEAN = Tensor(np.array([0.485, 0.456, 0.406]))
DPT_MEAN = 94.50936413431783
# STD = Tensor(np.array([0.229, 0.224, 0.225]))
SCALE_RANGE = (0.5, 2.0)
CROP_SIZE = (350, 350)
IGNORE_LABEL = 0