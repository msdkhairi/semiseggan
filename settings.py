import numpy as np
from torch import Tensor


# Data settings
DATA_ROOT = '/path/to/VOC'
DATA_LIST = None
IMG_MEAN = Tensor(np.array([0.485, 0.456, 0.406]))
IMG_STD = Tensor(np.array([0.229, 0.224, 0.225]))
DPT_MEAN = 0.37
SCALE_RANGE = (0.5, 2.0)
CROP_SIZE = 350
IGNORE_LABEL = 0
DPT_IGNORE_LABEL = -1
BCE_IGNORE_LABEL = -2

