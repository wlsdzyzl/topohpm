import numpy as np
import nrrd
from utils import *


image_array, _, _ = load_itk("/media/wlsdzyzl/DATA/datasets/ASOCA/nii/cropped_middle_zoomed/val/data/Normal_8_cropped_zoomed.nii.gz")
print(image_array.shape, np.max(image_array), np.min(image_array))



