import numpy as np
import math
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt

def compute_mse(x, y):
    return np.mean(np.square(x-y))
    # return np.linalg.norm(x - y)

# x ref, y noisy
def compute_ssim(x, y):
    x = np.uint8(x*255.)
    y = np.uint8(y*255.)
    ssim_xy = ssim(x, y, data_range=255, multichannel=True)
    return ssim_xy

#assume RGB image
def compute_psnr(x, y):
    target_data = np.array(x)
    ref_data = np.array(y)
    
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20*math.log10(1.0/(rmse+1e-9))

def intensity_to_rgb(intensity, cmap='jet', normalize=False):
    assert intensity.ndim == 2, intensity.shape
    intensity = intensity.astype("float")

    if normalize:
        intensity -= intensity.min()
        intensity /= intensity.max()

    cmap = plt.get_cmap(cmap)
    intensity = cmap(intensity)[..., :3]
    return np.multiply(intensity.astype('float32'), 255.0)