from __future__ import print_function
import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import inspect, re
import os
import collections
import torch.nn.functional as F
import cv2

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im_old(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    if image_numpy.shape[2] < 3:
        image_numpy = np.dstack([image_numpy]*3)
    return image_numpy.astype(imtype)

def tensor2im(image_tensor, imtype=np.uint8):
    img = image_tensor[0].cpu().float().numpy()
    img = (((img - img.min()) * 255) / (img.max() - img.min())).transpose(1, 2, 0).astype(imtype)
    if img.shape[2] < 3:
        img = np.dstack([img]*3)
    return img

def gkern_2d(size=5, sigma=3):
    # Create 2D gaussian kernel
    dirac = np.zeros((size, size))
    dirac[size//2, size//2] = 1
    mask = gaussian_filter(dirac, sigma)
    # Adjust dimensions for torch conv2d
    return np.stack([np.expand_dims(mask, axis=0)] * 3)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def visual_attmaps(input_img, input_attmaps):
    attmaps_resize = F.interpolate(input_attmaps, size=[256, 256], mode='bilinear', align_corners=True)
    attmap_cpu = attmaps_resize[0].detach().cpu().numpy()
    attmap_numpy = attmap_cpu[0]

    # If use sigmoid as attention activation, normalization is not needed.
    attmap_norm = (attmap_numpy * 255).astype(np.uint8)

    img = input_img[0].cpu().float().numpy()
    img = (((img - img.min()) * 255) / (img.max() - img.min())).transpose(1, 2, 0).astype(np.uint8)
    if img.shape[2] < 3:
        img = np.dstack([img]*3)
    img_cv = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    height, width, _ = img_cv.shape
    heatmap = cv2.applyColorMap(cv2.resize(attmap_norm, (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img_cv * 0.5
    res_image = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)

    return res_image
