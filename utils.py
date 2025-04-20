import os
import torch
import torch.nn as nn
import math
import numbers
import torch.nn.functional as F
import numpy as np
import csv
import random
# import tensorflow as tf
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter           # https://pytorch.org/docs/stable/tensorboard.html
import torchvision
import scipy.misc 
from torch.optim.optimizer import Optimizer, required
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


def setup_seed(seed):
    np.random.seed(seed)
    # random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = True

def create_folder(root_dir, version, path):
        if not os.path.exists(os.path.join(root_dir, version, path)):
            os.makedirs(os.path.join(root_dir, version, path))


class Logger(object):
    """Create a tensorboard logger to log_dir."""
    def __init__(self, log_dir):
        """Initialize summary writer."""
        self.writer = SummaryWriter(log_dir=log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
        self.writer.add_scalar(tag, value, step)

    def images_summary(self, tag, images, step):
        """Log a list of images."""
        self.writer.add_images(tag, images, step)

    def histo_summary(self, tag, values, step, bins='tensorflow', walltime=None, max_bins=None):
        """Log a histogram of the tensor of values."""
        self.writer.add_histogram(
            tag, values, global_step=step, bins=bins, walltime=walltime, max_bins=max_bins
        )
        self.writer.flush()


class ImagePool():                              # preventing Mode Collapse (input이 들어오면 pool_size에서 50% 확률로 image가 교체되어서 return됨)
    def __init__(self, pool_size):              # default pool_size=50
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)
        return return_images

def denorm(x):                  # used for saving image
    out = (x + 1) / 2.0
    return out.clamp_(0, 1)

def str2bool(v):
    return v.lower() in ('true')