# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         19-1-21

Description:  共享网络权重权重dataset, 修改自 torchvision.datasets.ImageFolder
"""

import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def has_file_allowed_extension(filename, extensions=IMG_EXTENSIONS):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class ImagePathDataset(Dataset):

    def __init__(self, path_1, loader=default_loader, transform=None):

        classes_1, class_to_idx_1 = find_classes(path_1)
        samples_1 = make_dataset(path_1, class_to_idx_1, extensions=IMG_EXTENSIONS)
        if len(samples_1) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + path_1 + "\n"
                                                                              "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))


        random.shuffle(samples_1)

        self.loader = loader
        self.class_to_idx = class_to_idx_1
        self.samples_1 = samples_1

        self.transform = transform

    def __getitem__(self, index):

        path_1, target_1 = self.samples_1[index]
        sample_1 = self.loader(path_1)
        if self.transform is not None:
            sample_1 = self.transform(sample_1)

        return path_1, sample_1, target_1

    def __len__(self):
        return len(self.samples_1)
