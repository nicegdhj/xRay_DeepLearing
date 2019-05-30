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


class TransferDataset(Dataset):

    def __init__(self, path_1, path_2, loader=default_loader, transform=None):

        classes_1, class_to_idx_1 = find_classes(path_1)
        samples_1 = make_dataset(path_1, class_to_idx_1, extensions=IMG_EXTENSIONS)
        if len(samples_1) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + path_1 + "\n"
                                                                              "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        classes_2, class_to_idx_2 = find_classes(path_2)
        samples_2 = make_dataset(path_2, class_to_idx_2, extensions=IMG_EXTENSIONS)
        if len(samples_2) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + path_2 + "\n"
                                                                              "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        # sample1和sample2 样本数量不一样,将小数据集的那个样本数量补全到到大数据集的数据量
        # 比如[1,2]和[3,4,5,6,7,8]变为[1,2,1,2,1,2]和[3,4,5,6,7,8]
        # 并且自己shuffle, 在dataloader shuffle就免了

        if len(samples_1) > len(samples_2):
            max_samples = samples_1
            min_samples = samples_2
        else:
            max_samples = samples_2
            min_samples = samples_1

        count = 0
        idx = 0
        _min_samples = min_samples
        while count < len(max_samples) - len(_min_samples):
            # 循环添加
            if idx >= len(_min_samples) - 1:
                idx = 0
            min_samples.append(_min_samples[idx])
            count += 1
            idx += 1

        random.shuffle(max_samples)
        random.shuffle(min_samples)

        self.loader = loader

        # self.class_to_idx_1 = class_to_idx_1
        self.samples_1 = min_samples

        # self.class_to_idx_2 = class_to_idx_2
        self.samples_2 = max_samples

        self.transform = transform

    def __getitem__(self, index):
        # batch_size = 8 时, sample_1和sample_2 都会取8张

        path_1, target_1 = self.samples_1[index]
        sample_1 = self.loader(path_1)

        path_2, target_2 = self.samples_2[index]
        sample_2 = self.loader(path_2)

        if self.transform is not None:
            sample_1 = self.transform(sample_1)
            sample_2 = self.transform(sample_2)

        return sample_1, target_1, sample_2, target_2

    def __len__(self):
        return len(self.samples_1)
