# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         19-1-3

Description:  单元测试,检查代码问题

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


# 检查图片标签是否对应上
def checkloadData():
    data = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/train_resized'
    batch_size = 64

    transform_train = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(448),
        # transforms.CenterCrop(488),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.5),
        transforms.ToTensor(),  # 转换 range [0, 255] -> [0.0,1.0] 以及 降ndarray的(H x W x C)或PIL格式的转换为 (C x H x W)
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = torchvision.datasets.ImageFolder(root=data, transform=transform_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    class_names = dataset.classes

    print(class_names)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if batch_idx > 0:
            break
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in targets])
        print(targets)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(120)  # pause a bit so that plots are updated



if __name__ == '__main__':
    checkloadData()
