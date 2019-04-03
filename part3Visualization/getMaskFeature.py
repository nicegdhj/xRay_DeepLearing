# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         2019/3/24

Description:
提取cam特征：conv feature --> masked feature
参考：https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle

import myModels


def data_input(image_root, batch_size=1):
    '''
    :param root:  图片所在文件夹
    :param batch_size:
    '''
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = datasets.ImageFolder(root=image_root, transform=transform)
    dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=batch_size)
    return dataloader


def showHeatMap(heatmap):
    """
    观察heatmap阈值比例
    """
    print(np.max(heatmap), np.min(heatmap))
    plt.matshow(heatmap.squeeze())
    plt.colorbar()
    plt.show()


def get_descriptor(activations, heatmapThreshold=0.7, areaThreshold=0.05):
    '''
    获取masked 后的 activations 特征，这个特征被归一化后再masked
    :param activations: 4-D tensor, b,c,h,w
    :param heatmapThreshold: 暂时不用，目前采用的方法是大于heatMap-2d均值就采用
    :param areaThreshold: 暂时不用，目前采用的方法是每张图取最大联通区域
    :return: 3-d  mask以后的 ndarray, 因为整个程序是batch_size =1
    '''

    featuremap_2d = torch.mean(activations, dim=1).squeeze()  # 3d->2d
    # relu on top of the heatmap, expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    featuremap_2d = np.maximum(featuremap_2d, 0)

    # 归一化，因为上一步min=0，所以x-0/max-0 =>x/max
    featuremap_2d /= torch.max(featuremap_2d)

    descriptor_threshold = torch.mean(featuremap_2d).numpy()
    featuremap_2d = featuremap_2d.numpy()

    # 可视化观察 featuremap -->heatmap 还需要压到255的范围,RGB才能有颜色,并且是单通道是灰度图
    # heatmap = np.uint8(255 * featuremap_2d)
    # _, mask = cv2.threshold(heatmap, descriptor_threshold*255, 255, cv2.THRESH_BINARY)  # 255是填充色，白
    # img_mask = cv2.bitwise_and(heatmap, mask)
    # # cv2.imwrite('./heatmapImages/heatmap_mask.jpg', img_mask)
    # cv2.imwrite('./heatmapImages/mask.jpg', mask)

    _, mask = cv2.threshold(featuremap_2d, descriptor_threshold, 255, cv2.THRESH_BINARY)
    mask = np.array(mask, np.uint8)

    # 找到最大的连通区域,
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)

    # 除开最大联通区域都被填为0
    for index, c in enumerate(contours):
        if index == max_index:
            continue
        cv2.fillConvexPoly(mask, c, 0)

    # 转换成0,1-->3d mask
    _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
    b, c, h, w = activations.size()

    # activations的每个featurmap进行归一化并乘以掩码
    activations = activations.numpy()
    for i in range(b):
        for j in range(c):
            scale = np.max(activations[i, j, :, :]) - np.min(activations[i, j, :, :])
            activations[i, j, :, :] = (activations[i, j, :, :] - np.min(activations[i, j, :, :])) / scale
            activations[i, j, :, :] *= mask
    # 4d-->3d 因为整个程序是batch_size =1
    return activations[0]


if __name__ == '__main__':
    feature_pred = []
    feature_label = []

    imgs = 'D:/workCode/data/xRay/dataset/copyBefore/val'
    ckpt = 'D:/workCode/xRay/local_LogAndCkpt/ckpt/densenet_34_ckpt_v1.pkl'
    num_classes = 4
    dataloader = data_input(imgs)

    model = myModels.DenseNet(ckpt, num_classes)
    model.eval()

    for num, (img, label) in enumerate(dataloader):
        pred = model(img)
        # 按label的cam提取mask特征
        # pred[:, label].backward()
        # # pull the gradients out of the model
        # gradients = model.get_activations_gradient()
        # # pool the gradients across the channels
        # pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        # # get the activations of the last convolutional layer
        # activations = model.get_activations(img).detach()
        #
        # # weight the channels by corresponding gradients
        # b, c, h, w = activations.size()
        # for i in range(c):
        #     activations[:, i, :, :] *= pooled_gradients[i]
        # masked_feature = get_descriptor(activations)
        # img_sample = (masked_feature, label)
        # feature_label.append(img_sample)

        # 按预测种类提取mask特征
        index = pred.argmax(dim=1)
        pred[:, index].backward()
        gradients = model.get_activations_gradient()
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
        activations = model.get_activations(img).detach()

        b, c, h, w = activations.size()
        for i in range(c):
            activations[:, i, :, :] *= pooled_gradients[i]
        masked_feature = get_descriptor(activations)
        img_sample = (masked_feature, label)
        feature_pred.append(img_sample)

        if (num + 1) % 100 == 0:
            print('=====>  ', num + 1, '  ====>imgs done')

    # label_mask_feature = open('D:/workCode/xRay/xRayData/mask_feature/copyBefore/label_mask/train_label_mask.pkl', 'wb')
    # pickle.dump(feature_label, label_mask_feature)
    pred_mask_feature = open('D:/workCode/xRay/xRayData/mask_feature/copyBefore/pred_mask/val_label_mask.pkl', 'wb')
    pickle.dump(feature_pred, pred_mask_feature)
