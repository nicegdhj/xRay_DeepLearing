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


def end2endMaskFeature(get_train_dir=False):
    '''
    尘肺病端到端四分类CNN 提取细粒度病理特征
    '''

    feature_pred = []
    feature_label = []
    if get_train_dir:
        imgs = 'D:/workCode/data/xRay/dataset/copyBefore/val'
    else:
        imgs = 'D:/workCode/data/xRay/dataset/copyBefore/train'
    ckpt = 'D:/workCode/xRay/local_LogAndCkpt/ckpt/densenet_34_ckpt_v1.pkl'
    num_classes = 4
    dataloader = data_input(imgs)

    model = myModels.DenseNet(ckpt, num_classes)
    model.eval()

    for num, (img, label) in enumerate(dataloader):
        pred = model(img)

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
    # 训练集和测试集分两次执行获取
    if get_train_dir:
        label_mask_feature = open('D:/workCode/xRay/xRayData/mask_feature/copyBefore/label_mask/train_label_mask.pkl',
                                  'wb')
        pickle.dump(feature_label, label_mask_feature)
    else:
        pred_mask_feature = open('D:/workCode/xRay/xRayData/mask_feature/copyBefore/pred_mask/val_label_mask.pkl', 'wb')
        pickle.dump(feature_pred, pred_mask_feature)


def twoStepMaskFeature(get_train_dir=False):
    """
    尘肺病层级分类方法CNN 提取细粒度病理特征
    分为两阶段，第一阶段图片如果预测正类几率超过0.4（1，2,3期）则进入下一阶段获取细粒度特征，若没超过则在第一阶段就计算病理特征
    不要用代价敏感调整后打的模型
    """
    pass


if __name__ == '__main__':
    # 端到端模型提取细粒度病理特征
    end2endMaskFeature(get_train_dir=False)
    end2endMaskFeature(get_train_dir=True)

    # 层级分类方法提取细粒度病理特征
    # twoStepMaskFeature(get_train_dir=False)
    # twoStepMaskFeature(get_train_dir=True)
