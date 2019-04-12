# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         2019/3/24

Description:
获取cam图片，只能一张张的出
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


def get_mask(heatMap, img, heatmapThreshold=0.75, areaThreshold=0.05):
    '''
    heatMap-->找到Mask——>Mask圈出来的对应bounding box--->从原图bounding box抠图
    如果有多个高亮区域,保留占比超过某个面积阈值的区域
    若有多个超过阈值区域的面积，保留最大的那个，否则这张图就去掉
    '''
    threshold = int(heatmapThreshold * 255)
    _, mask = cv2.threshold(heatMap, threshold, 255, cv2.THRESH_BINARY)  # 255是填充色
    img_mask = cv2.bitwise_and(img, img, mask=mask)
    img_mask_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    contours, hier = cv2.findContours(img_mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # mask 最大连通区域的mask##################################
    contours, hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    # for index, c in enumerate(contours):
    #     if index == max_index:
    #         continue
    #     cv2.fillConvexPoly(mask, c, 0)
    # cv2.imwrite('./heatmapImages/temp/mask_max.jpg', mask)
    #######################################################

    # 找到最大的连通区域,如果最大连通区域都超过面积阈值就保留，否者就不用
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    max_area = cv2.contourArea(cnt)

    h, w, c = np.shape(img)
    # if max_area >= h * w * areaThreshold:
    # 画出bonding_box  x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    crop_img = img[y:y + h, x:x + w]  # 注意x，y 顺序
    print(np.shape(crop_img))
    cv2.imwrite('./heatmapImages/crop.jpg', crop_img)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imwrite('./heatmapImages/temp/img_bounding_box_new.jpg', img)


if __name__ == '__main__':

    imgs = 'D:/workCode/xRay/xRay_DeepLearing/part2Visualization/single_data'
    ckpt = 'D:/workCode/xRay/local_LogAndCkpt/ckpt/densenet_26_ckpt.pkl'
    num_classes = 4

    dataloader = data_input(imgs)
    img, _ = next(iter(dataloader))

    model = myModels.DenseNet(ckpt, num_classes)
    model.eval()

    # get the most likely prediction of the model
    # get the gradient of the output with respect to the parameters of the model
    pred = model(img)
    index = pred.argmax(dim=1)
    print(index)
    probility = F.softmax(torch.autograd.Variable(pred)).data
    print(probility)
    # index = 1

    pred[:, index].backward()
    # pull the gradients out of the model
    gradients = model.get_activations_gradient()
    # pool the gradients across the channels
    print(gradients.size())
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # get the activations of the last convolutional layer
    activations = model.get_activations(img).detach()

    # weight the channels by corresponding gradients
    b, c, h, w = activations.size()
    for i in range(c):
        activations[:, i, :, :] *= pooled_gradients[i]

    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    # channel = 999
    # heatmap = activations[0, channel, :, :]
    # relu on top of the heatmap expression (2) in https://arxiv.org/pdf/1610.02391.pdf
    heatmap = np.maximum(heatmap, 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    img = cv2.imread(
        'D:/workCode/xRay/xRay_DeepLearing/part2Visualization/single_data/ele/img.jpg')
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    # showHeatMap(heatmap)

    heatmap = np.uint8(255 * heatmap)
    get_mask(heatmap, img)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.5 + img
    # cv2.imwrite('./heatmapImages/temp/featuremap_{}.jpg'.format(channel), superimposed_img)

    cv2.imwrite('./heatmapImages/temp/map_box.jpg', superimposed_img)
