# coding:utf-8
from skimage import data, io, transform
import matplotlib.pyplot as plt
from pylab import mpl
import itertools
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from PIL import Image

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# img0 = io.imread('D:/workCode/xRay/dataDemo/cnf_matrix/Vgg16.PNG')
# img1 = io.imread('D:/workCode/xRay/dataDemo/cnf_matrix/ResNet34.PNG')
# img2 = io.imread('D:/workCode/xRay/dataDemo/cnf_matrix/DenseNet121.PNG')
# # img3 = io.imread('D:/workCode/xRay/dataDemo/3/c1056_img20170426_17451721.jpg')
#
#
# plt.figure(num='astronaut', figsize=(8, 8))  # 创建一个名为astronaut的窗口,并设置大小
#
# plt.subplot(1, 3, 1)  # 将窗口分为两行两列四个子图,则可显示四幅图片
# plt.title('Vgg16')  # 第一幅图片标题
# plt.imshow(img0)  # 绘制第一幅图片
# plt.axis('off')
#
# plt.subplot(1, 3, 2)  # 第二个子图
# plt.title('ResNet34')  # 第二幅图片标题
# plt.imshow(img1)  # 绘制第二幅图片,且为灰度图
# plt.axis('off')  # 不显示坐标尺寸
#
# plt.subplot(1, 3, 3)  # 第三个子图
# plt.title('DenseNet121')  # 第三幅图片标题
# plt.imshow(img2)  # 绘制第三幅图片,且为灰度图
# plt.axis('off')  # 不显示坐标尺寸
#
# # plt.subplot(1, 4, 4)  # 第四个子图
# # plt.title(u'左右翻转')  # 第四幅图片标题
# # plt.imshow(img4)  # 绘制第四幅图片,且为灰度图
# # plt.axis('off')  # 不显示坐标尺寸
#
# plt.show()  # 显示窗口

cm_resNet = [[0.82, 0.08, 0.10, 0.00], [0.01, 0.97, 0.01, 0.01], [0.11, 0.00, 0.89, 0.00], [0.00, 0.63, 0.00, 0.37]]
cm_resNet = np.array(cm_resNet)
cm_vgg = np.array(
    [[0.88, 0.00, 0.12, 0.00], [0.02, 0.96, 0.01, 0.01], [0.11, 0.00, 0.89, 0.00], [0.00, 0.64, 0.08, 0.28]])

cm_densenet = np.array(
    [[0.85, 0.06, 0.09, 0.00], [0.01, 0.96, 0.01, 0.03], [0.06, 0.05, 0.87, 0.02], [0.00, 0.59, 0.00, 0.41]])


def plot_confusion_matrix(cm1, cm2, cm3, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.figure(num='astronaut', figsize=(10, 5))  # 创建一个名为astronaut的窗口,并设置大小

    plt.subplot(1, 3, 1)
    plt.imshow(cm1, interpolation='nearest', cmap=cmap)
    plt.title('Vgg16')
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm1.max() / 2.
    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        plt.text(j, i, format(cm1[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm1[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    ####

    plt.subplot(1, 3, 2)
    plt.imshow(cm2, interpolation='nearest', cmap=cmap)
    plt.title('ResNet34')
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm2.max() / 2.
    for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
        plt.text(j, i, format(cm2[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm2[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    #####

    plt.subplot(1, 3, 3)
    plt.imshow(cm3, interpolation='nearest', cmap=cmap)
    plt.title('DenseNet121')
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm3.max() / 2.
    for i, j in itertools.product(range(cm3.shape[0]), range(cm3.shape[1])):
        plt.text(j, i, format(cm3[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm3[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

    # plt.colorbar()
    plt.show()
    plt.savefig('D:/workCode/xRay/dataDemo/cnf_matrix/baseModel_4classes/cnf_all.png')


plot_confusion_matrix(cm_vgg, cm_resNet, cm_densenet, classes=['0', '1', '2', '3'])
