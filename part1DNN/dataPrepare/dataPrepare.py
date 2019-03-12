# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         18-11-26

Description:
数据准备

将所用的数据按照类别分类，如标签为1的图片全部在“1”的文件夹下
尘肺病数据原来一张大小约2M，进过压缩后大小在400k左右

1.复制文件到一定的比例1:3
    原始文件0,1,2,3期比例是300:2500:380:80

2.将数据集按6:1分为训练集和测试集
    a.先划分好train/vali(保证互相不重复)
    b.复制扩充

最后4类结果比例保持---->16:25:16:10
"""

import os
import shutil
import random
from pandas.core.frame import DataFrame
import cv2
from matplotlib import pyplot as plt


def splitTrainAndVal():
    """
    将数据分隔成为训练集与验证集,无放回抽样,train:val = 6:1
    """
    # 远程
    fromDir = '/root/Hejia/data/zipXrayImages'
    toTrainDir = '/root/Hejia/data/zipXrayImages/train'
    toValDir = '/root/Hejia/data/zipXrayImages/validation'

    # 本地
    # fromDir = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages'
    # toTrainDir = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/train'
    # toValDir = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/validation'
    labelDir = ['0', '1', '2', '3']
    ratio = 1. / 7
    for dir in labelDir:
        pathFrom = os.path.join(fromDir, dir)
        pathTrain = os.path.join(toTrainDir, dir)
        pathVal = os.path.join(toValDir, dir)

        if not os.path.isdir(pathTrain):
            os.mkdir(pathTrain)
        if not os.path.isdir(pathVal):
            os.mkdir(pathVal)

        allImages = os.listdir(pathFrom)
        # 按比例不放回抽样
        valImages = DataFrame(allImages).sample(frac=ratio, replace=False)[0].values.tolist()

        # 复制
        for (i, img) in enumerate(valImages):
            shutil.copy(pathFrom + '/' + img, pathVal + '/' + img)
            if (i + 1) % 100 == 0:
                print(("%s class   validation    %d images done" % (dir, i + 1)))

        for (i, img) in enumerate(allImages):
            if img in valImages:
                continue
            shutil.copy(pathFrom + '/' + img, pathTrain + '/' + img)
            if (i + 1) % 100 == 0:
                print(("%s class   train         %d images done" % (dir, i + 1)))


def overSampleImages(path):
    '''
    对图片较少的类别上采样, 0,1,2,3类的数据计划比例变为16:25:16:10
    '''

    dirs = ['0', '2', '3']
    ratio = [25. / 16, 25. / 16, 25. / 10]
    poviotLen = len(os.listdir(path + '/1'))
    for i in range(3):
        p = os.path.join(path, dirs[i])
        for _, _, files in os.walk(p):
            num = 0
            max = (poviotLen / ratio[i])
            copyNum = max - len(files)
            # 复制
            while num <= copyNum:
                num += 1
                im = random.choice(files)
                copyFile = 'c' + str(random.randint(1, 10000)) + '_' + im
                shutil.copy(p + '/' + im, p + '/' + copyFile)
                if (num + 1) % 100 == 0:
                    print(('oversample %s class  %d images done') % (dirs[i], num + 1))


def checkXray():
    """
    检查X光片通道数,像素范围值,3个通道都是分布都是一样的
    """
    im = cv2.imread('/home/nicehjia/myProjects/data/chest_xray_Pneumonia/test/NORMAL/NORMAL2-IM-0381-0001.jpeg')
    # print(im.shape)

    color = ('r', 'g', 'b')
    for i, col in enumerate(color):
        histr = cv2.calcHist([im], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
        plt.show()


def resizeImages(fromdir, todir):
    '''
    i/o 太耗时间,先resize
    '''
    classes_dirs = ['0', '1', '2', '3']
    for l in classes_dirs:
        fromPath = os.path.join(fromdir, l)
        toPath = os.path.join(todir, l)
        if not os.path.isdir(toPath):
            os.mkdir(toPath)
        count = 0
        for _, _, imageList in os.walk(fromPath):
            for image in imageList:
                im_path = os.path.join(fromPath, image)
                im = cv2.imread(im_path)
                im = cv2.resize(im, (600, 600), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(toPath, image), im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  # jpg 表示的是图像的质量
                count += 1
                if (count + 1) % 100 == 0:
                    print('%s class resized %d images done' % (l, count + 1))


if __name__ == '__main__':
    # 均分
    splitTrainAndVal()
    print('---------------split done--------------------------')
    #
    overSampleImages('/root/Hejia/data/zipXrayImages/train')
    print('---------------train set overSampling done--------------------------')
    #
    overSampleImages('/root/Hejia/data/zipXrayImages/validation')
    print('---------------val set overSampling done--------------------------')
    #
    resizeImages('/root/Hejia/data/zipXrayImages/validation',
                 '/root/Hejia/data/zipXrayImages/validation_resized')
    resizeImages('/root/Hejia/data/zipXrayImages/train',
                 '/root/Hejia/data/zipXrayImages/train_resized')
