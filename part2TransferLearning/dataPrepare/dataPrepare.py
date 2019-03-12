# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         19-1-14

Description:

处理chesray14 数据集. 下载地址https://nihcc.app.box.com/v/ChestXray-NIHCC

数据准备
1.将数据下载后, 首先挑选出 No Finding, Fibrosis, Emphysema, Nodule四类.因为原数据是多标签数据,但是我在这里把它们划分为
单独标签的数据,例如如image 1 有标签是Fibrosis, Emphysem时, 两个文件夹都应该放;标签是Fibrosis, 其他病时,只放入Fibrosis
调选的过程中加入了resized --->变为600*600

2. 挑出来之后No Finding: Fibrosis : Emphysema :Nodule = 58000:1607:2358:6005
需要数据平衡

3.数据切分分为train和val, 保证train和val的数据不互相包含

4.train和val分别扩充. 因为No Finding 和其他的差别太多, No Finding降采样到tran和val ,其他3类先分割成train和val再上采样

方案:

No Finding: Fibrosis : Emphysema :Nodule = 58000:1607:2358:6005
经过分割:
No Finding 直接下采样到最终方案,绕过上采样的步骤
train: No Finding: Fibrosis : Emphysema :Nodule =20000: 1407: 2058: 5225
val:  No Finding: Fibrosis : Emphysema :Nodule = 2000:  200: 300: 750

经过Fibrosis : Emphysema :Nodule三类上采样
train: No Finding: Fibrosis : Emphysema :Nodule = 20000: 5000: 5000: 10000
val:  No Finding: Fibrosis : Emphysema :Nodule = 2000:500:500:1000
"""

import os
import pandas as pd
import shutil
import cv2
import random
from pandas.core.frame import DataFrame


def selectImages():
    '''
    将 No Finding, Fibrosis, Emphysema, Nodule 分门别类的调选出来
    '''
    df = pd.read_csv('/home/nicehjia/myProjects/data/chestXray14/Data_Entry_2017.csv')

    # print(df.head(10))
    count = 0
    imageLabelMap = {}
    for idx, row in df.iterrows():
        # print(row['Image Index'], row['Finding Labels'])
        count += 1
        imageLabelMap[row['Image Index']] = row['Finding Labels']

    print(count)
    print(len(imageLabelMap))

    from_dir = '/home/nicehjia/myProjects/data/chestXray14/chesXray14_all_images'
    to_dir = '/home/nicehjia/myProjects/data/chestXray14/chesXray14_4classes/test'
    dir_list = ['images01', 'images02', 'images03', 'images04', 'images05', 'images06', 'images07', 'images08',
                'images09', 'images10', 'images11', 'images12']

    countNum = [0, 0, 0, 0]  # 统计各有多少张, No Finding, Fibrosis, Emphysema, Nodule
    count = 0
    # dir_list = ['images08']
    for dir in dir_list:
        dir = os.path.join(from_dir, dir)
        for img in os.listdir(dir):
            count += 1
            if count % 500 == 0:
                print('>>>>>>>> %s  cases done >>>>>>>' % count)
            label = imageLabelMap[img].split('|')
            for item in label:
                if item == 'No Finding':
                    resizeImage(dir, to_dir, classDir=item, imageName=img)
                    countNum[0] += 1
                elif item == 'Fibrosis':
                    resizeImage(dir, to_dir, classDir=item, imageName=img)
                    countNum[1] += 1
                elif item == 'Emphysema':
                    resizeImage(dir, to_dir, classDir=item, imageName=img)
                    countNum[2] += 1
                elif item == 'Nodule':
                    resizeImage(dir, to_dir, classDir=item, imageName=img)
                    countNum[3] += 1
    print(countNum)


def resizeImage(from_path, to_path, classDir, imageName):
    im = cv2.imread(os.path.join(from_path, imageName))
    im = cv2.resize(im, (600, 600), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(to_path, classDir, imageName), im)


def splitTrainAndVal(is_NoFinding):
    """
    将数据分隔成为训练集与验证集,无放回抽样
    """
    # pc上
    fromDir = '/home/nicehjia/myProjects/data/chestXray14/chesXray14_4classes/selectedImages'
    toTrainDir = '/home/nicehjia/myProjects/data/chestXray14/chesXray14_4classes/train'
    toValDir = '/home/nicehjia/myProjects/data/chestXray14/chesXray14_4classes/val'

    # No Finding类别直接降采样, 其他三类先分割再上采样
    if is_NoFinding:
        labelDir = ['No Finding']
        for dir in labelDir:
            pathFrom = os.path.join(fromDir, dir)
            pathTrain = os.path.join(toTrainDir, dir)
            pathVal = os.path.join(toValDir, dir)

            if not os.path.isdir(pathTrain):
                os.mkdir(pathTrain)
            if not os.path.isdir(pathVal):
                os.mkdir(pathVal)

            # 按比例不放回抽样,降采样
            allImages = os.listdir(pathFrom)
            df = DataFrame(allImages)
            trainImages = df.sample(n=20000, replace=False)[0].values.tolist()
            valImages = df.sample(n=2000, replace=False)[0].values.tolist()

            # 复制
            for (i, img) in enumerate(trainImages):
                shutil.copy(pathFrom + '/' + img, pathTrain + '/' + img)
                if (i + 1) % 1000 == 0:
                    print(("%s class   train       %d images done" % (dir, i + 1)))

            for (i, img) in enumerate(valImages):
                shutil.copy(pathFrom + '/' + img, pathVal + '/' + img)
                if (i + 1) % 1000 == 0:
                    print(("%s class   validation  %d images done" % (dir, i + 1)))

    else:
        labelDir = ['Fibrosis', 'Emphysema', 'Nodule']
        valNum = [200, 300, 750]

        for indx, dir in enumerate(labelDir):
            pathFrom = os.path.join(fromDir, dir)
            pathTrain = os.path.join(toTrainDir, dir)
            pathVal = os.path.join(toValDir, dir)

            if not os.path.isdir(pathTrain):
                os.mkdir(pathTrain)
            if not os.path.isdir(pathVal):
                os.mkdir(pathVal)

            # 不放回抽样
            allImages = os.listdir(pathFrom)
            valImages = DataFrame(allImages).sample(n=valNum[indx], replace=False)[0].values.tolist()

            for (i, img) in enumerate(valImages):
                shutil.copy(pathFrom + '/' + img, pathVal + '/' + img)
                if (i + 1) % 1000 == 0:
                    print(("%s class   validation    %d images done" % (dir, i + 1)))

            for (i, img) in enumerate(allImages):
                if img in valImages:
                    continue
                shutil.copy(pathFrom + '/' + img, pathTrain + '/' + img)
                if (i + 1) % 1000 == 0:
                    print(("%s class   train         %d images done" % (dir, i + 1)))


def overSampleImages(isTrain):
    '''
    对图片较少的Fibrosis : Emphysema :Nodule 三类上采样
    train: 1400: 2058: 5225 ---> 5000: 5000: 10000
    val:   200: 300: 750 ---> 500:500:1000
    '''

    toTrainDir = '/home/nicehjia/myProjects/data/chestXray14/chesXray14_4classes/train'
    toValDir = '/home/nicehjia/myProjects/data/chestXray14/chesXray14_4classes/val'
    labelDirs = ['Fibrosis', 'Emphysema', 'Nodule']

    # 处理训练集
    if isTrain:
        overSampleNum = [5000, 5000, 10000]
        for index, dir in enumerate(labelDirs):
            currPath = os.path.join(toTrainDir, dir)
            curr = os.listdir(currPath)
            currNum = len(curr)
            sampleImages = DataFrame(curr).sample(n=overSampleNum[index] - currNum, replace=True)[0].values.tolist()

            # 复制
            for (i, img) in enumerate(sampleImages):
                copyImg = 'copy_' + str(random.randint(1, 10000)) + '_' + img
                shutil.copy(currPath + '/' + img, currPath + '/' + copyImg)
                if (i + 1) % 500 == 0:
                    print((' tranDataset oversample %s class %d images done' % (dir, i + 1)))
    else:
        overSampleNum = [500, 500, 1000]
        for index, dir in enumerate(labelDirs):
            currPath = os.path.join(toValDir, dir)
            curr = os.listdir(currPath)
            currNum = len(curr)
            sampleImages = DataFrame(curr).sample(n=overSampleNum[index] - currNum, replace=True)[0].values.tolist()

            # 复制
            for (i, img) in enumerate(sampleImages):
                copyImg = 'copy_' + str(random.randint(1, 10000)) + '_' + img
                shutil.copy(currPath + '/' + img, currPath + '/' + copyImg)
                if (i + 1) % 500 == 0:
                    print((' tranDataset oversample %s class %d images done' % (dir, i + 1)))


## step1 挑选出数据
# selectImages()

# step2 数据分割

# #splitTrainAndVal(True)
# splitTrainAndVal(False)

# #step3 部分上采样
# overSampleImages(True)
# overSampleImages(False)
