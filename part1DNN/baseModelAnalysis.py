# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         19-1-10

Description:  分析trainBaseModel的分类结果,包括混淆矩阵等

"""
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
import pandas as pd
from sklearn.metrics import recall_score

from utils import initialize_model
import flags
import myModelAnalysisDataset

os.environ['CUDA_VISIBLE_DIVICE'] = '1'


def getDataLoader(data, isMyImagePathDataLoader=False):
    '''
    获取dataLoader 和 label<-->label name
    '''

    size_1 = 256
    size_2 = 224

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_val = transforms.Compose([
        transforms.Resize(size_1),
        transforms.TenCrop(size_2),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
    ])

    if isMyImagePathDataLoader:
        datasetVal = myModelAnalysisDataset.ImagePathDataset(path_1=data, transform=transform_val)

    else:
        datasetVal = datasets.ImageFolder(root=data, transform=transform_val)

    dataLoaderVal = DataLoader(datasetVal, batch_size=2, shuffle=False, num_workers=2,
                               pin_memory=True)

    class_to_idx = [[], []]  # [[name], [index]]
    for class_name, class_idx in datasetVal.class_to_idx.items():
        class_to_idx[0].append(class_name)
        class_to_idx[1].append(class_idx)

    return dataLoaderVal, class_to_idx


def confuseMatrix(dataLoader, model_name, ckpt_path, labels_name, num_classes):
    """
    baselinemodel 输出结果并分析结果
    """
    print("initialize model")
    model_ft, input_size = initialize_model(model_name, num_classes=num_classes, feature_extract=False,
                                            use_pretrained=False)
    model_ft.load_state_dict(torch.load(ckpt_path)['model'])
    model_ft.cuda()
    model_ft.eval()

    y_predict = []
    y_true = []
    print("validation ")
    for index, (inputs, labels) in enumerate(dataLoader):
        inputs = inputs.to(flags.gpu)
        labels = labels.to(flags.gpu)

        # val过程中有
        bs, ncrops, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w)
        outputs = model_ft(inputs)
        outputs = outputs.view(bs, ncrops, -1).mean(1)

        _, preds = torch.max(outputs, 1)
        y_predict.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

    epoch_acc = recall_score(y_true, y_predict, average='macro')
    print(epoch_acc)
    cnf_matrix = confusion_matrix(y_true, y_predict, labels=labels_name)
    return cnf_matrix


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def get_cnf_matrix():
    """
    获取混淆矩阵
    :return:
    """
    data = '/home/njuciairs/Hejia/xRaydata/zipXrayImages/threeClasses_val'

    model_name = 'densenet'
    model_ckpt = '/home/njuciairs/Hejia/local_LogAndCkpt/ckpt/densenet_33_ckpt.pkl'
    num_classes = 3

    dataLoader, class_to_idx = getDataLoader(data)  # class_to_idx--> [[name], [index]]

    # 正负分类是需要用到reverse，为了好看。 数字多分分类分类时候不需要
    if num_classes == 2:
        class_to_idx[0].reverse()
        class_to_idx[1].reverse()

    print(class_to_idx)

    cnf_matrix = confuseMatrix(dataLoader, model_name, model_ckpt, class_to_idx[1], num_classes)

    # 保留数字的精度
    np.set_printoptions(precision=4)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_to_idx[0],
                          title='Confusion matrix, without normalization')
    plt.savefig('/home/njuciairs/Hejia/xRay_DeepLearing/part1DNN/tables/twoStep/densenet_2_sensitive_num.png')
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_to_idx[0], normalize=True,
                          title='Normalized confusion matrix')

    # plt.show()
    plt.savefig('/home/njuciairs/Hejia/xRay_DeepLearing/part1DNN/tables/twoStep/densenet_2_sensitive_rate.png')


def get_image_score():
    """
    获取每张图片的打分,格式：图片名字 标签 原始预测类 分类器打分1，分类器打分2,...
    """
    data = '/home/njuciairs/Hejia/xRaydata/zipXrayImages/fourClasses_val'
    model_name = 'densenet'
    model_ckpt = '/home/njuciairs/Hejia/local_LogAndCkpt/ckpt/densenet_33_ckpt.pkl'
    num_classes = 3

    dataLoader, class_to_idx = getDataLoader(data, isMyImagePathDataLoader=True)

    print("initialize model")
    model_ft, input_size = initialize_model(model_name, num_classes=num_classes, feature_extract=False,
                                            use_pretrained=False)
    model_ft.load_state_dict(torch.load(model_ckpt)['model'])
    model_ft.cuda()
    model_ft.eval()

    y_image_path = []
    y_predict = []
    y_predict_score = []
    y_true = []
    print("validation ")
    for index, (path, inputs, labels) in enumerate(dataLoader):
        inputs = inputs.to(flags.gpu)
        labels = labels.to(flags.gpu)

        # val过程中有
        bs, ncrops, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w)
        outputs = model_ft(inputs)
        outputs = outputs.view(bs, ncrops, -1).mean(1)

        _, preds = torch.max(outputs, 1)

        y_predict.extend(preds.cpu().numpy())
        y_true.extend(labels.cpu().numpy())
        y_image_path.extend(path)
        y_predict_score.extend(outputs.detach().cpu().numpy().tolist())

    record_csv = pd.DataFrame({'path': y_image_path, 'label': y_true, 'predict': y_predict, 'score': y_predict_score})
    record_csv.to_csv(
        '/home/njuciairs/Hejia/xRay_DeepLearing/part1DNN/tables/cultPoint2stepResult/4classesData_3classes_record_densenet33_sensitive.csv',
        encoding='utf-8')


if __name__ == '__main__':
    get_cnf_matrix()
    get_image_score()
