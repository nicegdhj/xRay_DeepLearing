# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         19-1-6

Description:  辅助工具,包括画曲线图,模型初始化

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

import flags

plt.switch_backend('agg')


def saveTrainCurve(val_acc, val_loss, train_acc, train_loss):
    """
    画训练曲线,并且保存训练曲线
    """
    num_epochs = flags.num_epochs

    log_dir = os.path.join(flags.log_path, flags.model_vrsion)
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    print(log_dir)
    t_loss_file = os.path.join(log_dir, 'train_loss.pkl')
    t_acc_file = os.path.join(log_dir, 'train_acc.pkl')
    v_loss_file = os.path.join(log_dir, 'val_loss.pkl')
    v_acc_file = os.path.join(log_dir, 'val_acc.pkl')

    pickle.dump(train_loss, open(t_loss_file, 'wb'))
    pickle.dump(train_acc, open(t_acc_file, 'wb'))
    pickle.dump(val_loss, open(v_loss_file, 'wb'))
    pickle.dump(val_acc, open(v_acc_file, 'wb'))

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)

    plt.title(" Accuracy vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel(" Accuracy")
    # plt.grid(axis="y")
    plt.plot(range(1, num_epochs + 1), val_acc, label="validation")
    plt.plot(range(1, num_epochs + 1), train_acc, label="train")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(0, num_epochs, 10.0))
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title(" Loss vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel(" Loss")
    # plt.grid(axis="y", ls='--')
    plt.plot(range(1, num_epochs + 1), val_loss, label="validation")
    plt.plot(range(1, num_epochs + 1), train_loss, label="train")
    plt.xticks(np.arange(0, num_epochs, 10.0))
    plt.legend()
    # plt.show()  # 展示
    plt.savefig(os.path.join(log_dir, 'chart'))


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ resnet34
        """
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG16_bn
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


