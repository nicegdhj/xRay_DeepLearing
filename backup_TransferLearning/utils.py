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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix

import flags

plt.switch_backend('agg')


def saveTrainCurve(val_acc, val_loss, train_acc, train_loss):
    """
    画训练曲线,并且保存训练曲线
    """
    num_epochs = flags.num_epochs

    log_dir = os.path.join(flags.save_log_path, flags.model_vrsion)
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

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def loadMyStateDict(myModel, ckpt):
    '''
    在CheXNet基础上finetune
    ckpt的参数和my model虽然都是cheXNet(改自与densenet)但可能因版本的问题? ckpt的无法对应上mymodel的key
    '''

    own_state = myModel.state_dict()
    ckpt_state = torch.load(ckpt)['state_dict']
    own_rename_state = {}

    notLoadLayers = ['module.densenet121.classifier.0.weight', 'module.densenet121.classifier.0.bias']
    match_parttern = ['denseblock1', 'denseblock2', 'denseblock3', 'denseblock4']

    # 修改own model的 key的指向
    # 例如 features.denseblock4.denselayer16.conv2.weight
    # module.densenet121.features.denseblock4.denselayer16.conv.2.weight->features.denseblock4.denselayer16.conv2.weight
    my_layer_num = []
    for name, param in own_state.items():
        name_item = name.split('.')

        if name_item[1] in match_parttern:
            # conv2 ->conv和2
            item1 = name_item[-2][:-1]
            item2 = name_item[-2][-1]
            item3 = name_item[-1]

            new_name_item = name_item[:-2]
            new_name_item.append(item1)
            new_name_item.append(item2)
            new_name_item.append(item3)
            new_name = '.'.join(new_name_item)
            new_name = 'module.densenet121.' + new_name
        else:

            new_name = 'module.densenet121.' + name
        my_layer_num.append(name)
        own_rename_state[new_name] = name

    in_layer_num = []
    all_pretrain_layer_num = []
    for name, param in ckpt_state.items():
        all_pretrain_layer_num.append(name)
        if name not in own_rename_state:
            print('except not in own_rename_state layer: ', name)
            continue
        if name in notLoadLayers:
            print('except notLoadLayers layer: ', name)
            continue
        own_state[own_rename_state[name]].copy_(param.data)
        in_layer_num.append(name)
    # check

    print("-------------------------")
    print('all pretrain layer num:  %d, laoded layer num:  %d' % (len(all_pretrain_layer_num), len(in_layer_num)))
    print('my own model layer num: %d' % len(my_layer_num))
    print("-------------------------")

    return myModel
