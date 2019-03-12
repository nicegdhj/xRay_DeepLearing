# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         19-1-21

Description:   共享网络权重

"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import recall_score
import torch.backends.cudnn as cudnn

import flags
import utils
import transferDataset


def train_model(model, dataloaders, criterion_1, criterion_2, optimizer_1, optimizer_2, scheduler, num_epochs,
                is_inception=False):
    '''
    scheduler 只用一个 主要是监控自己my_data训练的情况(迁移学习中迁移的目标领域)
    '''

    since = time.time()

    val_acc_history = []
    val_loss_history = []

    train_acc_history = []
    train_loss_history = []

    # mean_accurary 其实就是宏查全率
    best_acc = 0.0

    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            y_pred_1 = []
            y_true_1 = []
            y_pred_2 = []
            y_true_2 = []

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss_1 = 0.0
            running_corrects_normal_1 = 0  # normal accuracy 只统计普通的正确率
            running_loss_2 = 0.0
            running_corrects_normal_2 = 0

            # Iterate over data. inputs_1是my_data数量较少的数据集
            for inputs_1, labels_1, inputs_2, labels_2 in dataloaders[phase]:
                inputs_1 = inputs_1.to(device)
                labels_1 = labels_1.to(device)

                inputs_2 = inputs_2.to(device)
                labels_2 = labels_2.to(device)

                optimizer_1.zero_grad()
                optimizer_2.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs_1, aux_outputs_1 = model(inputs_1)
                        loss11 = criterion_1(outputs_1, labels_1)
                        loss12 = criterion_1(aux_outputs_1, labels_1)
                        loss_1 = loss11 + 0.4 * loss12

                        outputs_2, aux_outputs_2 = model(inputs_2)
                        loss21 = criterion_2(outputs_2, labels_2)
                        loss22 = criterion_2(aux_outputs_2, labels_2)
                        loss_2 = loss21 + 0.4 * loss22

                    else:
                        # 处理val过程中的tencrop
                        if phase == 'val':
                            bs, ncrops, c, h, w = inputs_1.size()
                            inputs = inputs_1.view(-1, c, h, w)
                            outputs_1 = model(inputs)
                            outputs_1 = outputs_1.view(bs, ncrops, -1).mean(1)

                            bs, ncrops, c, h, w = inputs_2.size()
                            inputs = inputs_2.view(-1, c, h, w)
                            outputs_2 = model(inputs)
                            outputs_2 = outputs_2.view(bs, ncrops, -1).mean(1)
                        else:
                            outputs_1 = model(inputs_1)
                            outputs_2 = model(inputs_2)

                        loss_1 = criterion_1(outputs_1, labels_1)
                        loss_2 = criterion_2(outputs_2, labels_2)

                    _, preds_1 = torch.max(outputs_1, 1)
                    _, preds_2 = torch.max(outputs_2, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss_1.backward()
                        optimizer_1.step()

                        loss_2.backward()
                        optimizer_2.step()

                running_loss_1 += loss_1.item() * inputs_1.size(0)
                running_corrects_normal_1 += torch.sum(preds_1 == labels_1.data)
                # mean acc 等价于 宏查全率
                y_pred_1.extend(preds_1.cpu().numpy())
                y_true_1.extend(labels_1.cpu().numpy())

                running_loss_2 += loss_2.item() * inputs_2.size(0)
                running_corrects_normal_2 += torch.sum(preds_2 == labels_2.data)
                y_pred_2.extend(preds_2.cpu().numpy())
                y_true_2.extend(labels_2.cpu().numpy())

            scheduler.step()

            epoch_loss_1 = running_loss_1 / len(dataloaders[phase].dataset)
            epoch_acc_normal_1 = running_corrects_normal_1.double() / len(dataloaders[phase].dataset)  # 修改mean_acc
            epoch_acc_1 = recall_score(y_true_1, y_pred_1, average='macro')

            epoch_loss_2 = running_loss_2 / len(dataloaders[phase].dataset)
            epoch_acc_normal_2 = running_corrects_normal_2.double() / len(dataloaders[phase].dataset)  # 修改mean_acc
            epoch_acc_2 = recall_score(y_true_2, y_pred_2, average='macro')

            epoch_acc_1 = float('{:.4f}'.format(epoch_acc_1))
            epoch_loss_1 = float('{:.4f}'.format(epoch_loss_1))
            epoch_acc_normal_1 = float('{:.4f}'.format(epoch_acc_normal_1))

            epoch_acc_2 = float('{:.4f}'.format(epoch_acc_2))
            epoch_loss_2 = float('{:.4f}'.format(epoch_loss_2))
            epoch_acc_normal_2 = float('{:.4f}'.format(epoch_acc_normal_2))

            if phase == 'train':
                print('my data---train Loss: {} mean Acc: {}  normal Acc : {}'.format(epoch_loss_1, epoch_acc_1,
                                                                                      epoch_acc_normal_1))
            else:
                print('my data---val   Loss: {} mean Acc: {}  normal Acc : {}'.format(epoch_loss_1, epoch_acc_1,
                                                                                      epoch_acc_normal_1))

            if phase == 'train':
                print('transfer data---train Loss: {} mean Acc: {}  normal Acc : {}'.format(epoch_loss_2, epoch_acc_2,
                                                                                            epoch_acc_normal_2))
            else:
                print('transfer data---val   Loss: {} mean Acc: {}  normal Acc : {}'.format(epoch_loss_2, epoch_acc_2,
                                                                                            epoch_acc_normal_2))

            # deep copy the model and save
            if phase == 'val' and epoch_acc_1 > best_acc:
                best_acc = epoch_acc_1
                # best_model_wts = copy.deepcopy(model.state_dict())

                print('Saving checkpoints...')
                state = {
                    'model': model.state_dict(),
                    'acc': epoch_acc_1,
                    'epoch': epoch,
                }
                if not os.path.isdir(flags.save_ckpt_path):
                    os.mkdir(flags.save_ckpt_path)
                torch.save(state,
                           os.path.join(flags.save_ckpt_path, '%s_ckpt.pkl' % (flags.model_vrsion)))

            if phase == 'val':
                val_acc_history.append(epoch_acc_1)
                val_loss_history.append(epoch_loss_1)
            else:
                train_acc_history.append(epoch_acc_1)
                train_loss_history.append(epoch_loss_1)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    with open('./record.txt', 'a') as f:
        f.writelines(['\n', str(flags.model_vrsion), '   ', str(best_acc)])

    return val_acc_history, val_loss_history, train_acc_history, train_loss_history


if __name__ == '__main__':
    print("Initializing Datasets and Dataloaders...")
    model_ft, input_size = utils.initialize_model(flags.model_name, flags.num_classes, flags.feature_extract,
                                                  use_pretrained=flags.useImageNetPretrained)

    if flags.useCheXNetPretrained:
        print("reload cheXNet...")
        model_ft = utils.loadMyStateDict(model_ft, flags.reload_ckpt)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 自动转换 range [0, 255] -> [0.0,1.0]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        # 256-->224, 332-->299是 inception的
        transforms.Resize(256),
        transforms.TenCrop(input_size),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
    ])

    datasetTrain = transferDataset.TransferDataset(path_1=flags.my_train_data, path_2=flags.train_data,
                                                   transform=transform_train)
    datasetVal = transferDataset.TransferDataset(path_1=flags.my_val_data, path_2=flags.val_data,
                                                 transform=transform_test)

    dataLoaderTrain = DataLoader(datasetTrain, batch_size=flags.batch_size, shuffle=False, num_workers=8,
                                 pin_memory=True)
    dataLoaderVal = DataLoader(datasetVal, batch_size=flags.batch_size_val, shuffle=False, num_workers=8,
                               pin_memory=True)
    dataloaders_dict = {'train': dataLoaderTrain, 'val': dataLoaderVal}

    device = flags.gpu if torch.cuda.is_available() else "cpu"
    model_ft = model_ft.to(device)
    if flags.multi_gpu:
        if device == 'cuda':
            model_ft = torch.nn.DataParallel(model_ft)  # 多卡
            cudnn.benchmark = True

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if flags.feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    optimizer_ft_1 = optim.SGD(params_to_update, lr=flags.lr, momentum=flags.momentum)
    optimizer_ft_2 = optim.SGD(model_ft.parameters(), lr=flags.lr, momentum=flags.momentum)
    # optimizer_ft_1 = optim.Adam(params_to_update, lr=flags.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # optimizer_ft_2 = optim.Adam(model_ft.parameters(), lr=flags.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = StepLR(optimizer_ft_1, step_size=flags.step_size, gamma=flags.rate, )
    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = nn.CrossEntropyLoss()

    # Train and evaluate
    val_acc, val_loss, train_acc, train_loss = train_model(model_ft, dataloaders_dict, criterion_1, criterion_2,
                                                           optimizer_ft_1,
                                                           optimizer_ft_2,
                                                           scheduler,
                                                           num_epochs=flags.num_epochs,
                                                           is_inception=(flags.model_name == "inception"))

    # save logs
    utils.saveTrainCurve(val_acc, val_loss, train_acc, train_loss)
