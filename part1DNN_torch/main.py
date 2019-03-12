# coding=utf-8
'''
Train CIFAR10 with PyTorch
'''
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.init import xavier_uniform_
import pickle

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar
import flags

model_name = 'PreActResNet18'
net = preact_resnet.myPreActResNet18()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

train_batch_loss = []
val_batch_loss = []
train_batch_acc = []
val_batch_acc = []
train_epoch_acc = []
val_epoch_acc = []
batch_step = 0
epoch_step = 0

# Data
print('==> Preparing data...')
transform_train = transforms.Compose([
    # transforms.Resize(500),
    # transforms.RandomRotation([90, 180]),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(448),
    transforms.ToTensor(),  # 自动转换 range [0, 255] -> [0.0,1.0]
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(448),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.ImageFolder(root=flags.train_data, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=flags.batch_size, shuffle=True, num_workers=6)

testset = torchvision.datasets.ImageFolder(root=flags.val_data, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=flags.batch_size, shuffle=False, num_workers=6)

# Model
print('==> Building models...')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)  # 多卡
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 指定GPU
    cudnn.benchmark = True

if flags.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

weight = torch.Tensor([4, 3, 4, 7])
criterion = nn.CrossEntropyLoss(weight=weight).cuda()
optimizer = optim.SGD(net.parameters(), lr=flags.lr, momentum=flags.momentum, weight_decay=flags.momentum)


# # Training
# def train(epoch, train_batch_loss, train_batch_acc):
#     print('\nEpoch: %d' % epoch)
#     net.train()
#     train_loss = 0
#     correct = 0
#     total = 0
#     for batch_idx, (inputs, targets) in enumerate(trainloader):
#         inputs, targets = inputs.to(device), targets.to(device)
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()
#
#         train_loss += loss.item()
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()
#
#         batch_acc = ('%.4f' % (100. * correct / total))
#         batch_loss = ('%.4f' % (train_loss / (batch_idx + 1)))
#
#         progress_bar(batch_idx, len(trainloader), 'Loss: %s | Acc: %s%% (%d/%d)'
#                      % (batch_loss, batch_acc, correct, total))
#
#         train_batch_acc.append(batch_acc)
#         train_batch_loss.append(batch_loss)
#
#
# def validation(epoch, val_batch_loss, val_batch_acc, val_epoch_acc):
#     global best_acc
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_idx, (inputs, targets) in enumerate(testloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = net(inputs)
#             loss = criterion(outputs, targets)
#
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#             batch_acc = ('%.4f' % (100. * correct / total))
#             batch_loss = ('%.4f' % (test_loss / (batch_idx + 1)))
#
#             progress_bar(batch_idx, len(testloader), 'Loss: %s | Acc: %s%% (%d/%d)'
#                          % (batch_loss, batch_acc, correct, total))
#             val_batch_acc.append(batch_acc)
#             val_batch_loss.append(batch_loss)
#
#     # Save checkpoint.
#     acc = 100. * correct / total
#     val_epoch_acc.append(acc) # float
# #
#     if acc > best_acc:
#         print('Saving..')
#         state = {
#             'net': net.state_dict(),
#             'acc': acc,
#             'epoch': epoch,
#         }
#         if not os.path.isdir(flags.ckpt_path):
#             os.mkdir(flags.ckpt_path)
#         torch.save(state, os.path.join(flags.ckpt_path, '%s_%s_ckpt.pkl' % (model_name, epoch)))
#         best_acc = acc
#
#
# for epoch in range(start_epoch, start_epoch + flags.epoch):
#     train(epoch, train_batch_loss, train_batch_acc)
#     validation(epoch, val_batch_loss, val_batch_acc, val_epoch_acc)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # print(outputs)
        # print(targets)
        # print(loss)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            # output_np = outputs.data.cpu().numpy()
            # np.save('output1.npy', output_np)
            # targets_np = targets.data.cpu().numpy()
            # np.save('target1.npy', targets_np)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch + flags.epoch):
    train(epoch)
    test(epoch)

    # 每个epoch结束时保存一次log,避免log在中途丢失
    # t_loss_file = os.path.join(flags.log_path, model_name + '_train_batch_loss.pkl')
    # t_acc_file = os.path.join(flags.log_path, model_name + '_train_batch_acc.pkl')
    # v_loss_file = os.path.join(flags.log_path, model_name + '_val_batch_loss.pkl')
    # v_acc_file = os.path.join(flags.log_path, model_name + '_val_batch_acc.pkl')
    # v_acc_epoch_file = os.path.join(flags.log_path, model_name + '_val_epoch_acc.pkl')
    #
    # pickle.dump(train_batch_loss, open(t_loss_file, 'wb'))
    # pickle.dump(train_batch_acc, open(t_acc_file, 'wb'))
    # pickle.dump(val_batch_loss, open(v_loss_file, 'wb'))
    # pickle.dump(val_batch_acc, open(v_acc_file, 'wb'))
    # pickle.dump(val_epoch_acc, open(v_acc_epoch_file, 'wb'))
