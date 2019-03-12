"""
Finetuning Torchvision Models
============================
参考:https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
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

os.environ['CUDA_VISIBLE_DIVICE']='1'

import flags
import utils


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []
    val_loss_history = []

    train_acc_history = []
    train_loss_history = []

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0  # mean_accurary 其实就是宏查全率

    for epoch in range(0, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            y_pred = []
            y_true = []

            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects_normal = 0  # normal accuracy 只统计普通的正确率
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        # 处理val过程中的tencrop
                        if phase == 'val':
                            bs, ncrops, c, h, w = inputs.size()
                            inputs = inputs.view(-1, c, h, w)
                            outputs = model(inputs)
                            outputs = outputs.view(bs, ncrops, -1).mean(1)
                        else:
                            outputs = model(inputs)

                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects_normal += torch.sum(preds == labels.data)
                # mean acc 等价于 宏查全率
                y_pred.extend(preds.cpu().numpy())
                y_true.extend(labels.cpu().numpy())

            scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc_normal = running_corrects_normal.double() / len(dataloaders[phase].dataset)  # 修改mean_acc
            epoch_acc = recall_score(y_true, y_pred, average='macro')

            epoch_acc = float('{:.4f}'.format(epoch_acc))
            epoch_loss = float('{:.4f}'.format(epoch_loss))
            epoch_acc_normal = float('{:.4f}'.format(epoch_acc_normal))

            if phase == 'train':
                print('train Loss: {} mean Acc: {}  normal Acc : {}'.format(epoch_loss, epoch_acc, epoch_acc_normal))
            else:
                print('val   Loss: {} mean Acc: {}  normal Acc : {}'.format(epoch_loss, epoch_acc, epoch_acc_normal))

            # deep copy the model and save
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())

                print('Saving checkpoints...')
                state = {
                    'model': model.state_dict(),
                    'acc': epoch_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir(flags.ckpt_path):
                    os.mkdir(flags.ckpt_path)
                torch.save(state,
                           os.path.join(flags.ckpt_path, '%s_ckpt.pkl' % (flags.model_vrsion)))

            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            else:
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    with open('./record.txt', 'a') as f:
        f.writelines(['\n', str(flags.model_vrsion), '   ', str(best_acc)])

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return val_acc_history, val_loss_history, train_acc_history, train_loss_history


# Initialize the model for this run
model_ft, input_size = utils.initialize_model(flags.model_name, flags.num_classes, flags.feature_extract,
                                              use_pretrained=True)

print("Initializing Datasets and Dataloaders...")
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # 自动转换 range [0, 255] -> [0.0,1.0]
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
transform_test = transforms.Compose([
    #256-->224, 332-->299是 inception的
    transforms.Resize(256),
    transforms.TenCrop(input_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
])
datasetTrain = torchvision.datasets.ImageFolder(root=flags.train_data, transform=transform_train)
datasetVal = torchvision.datasets.ImageFolder(root=flags.val_data, transform=transform_test)
dataLoaderTrain = DataLoader(datasetTrain, batch_size=flags.batch_size, shuffle=True, num_workers=12, pin_memory=True)
dataLoaderVal = DataLoader(datasetVal, batch_size=flags.batch_size_val, shuffle=False, num_workers=12, pin_memory=True)

dataloaders_dict = {'train': dataLoaderTrain, 'val': dataLoaderVal}

# Detect if we have a GPU available
device = torch.device(flags.gpu if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)

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

optimizer_ft = optim.SGD(params_to_update, lr=flags.lr, momentum=flags.momentum)
scheduler = StepLR(optimizer_ft, step_size=flags.step_size, gamma=flags.rate, )
criterion = nn.CrossEntropyLoss()

# Train and evaluate
val_acc, val_loss, train_acc, train_loss = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft,
                                                       scheduler,
                                                       num_epochs=flags.num_epochs,
                                                       is_inception=(flags.model_name == "inception"))

# save logs
utils.saveTrainCurve(val_acc, val_loss, train_acc, train_loss)
