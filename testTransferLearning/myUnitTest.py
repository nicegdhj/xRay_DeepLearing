# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         19-1-15

Description:  测试代码

"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
import matplotlib.pyplot as plt

import transferDataset
import utils
import flags

# 测试加载部分pretrain model
modelA = '/home/nicehjia/Downloads/code/chexnet-master/models/m-25012018-123527.pth.tar'
modelB = '/home/nicehjia/Downloads/code/CheXNet-master/model.pth.tar'

pretrain = torch.load(modelB, map_location='cpu')['state_dict']


def showModelState():
    a = torch.load(modelB, map_location='cpu')['state_dict']
    for k, v in a.items():
        print(k)

    model, inputsize = utils.initialize_model('densenet', 4, False, False)
    for k, v in model.state_dict().items():
        print(k)


def loadMyStateDict(myModel, ckpt):
    '''
    在CheXNet基础上finetune
    ckpt的参数和my model虽然都是cheXNet(改自与densenet)但可能因版本的问题? ckpt的无法对应上mymodel的key
    '''

    own_state = myModel.state_dict()
    ckpt_state = torch.load(ckpt, map_location='cpu')['state_dict']
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


############################################
# 测试myTransferDataset

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(120)  # pause a bit so that plots are updated


def check_myTransferDataset():
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([
        # 256-->224, 332-->299是 inception的
        transforms.Resize(256),
        transforms.TenCrop(224),
        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
    ])

    path1_train = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/fourClasses_train'
    path2_train = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/fourClasses_val'

    dataset = transferDataset.TransferDataset(path_1=path1_train, path_2=path2_train, transform=transform)
    dataLoader = DataLoader(dataset, batch_size=8, num_workers=4)
    # class_names = dataset.classes_1

    for idx, (img1, target1, img2, target2) in enumerate(dataLoader):
        if idx > 0:
            break
        print(np.shape(img1))
        print(np.shape(img2))



if __name__ == '__main__':
    # check code

    # 检查loadMyStateDict
    # model, inputsize = utils.initialize_model('densenet', 4, False, False)
    # print(model.state_dict()['features.norm0.weight'])
    #
    # model = loadMyStateDict(model, modelB)
    # print(model.state_dict()['features.norm0.weight'])
    #
    # print(pretrain['module.densenet121.features.norm0.weight'])

    # 检查myTransferDataset
    check_myTransferDataset()
