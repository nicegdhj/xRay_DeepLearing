# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         19-1-5

Description:  训练的超参数

"""
# 数据
# 云主机
# chestRay14 子数据集
train_data = '/root/Hejia/data/chestRay14/chesRay14_4classes/train'
val_data = '/root/Hejia/data/chestRay14/chesRay14_4classes/val'
save_log_path = '/root/Hejia/xRayLogs'
save_ckpt_path = '/root/Hejia/xRayCheckpoints'

reload_ckpt = '/root/Hejia/reload_ckpt/model.pth.tar'

my_train_data = '/root/Hejia/data/zipXrayImages/train_resized'
my_val_data = '/root/Hejia/data/zipXrayImages/validation_resized'



# 笔记本
# train_data = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/train_resized'
# val_data = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/validation_resized'
# ckpt_path = '/home/nicehjia/myProjects/xRay/xRayCheckpoints/PreActResNet18'
# log_path = '/home/nicehjia/myProjects/xRay/xRayLogs/PreActResNet18'


num_classes = 4
batch_size = 32
# val  使用了tencrop 显存不够
batch_size_val = 4

# Flag for feature extracting. When False, we finetune the whole model,
feature_extract = False
useImageNetPretrained = True
useCheXNetPretrained = False
multi_gpu = False

gpu = "cuda"  # "cuda:0"

model_name = "densenet"  # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
version = 25

# sgd
# rate 衰减率, step_size 多少衰减一次
lr = 0.001
momentum = 0.9
step_size = 6
rate = 0.1
num_epochs = 20

model_vrsion = model_name + '_' + str(version)
