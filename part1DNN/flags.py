# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         19-1-5

Description:  训练的超参数

"""

train_data = '/home/njuciairs/Hejia/xRaydata/zipXrayImages/threeClasses_train'
val_data = '/home/njuciairs/Hejia/xRaydata/zipXrayImages/threeClasses_val'
ckpt_path = '/home/njuciairs/Hejia/local_LogAndCkpt/ckpt'
log_path = '/home/njuciairs/Hejia/local_LogAndCkpt/logs'

num_classes = 3

batch_size = 32
# val  使用了tencrop 显存不够
batch_size_val = 4

# Flag for feature extracting. When False, we finetune the whole model,
feature_extract = False

gpu = "cuda"
model_name = "densenet"  # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
version = 33

# rate 衰减率, step_size 多少衰减一次
lr = 0.001
momentum = 0.9
step_size = 20
rate = 0.1
num_epochs = 80

model_vrsion = model_name + '_' + str(version)
