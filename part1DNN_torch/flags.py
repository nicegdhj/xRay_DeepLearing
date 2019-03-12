# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         18-11-27

Description: 超参数

"""
resume = False

epoch = 200
batch_size = 128

lr = 0.1
momentum = 0.9
weight_decay = 5e-4


# 云主机
train_data = '/root/Hejia/data/zipXrayImages/train_resized'
val_data = '/root/Hejia/data/zipXrayImages/validation_resized'
ckpt_path = '/root/Hejia/xRayCheckpoints/PreActResNet18'
log_path = '/root/Hejia/xRayLogs/PreActResNet18'

# 笔记本
# train_data = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/train_resized'
# val_data = '/home/nicehjia/myProjects/xRay/xRayData/zipXrayImages/validation_resized'
# ckpt_path = '/home/nicehjia/myProjects/xRay/xRayCheckpoints/PreActResNet18'
# log_path = '/home/nicehjia/myProjects/xRay/xRayLogs/PreActResNet18'

# 工作站
