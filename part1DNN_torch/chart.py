# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         18-12-21

Description:  图表分析

"""

import numpy as np
import seaborn as sns
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt

ckpt_dir = '/home/nicehjia/myProjects/log/checkpoint'
train_acc = os.path.join(ckpt_dir, 'PreActResNet18_train_batch_acc.pkl')
train_loss = os.path.join(ckpt_dir, 'PreActResNet18_train_batch_loss.pkl')
val_acc = os.path.join(ckpt_dir, 'PreActResNet18_val_batch_acc.pkl')
val_loss = os.path.join(ckpt_dir, 'PreActResNet18_val_batch_loss.pkl')
val_acc_epoch = os.path.join(ckpt_dir, 'PreActResNet18_val_epoch_acc.pkl')

train_acc = pd.read_pickle(train_acc)
train_loss = pd.read_pickle(train_loss)
val_acc = pd.read_pickle(val_acc)
val_loss = pd.read_pickle(val_loss)
val_acc_epoch = pd.read_pickle(val_acc_epoch)

print(len(train_acc))
print(len(train_loss))
print(len(val_acc))
print(len(val_loss))
print(len(val_acc_epoch))

varible = train_acc

x = range(1, len(varible) + 1)
print(len(x))


print(varible)
#
l1 = plt.plot(x, varible, 'r--', label='train_acc')
# l2 = plt.plot(x, train_loss, 'g--', label='train_loss')
# plt.plot(x, train_acc, 'ro-', x, train_loss)
plt.title('The Lasers in Three Conditions')
plt.xlabel('step')
plt.ylabel('column')
plt.legend()
plt.show()



