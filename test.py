import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import recall_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from torchvision import models
import pandas as pd


def saveTrainCurve(val_ac, val_los, train_ac, train_los):
    """
    画训练曲线,并且保存训练曲线
    """
    num_epochs = 80

    with open(val_ac, 'rb') as p:
        val_acc = pickle.load(p)

    with open(val_los, 'rb') as p:
        val_loss = pickle.load(p)

    with open(train_ac, 'rb') as p:
        train_acc = pickle.load(p)

    with open(train_los, 'rb') as p:
        train_loss = pickle.load(p)

    #     train_acc = pickle.load(train_ac)
    # val_loss = pickle.load(val_los)
    # train_loss = pickle.load(train_los)

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)

    plt.title(" marco-R vs. Number of Training Epochs")
    plt.xlabel("Training Epochs")
    plt.ylabel(" marco-R")
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
    plt.show()  # 展示


v1 = 'C:/Users/nicehjia/Desktop/阿里校招文件/densenet_33/val_acc.pkl'
v2 = 'C:/Users/nicehjia/Desktop/阿里校招文件/densenet_33/val_loss.pkl'
t1 = 'C:/Users/nicehjia/Desktop/阿里校招文件/densenet_33/train_acc.pkl'
t2 = 'C:/Users/nicehjia/Desktop/阿里校招文件/densenet_33/train_loss.pkl'

saveTrainCurve(v1, v2, t1, t2)
