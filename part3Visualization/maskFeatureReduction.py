# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         2019/3/30

Description:  
MaskFeature特征降维并可视化
"""

from pylab import mpl
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from time import time
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def visulizition3D(data_pca, label):
    """
    降维到3维后可视化
    """

    fig = plt.figure()
    ax = Axes3D(fig)
    # ax = fig.add_subplot(111, projection='3d')
    print(np.shape(data_pca), np.shape(label))
    color = ['r', 'g', 'y', 'b']
    node = ['nan', 'nan', 'nan', 'nan']
    for i in range(int(data_pca.size / 3)):
        team = data_pca[i]
        ax.scatter(team[0], team[1], team[2], c=color[label[i]], alpha=0.5)
        if node[label[i]] == 'nan':
            node[label[i]] = ax.scatter(team[0], team[1], team[2], c=color[label[i]], alpha=0.5)

    ax.legend((node[0], node[1], node[2], node[3]), (u"0类", u"1类", u"2类", u"3类"), loc=0)
    plt.show()


def visulizition2D(data_pca, label):
    """
    降维到2维后可视化
    """
    fig, ax = plt.subplots()
    color = ['r', 'g', 'y', 'b']
    node = ['nan', 'nan', 'nan', 'nan']
    print(type(data_pca))
    for i in range(int(len(data_pca) / 2)):
        team = data_pca[i]
        ax.scatter(team[0], team[1], c=color[label[i]], alpha=0.5)
        if node[label[i]] == 'nan':
            node[label[i]] = ax.scatter(team[0], team[1], c=color[label[i]], alpha=0.5)
    ax.legend((node[0], node[1], node[2], node[3]), (u"0类", u"1类", u"2类", u"3类"), loc=0)
    plt.show()


def my_tSNE(X, n_components):
    '''
    t-SNE降维
    '''
    X_tsne = TSNE(n_components=n_components, learning_rate=100).fit_transform(X)
    return X_tsne


def myPCA(X, n_components):
    '''
    PCA 过程
    '''
    print("Extracting the top %d eigenfaces from %d samples"
          % (n_components, X.shape[0]))
    t0 = time()
    pca = PCA(n_components=n_components, svd_solver='auto',
              whiten=True).fit(X)
    print("pca compute done in %0.3fs" % (time() - t0))
    t0 = time()
    X_pca = pca.transform(X)
    print("pca transform done in %0.3fs" % (time() - t0))
    return X_pca


def plot_pca(X):
    """
    pca降维后数据集X整体情况，比如前n个成分的反差占比
    """
    top = 51
    pca = PCA()
    pca.fit(X)
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    print(pca.explained_variance_ratio_)
    plt.plot(pca.explained_variance_ratio_[:top], linewidth=2)
    plt.axis('tight')
    plt.xlabel('前50的成分')
    plt.ylabel('成分方差占比')
    plt.show()


if __name__ == '__main__':
    train_path = 'D:/workCode/xRay/xRayData/mask_feature/copyBefore/label_mask/train_label_mask.pkl'
    test_path = 'D:/workCode/xRay/xRayData/mask_feature/copyBefore/label_mask/val_label_mask.pkl'
    with open(train_path, 'rb')as pkl_file:
        train = pickle.load(pkl_file)
    with open(test_path, 'rb')as pkl_file:
        test = pickle.load(pkl_file)

    # shuffle
    np.random.shuffle(train)
    np.random.shuffle(test)
    X = []
    Y = []

    # img 为1024*7*7 分类器最高支持1维数据
    for img, label in train:
        X.append(img.reshape((-1)))
        Y.append(label.numpy()[0])

    for img, label in test:
        X.append(img.reshape((-1)))
        Y.append(label.numpy()[0])

    X = np.array(X)
    X = Normalizer().fit_transform(X)
    # 观察pca情况
    # plot_pca(X)

    # 降维
    X_reduction = myPCA(X, n_components=3)
    # 可视化画图
    visulizition3D(X_reduction, Y)
