# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         2019/3/13

Description:  2-step方案拿到两阶段的输出数据，选取阶段1的敏感性指标

"""
from pylab import mpl

import pandas as pd
import numpy as np
import copy
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


# mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
# mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def str2list(my_str):
    "字符串转化为list '[0.1, 0.2]'-->[0.1, 0.2] "
    my_str = my_str[1:-1].split(",")
    return [float(i) for i in my_str]


def softmax(x):
    "计算softmax"
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def thresholdSelect(l, threshold):
    if l[1] > threshold:
        return 1
    return 0


def fixFile2Predict(x):
    '''
    三分类器（1,2,3类）对原始所有的4分类的数据集进行打分
    原本是标签是（0,1,2,3），但是3类器在它的只能是（0,1,2）对比代表了4分类中的(1,2,3)。
    所以都加1
    '''
    return x + 1


def plot_cnfMatrix(y_true, y_predict, threshold, classes=[0, 1, 2, 3], normalize=False, title='Confusion matrix',
                   cmap=plt.cm.Blues):
    """
    画混淆矩阵 与 敏感性系数与平均召回率曲线
    """
    cm = confusion_matrix(y_true, y_predict, labels=classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(
        'D:/workCode/xRay/xRay_DeepLearing/part1DNN/cnf_matrix/record2step/plt_sensititve/{}_sensitive_cnf_rate.png'.format(
            threshold))


if __name__ == '__main__':

    file1 = pd.read_csv(
        'D:/workCode/xRay/xRay_DeepLearing/part1DNN/cnf_matrix/record2step/2classes_record_densenet27.csv')
    file2 = pd.read_csv(
        'D:/workCode/xRay/xRay_DeepLearing/part1DNN/cnf_matrix/record2step/4classesData_3classes_record_densenet28.csv')

    # file1.drop(columns='Unnamed: 0', inplace=True)
    # file2.drop(columns='Unnamed: 0', inplace=True)

    # file1['score_list'] = file1.score.apply(str2list)
    # file1.drop(columns='score', inplace=True)
    # 上面两句可以一起写成
    file1['score'] = file1.score.apply(str2list)
    file1['score_softmax'] = file1['score'].apply(softmax)

    file2['predict'] = file2['predict'].apply(fixFile2Predict)

    # a = file1['score_list'].iloc[0].apply(softmax)
    # dataFrame 定位 df.loc[行索引， 【列名1，列名2】] 行索引是值 1:10 是指包括10，

    ground_truth_kv = {}
    predict_kv = {}
    recall_kv = {}

    for index, row in file2.iterrows():
        img_name = row['path'].split('/')[-1]
        ground_truth_kv[img_name] = row['label']
        predict_kv[img_name] = row['predict']

    for i in np.arange(0.5, 0.55, 0.05):
        # 避免前一个i的已经修改过的predict_kv影响现在的predict_kv
        predict_kv_copy = copy.deepcopy(predict_kv)

        i = '%.2f' % i
        col_name = 'predict_threshold_{}'.format(i)
        # 超过阈值就判定为正类（1，患病类）
        file1[col_name] = file1['score_softmax'].apply(thresholdSelect, args=(float(i),))

        for index, row in file1.iterrows():
            img_name = row['path'].split('/')[-1]
            # 如果在这里被预测为负(0类)，则覆盖predict_kv中被错误分类的
            # 因为predict_kv之前是3分类器对4分类数据进行假分类
            if row[col_name] == 0:
                predict_kv_copy[img_name] = row[col_name]

        # 宏召回率计算
        true_list = []
        predict_list = []
        for k, v in ground_truth_kv.items():
            true_list.append(v)
            predict_list.append(predict_kv_copy[k])

        epoch_acc = recall_score(true_list, predict_list, average='macro')
        recall_kv[float(i)] = epoch_acc
        plot_cnfMatrix(true_list, predict_list, threshold=i, normalize=True)
    print(recall_kv)
