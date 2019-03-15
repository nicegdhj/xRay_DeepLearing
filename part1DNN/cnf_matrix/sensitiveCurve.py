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

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


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


def fixfile2(x):
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


def plot_Curve(recall_kv, save_img_name):
    '''
    画截断点曲线
    '''
    k_sorted = sorted(recall_kv.keys())
    v = [recall_kv[i] for i in k_sorted]
    # key 精确到小数点后两位， v精确到小数点后4位
    k_sorted = list(map(lambda x: float('%.2f' % x), k_sorted))
    v = list(map(lambda x: float('%.4f' % x), v))

    plt.figure(figsize=[10, 5])
    plt.title('截断点与2-step方法宏召回率关联性表')
    plt.plot(k_sorted, v, 'r', marker='o', label='Macro-Recall')
    # plt.plot(sub_axix, test_acys, color='green', label='testing accuracy')
    # plt.plot(x_axix, train_pn_dis, color='skyblue', label='PN distance')
    # plt.plot(x_axix, thresholds, color='blue', label='threshold')

    for a, b in zip(k_sorted, v):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    plt.xlabel('截断点')
    plt.ylabel('4分类宏召回率')
    # plt.xticks(np.arange(0.05, 1, 0.05))
    plt.savefig(
        'D:/workCode/xRay/xRay_DeepLearing/part1DNN/cnf_matrix/record2step/plt_sensititve/{}.png'.format(save_img_name))


def without_cutPoint_2step(file1, file2):
    '''
    没有加入截断点计算召回率
    '''

    ground_truth_kv = {}
    predict_kv = {}
    tmp_kv = {}

    for index, row in file1.iterrows():
        img_name = row['path'].split('/')[-1]
        # ground_truth_kv现在只有0,1
        ground_truth_kv[img_name] = row['label']
        predict_kv[img_name] = row['predict']
        if row['predict'] == 1:
            tmp_kv[img_name] = row['predict']
            ground_truth_kv[img_name] = '正'

    for index, row in file2.iterrows():
        img_name = row['path'].split('/')[-1]
        if img_name in tmp_kv:
            # 将正（1）类更新为1,2,3具体类
            ground_truth_kv[img_name] = row['label']
            predict_kv[img_name] = row['predict']

    # 宏召回率计算
    true_list = []
    predict_list = []
    for k, v in ground_truth_kv.items():
        true_list.append(v)
        predict_list.append(predict_kv[k])

    epoch_acc = recall_score(true_list, predict_list, average='macro')
    print(epoch_acc)


def cutPoint_2step(file1, file2):
    """
    有加入截断点
    """
    recall_kv = {}

    for i in np.arange(0.05, 1.0, 0.05):
        ground_truth_kv = {}
        predict_kv = {}
        tmp_kv = {}

        i = '%.2f' % i
        col_name = 'predict_threshold_{}'.format(i)
        # 超过阈值就判定为正类（1，患病类）
        file1[col_name] = file1['score_softmax'].apply(thresholdSelect, args=(float(i),))

        for index, row in file1.iterrows():
            img_name = row['path'].split('/')[-1]
            # ground_truth_kv现在只有0,1
            ground_truth_kv[img_name] = row['label']
            predict_kv[img_name] = row[col_name]
            if row[col_name] == 1:
                tmp_kv[img_name] = row[col_name]
                ground_truth_kv[img_name] = '正'

        for index, row in file2.iterrows():
            img_name = row['path'].split('/')[-1]
            if img_name in tmp_kv:
                # 将正（1）类更新为1,2,3具体类
                ground_truth_kv[img_name] = row['label']
                predict_kv[img_name] = row['predict']
            if ground_truth_kv[img_name] not in [0, 1, 2, 3]:
                print(ground_truth_kv[img_name])

        # 宏召回率计算
        true_list = []
        predict_list = []
        for k, v in ground_truth_kv.items():
            true_list.append(v)
            predict_list.append(predict_kv[k])

        macro_recall = recall_score(true_list, predict_list, average='macro')
        recall_kv[float(i)] = macro_recall
        # 画每个截断点的分类混淆矩阵
        # plot_cnfMatrix(true_list, predict_list, threshold=i, normalize=True)
        # plot_Curve(recall_kv, save_img_name='1')
    return recall_kv


if __name__ == '__main__':
    # dataFrame 定位 df.loc[行索引， 【列名1，列名2】] 行索引是值 1:10 是指包括10
    # df.iloc[行索引， [列索引1，列索引1]] iloc 不同于loc 只能是数字，且1:10指的是1到9

    file1 = pd.read_csv(
        'D:/workCode/xRay/xRay_DeepLearing/part1DNN/cnf_matrix/record2step/2classes_record_densenet27.csv')
    file2 = pd.read_csv(
        'D:/workCode/xRay/xRay_DeepLearing/part1DNN/cnf_matrix/record2step/4classesData_3classes_record_densenet28.csv')

    file2_sensitive = pd.read_csv(
        'D:/workCode/xRay/xRay_DeepLearing/part1DNN/cnf_matrix/record2step/4classesData_3classes_record_densenet33_sensitive.csv')

    file1['score'] = file1.score.apply(str2list)
    file1['score_softmax'] = file1['score'].apply(softmax)

    file2['predict'] = file2['predict'].apply(fixfile2)
    file2_sensitive['predict'] = file2_sensitive['predict'].apply(fixfile2)

    # without_cutPoint_2step(file1, file2)
    kv1 = cutPoint_2step(file1, file2_sensitive)
    kv1_sensitive = cutPoint_2step(file1, file2)

    # 画截断点折线图

    k_sorted = sorted(kv1.keys())
    v1 = [kv1[i] for i in k_sorted]
    v2 = [kv1_sensitive[i] for i in k_sorted]
    # key 精确到小数点后两位， v精确到小数点后4位
    k_sorted = list(map(lambda x: float('%.2f' % x), k_sorted))
    v1 = list(map(lambda x: float('%.4f' % x), v1))
    v2 = list(map(lambda x: float('%.4f' % x), v2))

    plt.figure(figsize=[10, 5])
    plt.title('截断点与2-step方法宏召回率关联性表')
    plt.plot(k_sorted, v1, 'r', marker='o', label='未使用代价敏感')
    plt.plot(k_sorted, v2, 'g', marker='v', label='使用代价敏感')

    for a, b in zip(k_sorted, v1):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    for a, b in zip(k_sorted, v2):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    plt.legend()
    plt.xlabel('截断点')
    plt.ylabel('4分类宏召回率')
    # plt.xticks(np.arange(0.05, 1, 0.05))
    plt.savefig(
        'D:/workCode/xRay/xRay_DeepLearing/part1DNN/cnf_matrix/record2step/plt_sensititve/curve.png')
