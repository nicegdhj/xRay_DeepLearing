# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         2019/3/13

Description:  2-step方案拿到两阶段的输出数据，选取阶段1的敏感性指标

"""

import pandas as pd
import numpy as np


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


file1 = pd.read_csv('D:/workCode/xRay/xRay_DeepLearing/part1DNN/cnf_matrix/record2step/2classes_record_vgg29.csv')
file2 = pd.read_csv(
    'D:/workCode/xRay/xRay_DeepLearing/part1DNN/cnf_matrix/record2step/4classesData_3classes_record_vgg30.csv')

# file1.drop(columns='Unnamed: 0', inplace=True)
# file2.drop(columns='Unnamed: 0', inplace=True)

# file1['score_list'] = file1.score.apply(str2list)
# file1.drop(columns='score', inplace=True)
# 上面两句可以一起写成 
file1['score'] = file1.score.apply(str2list)
file1['score_softmax'] = file1['score'].apply(softmax)

# a = file1['score_list'].iloc[0].apply(softmax)
# dataFrame 定位 df.loc[行索引， 【列名1，列名2】] 行索引是值 1:10 是指包括10，
for i in np.arange(0.05, 1, 0.05):
    i = '%.2f'%i
    col_name = 'predict_threshold_{}'.format(i)
    file1[col_name] = file1['score_softmax'].apply(thresholdSelect, args=(float(i),))
