# -*- coding: utf-8 -*-
"""
   Author:       Hejia
   Date:         2019/3/24

Description:
对mask feature 降维进行训练并测试

1.运行slect_model_param()对挑选合适的模型参数, 网格搜索最好的模型,
2.运行learning_curve() 找到目标的模型然后观察：画出learning curve

notice：
直接用验证集代替测试集,没有测试集
"""
from pylab import mpl
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve, ShuffleSplit, KFold, train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from time import time
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def plot_confusion_matrix(cm, classes, title, normalize=True, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=18,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    画学习曲线
    参考：https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.
    html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Marco-Recall score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='recall_macro')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def slect_model_param(X, Y, estimator):
    """
    用网格搜索选择模型参数，包括pca参数，模型参数
    最好用Spyder或者ipython notbook，将estimator.cv_results_转换为dataFrame进行观察，可以看到所有指标和参数情况

    关于GridSearchCV的scoring指标参考：
    https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    """
    pca = PCA()
    t0 = time()
    if estimator == 'lr':
        logistic = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
        n_components = [4, 8, 16]
        Cs = np.logspace(-1, 2, 4)
        # ‘__’ separated parameter names
        # 多分类中micro-recall 和 accuracy好像是等价的???
        estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, logistic__C=Cs), scoring='recall_macro',
                                 cv=5)
    elif estimator == 'svc':
        svc = SVC(kernel='rbf', class_weight='balanced')
        pipe = Pipeline(steps=[('pca', pca), ('svc', svc)])
        n_components = [8, 16]
        param_grid = {'C': [1e3, 1e4, 1e5],
                      'gamma': [0.001, 0.01, 0.1], }
        estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, svc__C=param_grid['C'],
                                            svc__gamma=param_grid['gamma']), scoring='recall_macro', cv=5)
    elif estimator == 'rf':

        rf = RandomForestClassifier(min_samples_split=3, min_samples_leaf=3)
        pipe = Pipeline(steps=[('pca', pca), ('rf', rf)])
        n_components = [4, 8, 16]
        param_grid = {'n_estimators': range(5, 31, 5)}

        estimator = GridSearchCV(pipe,
                                 dict(pca__n_components=n_components, rf__n_estimators=param_grid['n_estimators']),
                                 scoring='recall_macro', cv=5)

    else:
        raise Exception("指定模型！lr, svc, rf")

    estimator.fit(X, Y)
    print("GridSearchCV done in %0.3fs" % (time() - t0))
    print("all:")
    # print(estimator.cv_results_)
    df_cv_results = pd.DataFrame(estimator.cv_results_)
    return estimator, df_cv_results


def get_learning_curve(X, Y, estimator, title):
    """
      模型参数选好后，训练和画学习曲线
    """
    # 按目标模型参数设定
    pca_n_components = {'lr': 16, 'svc': 16, 'rf': 16}
    pca = PCA(copy=True, iterated_power='auto', n_components=pca_n_components[estimator], random_state=None,
              svd_solver='auto', tol=0.0,
              whiten=False)

    if estimator == 'lr':
        estimator = LogisticRegression(C=100.0, random_state=0, solver='lbfgs', multi_class='multinomial')
    elif estimator == 'svc':
        estimator = SVC(kernel='rbf', class_weight='balanced', C=10000, gamma=0.001)
    elif estimator == 'rf':
        estimator = RandomForestClassifier(n_estimators=20, min_samples_split=3, min_samples_leaf=3)

    else:
        raise Exception("指定模型！lr, svc, rf")

    # pca
    t0 = time()
    pca.fit(X)
    X_pca = pca.transform(X)
    print("pca done in %0.3fs" % (time() - t0))

    t0 = time()
    cv = KFold(n_splits=5)
    plot_learning_curve(estimator, title, X_pca, Y, (0.4, 0.9), cv=cv, n_jobs=4)
    print("learning cure done in %0.3fs" % (time() - t0))
    plt.show()

    # 接着画一次混淆矩阵,肺交叉验证的形式，有随机性
    get_confusion_matrix(X_pca, Y, estimator, title)


def get_confusion_matrix(X, Y, estimator, title):
    print("Predicting ")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=6)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['0期', '1期', '2期', '3期']))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])
    plot_confusion_matrix(cm, classes=[0, 1, 2, 3], title=title)


if __name__ == '__main__':
    train_path = 'D:/workCode/xRay/xRayData/mask_feature/copyBefore/pred_mask/train_label_mask.pkl'
    test_path = 'D:/workCode/xRay/xRayData/mask_feature/copyBefore/pred_mask/val_label_mask.pkl'
    with open(train_path, 'rb')as pkl_file:
        train = pickle.load(pkl_file)
    with open(test_path, 'rb')as pkl_file:
        test = pickle.load(pkl_file)

    # shuffle train，test 均为[(feature，label),...]
    np.random.shuffle(train)
    np.random.shuffle(test)
    X = []
    Y = []
    # feature为1024*7*7 分类器最高支持1维数据
    for img, label in train:
        X.append(img.reshape((-1)))
        Y.append(label.numpy()[0])

    for img, label in test:
        X.append(img.reshape((-1)))
        Y.append(label.numpy()[0])

    X = np.array(X)
    X = Normalizer().fit_transform(X)

    # #step1: 网格搜索出最好的模型，这里最好用交互式编程
    # estimator_cv, df_cv_results = slect_model_param(X, Y, 'rf')
    # print(df_cv_results)

    # #step2:观察目标的模型的学习曲线并画出混淆矩阵
    # Logistic regression, SVM, Random Forest
    get_learning_curve(X, Y, estimator='rf', title='Random Forest')
