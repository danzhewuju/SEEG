#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/1 20:19
# @Author  : Alex
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import numpy as np
from util.util_file import IndicatorCalculation

# path = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/VMAML/precision/BDP_val_prediction.pkl"
# data = np.load(path, allow_pickle=True)
# pre_list = []
# grount_truth = []
# for id, data in data.items():
#     pre_list.append(data["prediction"])
#     grount_truth.append(data["ground truth"])
#
# print(pre_list)
# print(grount_truth)
# cal = IndicatorCalculation()
# cal.set_values(pre_list, grount_truth)
# print("Accuracy:{} Precision:{} Recall:{} F1score:{}".format(cal.get_accuracy(), cal.get_precision(), cal.get_recall(),
#                                                              cal.get_f1score()))

# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris.data
# print(X)
# from sklearn.cluster import KMeans
# from sklearn.metrics import davies_bouldin_score
# kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
# labels = kmeans.labels_
# print(labels)
# score = davies_bouldin_score(X, labels)
#
# print(score)

# path_data = "./data/feature_true_id_prediction.pkl"
# feature_true_id_prediction = np.load(path_data, allow_pickle=True)
# print(feature_true_id_prediction)

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# 使用mtcars数据集，通过一些数字变量提供几辆汽车的性能参数。
# Data set mtcars数据集 下载
url = 'https://python-graph-gallery.com/wp-content/uploads/mtcars.csv'
df = pd.read_csv(url)
df = df.set_index('model')
# 横轴为汽车性能参数，纵轴为汽车型号
print(df.head())

my_palette = dict(zip(df.cyl.unique(), ["orange","yellow","brown"]))
my_palette
# 列出不同汽车的发动机汽缸数
row_colors = df.cyl.map(my_palette)
row_colors
# metric数据度量方法, method计算聚类的方法
# standard_scale标准维度（0：行或1：列即每行或每列的含义，减去最小值并将每个维度除以其最大值）
sns.clustermap(df, metric="correlation", method="single", cmap="Blues", standard_scale=1, row_colors=row_colors)
sns.clustermap(df, standard_scale=1)
# Normalize 正则化
sns.clustermap(df, z_score=1)
plt.show()

