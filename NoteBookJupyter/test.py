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

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
print(X)
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
labels = kmeans.labels_
print(labels)
score = davies_bouldin_score(X, labels)

print(score)

# path_data = "./data/feature_true_id_prediction.pkl"
# feature_true_id_prediction = np.load(path_data, allow_pickle=True)
# print(feature_true_id_prediction)
