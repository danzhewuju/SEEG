#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/1 20:19
# @Author  : Alex
# @Site    : 
# @File    : test.py
# @Software: PyCharm
import numpy as np
path = "./data/feature_true_id_prediction"
data = np.load(path, allow_pickle=True)
print(data)
