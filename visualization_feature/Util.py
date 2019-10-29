#!/usr/bin/Python
"""
@File    : Util.py
@Time    : 2019/9/9 20:48
@Author  : Alex
@Software: PyCharm
@Function:
"""
import pandas as pd
import os
import json

config = json.load(open("./json_path/config.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']
classification = config['classification']


def cal_acc_visualization():
    path = "./log/{}/{}/heatmap.csv".format(patient_test, classification)
    if os.path.exists(path) is False:
        print("{} file is not existed!".format(path))
        exit(0)
    data = pd.read_csv(path, sep=',')
    data_p = data['prediction'].tolist()
    data_g = data['ground truth'].tolist()
    count = 0
    for p in range(len(data_p)):
        if data_p[p] == data_g[p]:
            count += 1
    acc = count / len(data_p)
    print("accuracy: {:.2f}%".format(acc * 100))


if __name__ == '__main__':
    cal_acc_visualization()
