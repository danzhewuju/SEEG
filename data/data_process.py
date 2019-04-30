'''
数据预处理，主要是讲数据进行划分，训练集和测试集
'''

import argparse
from main import *
import os
import random
import numpy as np

parser = argparse.ArgumentParser(description="data split")
parser.add_argument('-r', '--ratio', type=float, default=0.8)  # 将数据集划分为测试集，以及验证集
args = parser.parse_args()

# Hyper Parameters

RATIO = args.ratio


def data_process():
    train_folder = "./seeg/train"
    test_folder = "./seeg/test"

    if os.path.exists(train_folder) is not True:
        os.makedirs(train_folder)
    if os.path.exists(test_folder) is not True:
        os.makedirs(test_folder)

    path_normal = "sleep_normal"
    path_pre_seizure = "pre_zeizure"

    train_folder_dir_normal = os.path.join(train_folder, path_normal)
    train_folder_dir_pre = os.path.join(train_folder, path_pre_seizure)

    test_folder_dir_normal = os.path.join(test_folder, path_normal)
    test_folder_dir_pre = os.path.join(test_folder, path_pre_seizure)

    if os.path.exists(train_folder_dir_normal) is not True:
        os.makedirs(train_folder_dir_normal)
    if os.path.exists(train_folder_dir_pre) is not True:
        os.makedirs(train_folder_dir_pre)

    if os.path.exists(test_folder_dir_normal) is not True:
        os.makedirs(test_folder_dir_normal)
    if os.path.exists(test_folder_dir_pre) is not True:
        os.makedirs(test_folder_dir_pre)

    seeg = seegdata()
    tmp_normal = seeg.get_all_path_by_keyword('sleep')
    sleep_label0 = tmp_normal['SGH']  # 正常人的睡眠时间
    sleep_pre = seeg.get_all_path_by_keyword('normal')
    sleep_label1 = sleep_pre['LK']  # 发病前的一段时间

    print("0:{} 1:{}".format(len(sleep_label0), len(sleep_label1)))
    random.seed(1)
    random.shuffle(sleep_label0)
    random.shuffle(sleep_label1)
    for (i, p) in enumerate(sleep_label0):
        name = p.split('/')[-1]
        d = np.load(p)
        if i <= int(RATIO * len(sleep_label0)):
            save_path = os.path.join(train_folder_dir_normal, name)
        else:
            save_path = os.path.join(test_folder_dir_normal, name)
        np.save(save_path, d)

    print("Successfully write for normal data!!!")
    for (i, p) in enumerate(sleep_label1):
        name = p.split('/')[-1]
        d = np.load(p)
        if i <= int(RATIO * len(sleep_label1)):
            save_path = os.path.join(train_folder_dir_pre, name)
        else:
            save_path = os.path.join(test_folder_dir_pre, name)
        np.save(save_path, d)
    print("Successfully write for pre sleep data!!!")


if __name__ == '__main__':
    data_process()
