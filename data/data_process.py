'''
数据预处理，主要是讲数据进行划分，训练集和测试集
'''

import argparse
from main import *
import os

parser = argparse.ArgumentParser(description="data split")
parser.add_argument('-r', '--ratio', type=float, default=0.2)  # 将数据集划分为测试集，以及验证集
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

    seeg = seegdata()

