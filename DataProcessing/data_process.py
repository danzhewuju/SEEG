'''
数据预处理，主要是讲数据进行划分，训练集和测试集以及验证集，划分的数据集用于few-shot learning, cnn 的训练效果
训练集的划分主要是根据参数来确定
'''

import argparse
import os
import random

import numpy as np
from dataset_info import up_sampling, sampling_rewrite
import sys

sys.path.append("../")

from RelationNet import *
import json

parser = argparse.ArgumentParser(description="data split")
parser.add_argument('-r', '--ratio', type=float, default=0.6)  # 将数据集划分为测试集，以及验证集
parser.add_argument('-v', '--val', type=float, default=0.2)
args = parser.parse_args()

# Hyper Parameters

TRAIN_RATIO = args.ratio
VAL_RATIO = args.val
config = json.load(open("./config/fig.json", 'r'))
patient_test = config['patient_test']
print("test patient is :{}".format(patient_test))


def data_process():
    '''
    function： 混合式的数据划分，
    :return:
    '''
    train_folder = "../data/seeg/mixed_data/{}/train".format(patient_test)
    test_folder = "../data/seeg/mixed_data/{}/test".format(patient_test)
    val_folder = '../data/seeg/mixed_data/{}/val'.format(patient_test)

    if os.path.exists(train_folder) is not True:
        os.makedirs(train_folder)
    else:
        os.system("rm -r ../data/seeg/mixed_data/{}/train/*".format(patient_test))
    if os.path.exists(test_folder) is not True:
        os.makedirs(test_folder)
    else:
        os.system("rm -r ../data/seeg/mixed_data/{}/test/*".format(patient_test))
    if os.path.exists(val_folder) is not True:
        os.makedirs(val_folder)
    else:
        os.system("rm -r ../data/seeg/mixed_data/{}/val/*".format(patient_test))

    path_normal = "sleep_normal"
    path_pre_seizure = "pre_zeizure"
    path_awake = "awake"

    train_folder_dir_normal = os.path.join(train_folder, path_normal)
    train_folder_dir_pre = os.path.join(train_folder, path_pre_seizure)
    # train_folder_dir_awake = os.path.join(train_folder, path_awake)

    test_folder_dir_normal = os.path.join(test_folder, path_normal)
    test_folder_dir_pre = os.path.join(test_folder, path_pre_seizure)
    # test_folder_dir_awake = os.path.join(test_folder, path_awake)

    val_folder_dir_normal = os.path.join(val_folder, path_normal)
    val_folder_dir_pre = os.path.join(val_folder, path_pre_seizure)
    # val_folder_dir_awake = os.path.join(val_folder, path_awake)

    if os.path.exists(train_folder_dir_normal) is not True:
        os.makedirs(train_folder_dir_normal)
    if os.path.exists(train_folder_dir_pre) is not True:
        os.makedirs(train_folder_dir_pre)
    # if os.path.exists(train_folder_dir_awake) is not True:
    #     os.makedirs(train_folder_dir_awake)

    if os.path.exists(test_folder_dir_normal) is not True:
        os.makedirs(test_folder_dir_normal)
    if os.path.exists(test_folder_dir_pre) is not True:
        os.makedirs(test_folder_dir_pre)
    # if os.path.exists(test_folder_dir_awake) is not True:
    #     os.makedirs(test_folder_dir_awake)

    if os.path.exists(val_folder_dir_normal) is not True:
        os.makedirs(val_folder_dir_normal)
    if os.path.exists(val_folder_dir_pre) is not True:
        os.makedirs(val_folder_dir_pre)
    # if os.path.exists(val_folder_dir_awake) is not True:
    #     os.makedirs(val_folder_dir_awake)

    seeg = seegdata()
    tmp_normal = seeg.get_all_path_by_keyword('sleep')
    sleep_normal = []  # 正常人的睡眠时间
    for dp in tmp_normal.values():
        for p in dp:
            sleep_normal.append(p)
    sleep_pre = seeg.get_all_path_by_keyword('preseizure')
    sleep_pre_seizure = []

    for dp in sleep_pre.values():  # 获取的是所有的癫痫发作前数据
        for p in dp:
            sleep_pre_seizure.append(p)

    # print("normal sleep:{} pre seizure:{} awake:{} ".format(len(sleep_label0), len(sleep_label1), len(awake_label2)))
    # 三分类
    print("normal sleep:{} pre seizure:{}".format(len(sleep_normal), len(sleep_pre_seizure)))
    random.shuffle(sleep_normal)
    random.shuffle(sleep_pre_seizure)
    # random.shuffle(awake_label2)
    min_data = min(len(sleep_normal), len(sleep_pre_seizure))  # 让两个数据集的个数相等
    sleep_label1 = sleep_pre_seizure[:min_data]
    sleep_label0 = sleep_normal[:min_data]
    train_num = int(TRAIN_RATIO * len(sleep_label0))
    test_num = int(VAL_RATIO * len(sleep_label0))

    print("train number:{}, test number:{}, val number:{}".format(train_num, test_num,
                                                                  len(sleep_label0) - test_num - train_num))

    for (i, p) in enumerate(sleep_label0):
        name = p.split('/')[-1]
        d = np.load(p)
        if i <= int(train_num):
            save_path = os.path.join(train_folder_dir_normal, name)
        else:
            if i < (train_num + test_num):
                save_path = os.path.join(test_folder_dir_normal, name)
            else:
                save_path = os.path.join(val_folder_dir_normal, name)

        np.save(save_path, d)

    print("Successfully write for normal sleep data!!!")
    for (i, p) in enumerate(sleep_label1):
        name = p.split('/')[-1]
        d = np.load(p)
        if i <= int(train_num):
            save_path = os.path.join(train_folder_dir_pre, name)
        else:
            if i <= (train_num + test_num):
                save_path = os.path.join(test_folder_dir_pre, name)
            else:
                save_path = os.path.join(val_folder_dir_pre, name)

        np.save(save_path, d)
    print("Successfully write for pre seizure sleep data!!!")

    # for (i, p) in enumerate(awake_label2):
    #     name = p.split('/')[-1]
    #     d = np.load(p)
    #     if i <= int(len(awake_label2) * TRAIN_RATIO):
    #         save_path = os.path.join(train_folder_dir_awake, name)
    #     else:
    #         if i <= int(len(awake_label2) * (TRAIN_RATIO + TRAIN_RATIO)):
    #             save_path = os.path.join(test_folder_dir_awake, name)
    #         else:
    #             save_path = os.path.join(val_folder_dir_awake, name)
    #
    #     np.save(save_path, d)
    # print("Successfully write for awake data!!!")


def data_process_n_1():
    '''
    function: 数据划分流程：
              1. 将数据集进行划分，训练的数据集来自于前n-1个人
              2. 然后利用第n个人的数据进行测试，其中第n个人的数据集并没有见过
    :return:
    '''
    TRAIN_RATIO = 0.7
    TEST_RATIO = 0.3

    resampling_base_size = 3000
    dataset_dir = '../data/seeg/zero_data/{}'.format(patient_test)  # 当前所在的数据集, 不同的方法会在不同的数据集上
    train_folder = os.path.join(dataset_dir, 'train')
    test_folder = os.path.join(dataset_dir, 'test')
    val_folder = os.path.join(dataset_dir, 'val')

    if os.path.exists(train_folder) is not True:
        os.makedirs(train_folder)
    else:
        os.system("rm -r {}".format(train_folder))
    if os.path.exists(test_folder) is not True:
        os.makedirs(test_folder)
    else:
        os.system("rm -r {}".format(test_folder))
    if os.path.exists(val_folder) is not True:
        os.makedirs(val_folder)
    else:
        os.system("rm -r {}".format(val_folder))

    path_normal = "sleep_normal"
    path_pre_seizure = "pre_zeizure"
    train_folder_dir_normal = os.path.join(train_folder, path_normal)
    train_folder_dir_pre = os.path.join(train_folder, path_pre_seizure)
    # train_folder_dir_awake = os.path.join(train_folder, path_awake)

    test_folder_dir_normal = os.path.join(test_folder, path_normal)
    test_folder_dir_pre = os.path.join(test_folder, path_pre_seizure)
    # test_folder_dir_awake = os.path.join(test_folder, path_awake)

    val_folder_dir_normal = os.path.join(val_folder, path_normal)
    val_folder_dir_pre = os.path.join(val_folder, path_pre_seizure)
    # val_folder_dir_awake = os.path.join(val_folder, path_awake)

    if os.path.exists(train_folder_dir_normal) is not True:
        os.makedirs(train_folder_dir_normal)
    if os.path.exists(train_folder_dir_pre) is not True:
        os.makedirs(train_folder_dir_pre)
    # if os.path.exists(train_folder_dir_awake) is not True:
    #     os.makedirs(train_folder_dir_awake)

    if os.path.exists(test_folder_dir_normal) is not True:
        os.makedirs(test_folder_dir_normal)
    if os.path.exists(test_folder_dir_pre) is not True:
        os.makedirs(test_folder_dir_pre)
    # if os.path.exists(test_folder_dir_awake) is not True:
    #     os.makedirs(test_folder_dir_awake)

    if os.path.exists(val_folder_dir_normal) is not True:
        os.makedirs(val_folder_dir_normal)
    if os.path.exists(val_folder_dir_pre) is not True:
        os.makedirs(val_folder_dir_pre)
    # if os.path.exists(val_folder_dir_awake) is not True:
    #     os.makedirs(val_folder_dir_awake)

    # n-1 机制，前面的人进行训练，后面人的数据进行验证
    seeg = seegdata()
    tmp_normal = seeg.get_all_path_by_keyword('sleep')
    val_normal = []  # 将这个人的数据最为一个验证的数据集
    sleep_normal = []  # 正常人的睡眠时间
    test_persons_normal_sleep = [patient_test]
    sleep_pre = seeg.get_all_path_by_keyword('preseizure')
    sleep_pre_seizure = []
    val_pre_seizure = []
    test_persons_pre_seizure = [patient_test]  # 用于测试的病人的数据

    m_normal = len(tmp_normal) - 1
    n_preseizure = len(sleep_pre) - 1
    resampling_size_normal = int((n_preseizure / m_normal) * resampling_base_size)
    resampling_size_preseizure = resampling_base_size
    print("size of sampling normal sleep:{}  size of sampling preseizure:{}".format(resampling_size_normal,
                                                                                    resampling_size_preseizure))

    for key, dp in tmp_normal.items():

        if key not in test_persons_normal_sleep:
            result = up_sampling(dp, resampling_size_normal)
            for p in result.items():
                sleep_normal.append(p)
        else:
            result = up_sampling(dp, resampling_base_size)
            for p in result.items():
                val_normal.append(p)
    # 用于从采样的方法
    # min_length_pre_seizure = min([len(x) for key, x in sleep_pre.items() if key not in test_persons_pre_seizure])
    # print("{} sample of per pre seizure sleeping  person.".format(min_length_pre_seizure))
    for key, dp in sleep_pre.items():  # 获取的是所有的癫痫发前数据
        # 使用重采样的方法

        if key not in test_persons_pre_seizure:
            result = up_sampling(dp, resampling_size_preseizure)
            for p in result.items():
                sleep_pre_seizure.append(p)  # 加入的是字典
        else:
            result = up_sampling(dp, resampling_base_size)
            for p in result.items():
                val_pre_seizure.append(p)
    random.shuffle(sleep_normal)
    random.shuffle(sleep_pre_seizure)  # 用于训练的数据集

    random.shuffle(val_normal)  # 用于验证的数据集
    random.shuffle(val_pre_seizure)

    # 接下来是数据集的划分
    train_normal_number = int(TRAIN_RATIO * len(sleep_normal))
    train_preseizure_number = int(TRAIN_RATIO * len(sleep_pre_seizure))

    train_normal_data_dict = sleep_normal[:train_normal_number]
    test_normal_data_dict = sleep_normal[train_normal_number:]

    train_preseizure_data_dict = sleep_pre_seizure[:train_preseizure_number]
    test_preseizure_data_dict = sleep_pre_seizure[train_preseizure_number:]

    # 训练街和测试集的划分重写
    sampling_rewrite(train_normal_data_dict, train_folder_dir_normal)
    sampling_rewrite(test_normal_data_dict, test_folder_dir_normal)

    sampling_rewrite(train_preseizure_data_dict, train_folder_dir_pre)
    sampling_rewrite(test_preseizure_data_dict, test_folder_dir_pre)

    # 验证集的采样重写
    sampling_rewrite(val_normal, val_folder_dir_normal)
    sampling_rewrite(val_pre_seizure, val_folder_dir_pre)

    print("-" * 5 + "statistic information" + "-" * 5)
    print("training data number of normal sleep :{} testing data number of normal sleep :{}\n"
          "training data number of preseizure: {}  testing data number of preseizure: {}\n"
          "validation data number of preseizure: {}  validation data number of preseizure: {}\n"
          .format(len(train_normal_data_dict), len(test_normal_data_dict), len(train_preseizure_data_dict),
                  len(test_preseizure_data_dict), len(val_normal), len(val_pre_seizure)))


if __name__ == '__main__':
    # 两种模式， mixed, 前n-1个人的数据
    # 1. 混合式的数据划分，等比划分
    # data_process()
    # 2. zero_data 留一法的数据划分
    data_process_n_1()
