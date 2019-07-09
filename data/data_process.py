'''
数据预处理，主要是讲数据进行划分，训练集和测试集以及验证集，划分的数据集用于few-shot learning, cnn 的训练效果
训练集的划分主要是根据参数来确定
'''

import argparse

from main import *
import os
import numpy as np
import random

parser = argparse.ArgumentParser(description="data split")
parser.add_argument('-r', '--ratio', type=float, default=0.6)  # 将数据集划分为测试集，以及验证集
parser.add_argument('-v', '--val', type=float, default=0.2)
args = parser.parse_args()

# Hyper Parameters

TRAIN_RATIO = args.ratio
VAL_RATIO = args.val


def data_process():
    '''
    function： 混合式的数据划分，
    :return:
    '''
    train_folder = "./seeg/mixed_data/train"
    test_folder = "./seeg/mixed_data/test"
    val_folder = './seeg/mixed_data/val'

    if os.path.exists(train_folder) is not True:
        os.makedirs(train_folder)
    else:
        os.system("rm -r ./seeg/mixed_data/train/*")
    if os.path.exists(test_folder) is not True:
        os.makedirs(test_folder)
    else:
        os.system("rm -r ./seeg/mixed_data/test/*")
    if os.path.exists(val_folder) is not True:
        os.makedirs(val_folder)
    else:
        os.system("rm -r ./seeg/mixed_data/val/*")

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
    dataset_dir = './seeg/zero_data'  # 当前所在的数据集, 不同的方法会在不同的数据集上
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
    test_persons_normal_sleep = ['BDP']
    min_length_normal = min([len(x) for key, x in tmp_normal.items() if key not in test_persons_normal_sleep])
    print("{} sample of per normal sleeping  person.".format(min_length_normal))
    for key, dp in tmp_normal.items():
        dp = random.sample(dp, min_length_normal)  # 从数据中进行随机抽选
        if key not in test_persons_normal_sleep:
            for p in dp:
                sleep_normal.append(p)
        else:
            for p in dp:
                val_normal.append(p)
    sleep_pre = seeg.get_all_path_by_keyword('preseizure')
    sleep_pre_seizure = []
    val_pre_seizure = []
    test_persons_pre_seizure = ['BDP']  # 用于测试的脑电
    min_length_pre_seizure = min([len(x) for key, x in sleep_pre.items() if key not in test_persons_pre_seizure])
    print("{} sample of per pre seizure sleeping  person.".format(min_length_pre_seizure))
    for key, dp in sleep_pre.items():  # 获取的是所有的癫痫发前数据
        dp = random.sample(dp, min_length_pre_seizure)
        if key not in test_persons_pre_seizure:
            for p in dp:
                sleep_pre_seizure.append(p)
        else:
            for p in dp:
                val_pre_seizure.append(p)
    print("normal sleep:{} pre seizure:{}".format(len(sleep_normal), len(sleep_pre_seizure)))
    random.shuffle(sleep_normal)
    random.shuffle(sleep_pre_seizure)

    random.shuffle(val_normal)
    random.shuffle(val_pre_seizure)
    # random.shuffle(awake_label2)
    min_data = min(len(sleep_normal), len(sleep_pre_seizure))  # 让两个数据集的个数相等
    sleep_label1 = sleep_pre_seizure[:min_data]
    sleep_label0 = sleep_normal[:min_data]
    min_length_val = min(len(val_normal), len(val_pre_seizure))
    val_normal = val_normal[:min_length_val]
    val_pre_seizure = val_pre_seizure[:min_length_val]

    train_num = int(TRAIN_RATIO * len(sleep_label0))
    test_num = int((VAL_RATIO+VAL_RATIO) * len(sleep_label0))

    print("train number:{}, test number:{}, val number:{}".format(train_num, test_num,len(val_normal)))

    # 训练集和测试集的划分
    for (i, p) in enumerate(sleep_label0):
        name = p.split('/')[-1]
        d = np.load(p)
        if i <= int(train_num):
            save_path = os.path.join(train_folder_dir_normal, name)
        else:
            save_path = os.path.join(test_folder_dir_normal, name)
        np.save(save_path, d)

    for i, p in enumerate(val_normal):
        name = p.split('/')[-1]
        d = np.load(p)
        save_path = os.path.join(val_folder_dir_normal, name)
        np.save(save_path, d)

    # 训练集和测试集的划分
    print("Successfully write for normal sleep data!!!")
    for (i, p) in enumerate(sleep_label1):
        name = p.split('/')[-1]
        d = np.load(p)
        if i <= int(train_num):
            save_path = os.path.join(train_folder_dir_pre, name)
        else:
            save_path = os.path.join(test_folder_dir_pre, name)
        np.save(save_path, d)
    for i, p in enumerate(val_pre_seizure):
        name = p.split('/')[-1]
        d = np.load(p)
        save_path = os.path.join(val_folder_dir_pre, name)
        np.save(save_path, d)
    print("Successfully write for pre seizure sleep data!!!")


if __name__ == '__main__':
    # 两种模式， mixed, 前n-1个人的数据
    # data_process()
    data_process_n_1()
