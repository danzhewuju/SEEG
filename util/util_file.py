#!/usr/bin/python
# ---------------------------------
# 文件工具方法， 主要包含常见的文件处理方法
#  以及常见的文件存储的方法
# ---------------------------------
import glob
import os

import numpy as np


def get_all_file_path(path, suffix='fif'):  # 主要是获取某文件夹下面所有的文件列表
    '''
    :param path: 存储对应文件的路径
    :return: 文件夹下面对应的文件路径
    '''
    names = []
    file_map = {}
    path_dir = []
    dir = os.listdir(path)
    names = dir
    for d in dir:
        path_dir.append(os.path.join(path, d))

    for index, p in enumerate(path_dir):
        new_suffix = '*.' + suffix
        file_p = glob.glob(os.path.join(p, new_suffix))
        file_map[names[index]] = file_p
    return file_map


def matrix_normalization(data, resize_shape=(130, 200)):
    '''
    矩阵的归一化，主要是讲不通形状的矩阵变换为特定形状的矩阵, 矩阵的归一化主要是更改序列
    也就是主要更改行
    eg:(188, 200)->(130, 200)   归一化的表示
    :param data:
    :param resize_shape:
    :return:
    '''
    data_shape = data.shape  # 这个必须要求的是numpy的文件格式
    if resize_shape[0] > data_shape[0]:  # 做插入处理
        '''
        扩大原来的矩阵
        '''
        d = resize_shape[0] - data_shape[0]
        channels_add = np.random.randint(1, data_shape[0] - 1, d)  # 随机的添加的信道列表
        fake_channel = []  # 添加信道列表的值
        for c in channels_add:
            tmp = (data[c - 1] + data[c]) * 1.0 / 2
            fake_channel.append(tmp)
        data = np.insert(data, channels_add, fake_channel, axis=0)
    else:
        if resize_shape[0] < data_shape[0]:  # 做删除处理
            '''
            删除掉原来的矩阵
            '''
            d = data_shape[0]-resize_shape[0]
            channels_del = np.random.randint(1, data_shape[0]-1, d)
            data = np.delete(data, channels_del, axis=0)
    return data
