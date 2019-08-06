#!/usr/bin/python
# ---------------------------------
# 文件工具方法， 主要包含常见的文件处理方法
#  以及常见的文件存储的方法
# ---------------------------------
import glob
import os
import random

import cv2
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
    if data_shape[0] != resize_shape[0]:
        if resize_shape[0] > data_shape[0]:  # 做插入处理
            '''
            扩大原来的矩阵
            '''
            d = resize_shape[0] - data_shape[0]
            channels_add = random.sample(range(1, data_shape[0] - 1), d)
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
                d = data_shape[0] - resize_shape[0]
                channels_del = random.sample(range(1, data_shape[0] - 1), d)
                data = np.delete(data, channels_del, axis=0)
    return data


def get_label_data(path):  # get data include label
    '''

    :param path:
    :return: {"path":1, "path2":2}
    '''
    class_name = os.listdir(path)
    data_name = []
    data_label = []
    for i, name in enumerate(class_name):
        new_path = os.path.join(path, name)
        data_file = os.listdir(new_path)
        path_file = [os.path.join(new_path, x) for x in data_file]
        data_name += path_file
        data_label += [i] * len(data_file)
    result_data_label = dict(zip(data_name, data_label))
    return result_data_label


def get_first_dir_path(path, suffix="jpg"):
    names = glob.glob(os.path.join(path, '*.' + suffix))
    result = [os.path.join(path, x) for x in names]
    return result


def get_matrix_max_location(mtx_data, k, reverse=False):
    '''

    :param mtx_data: 矩阵的数据
    :param k: 获取前K 个最大、最小
    :param reverse: True : 最大， False: 最小
    :return:
    '''
    d_f = mtx_data.flatten()
    if reverse is not True:
        index_id = d_f.argsort()[-k:]
    else:
        index_id = d_f.argsort()[k:]
    x_index, y_index = np.unravel_index(index_id, mtx_data.shape)
    location = list(zip(x_index, y_index))
    return location


def mtx_similarity(mtx_a, mtx_b):
    '''

    :param mtx_a:
    :param mtx_b:
    :return:  计算矩阵的相似度
    '''
    mtx_1 = mtx_a.flatten()  # 展开为一维
    mtx_2 = mtx_b.flatten()
    min_length = min(len(mtx_1), len(mtx_2))
    mtx_1 = mtx_1[:min_length]
    mtx_2 = mtx_2[:min_length]
    result = np.dot(mtx_1, mtx_2.T) / (np.linalg.norm(mtx_1) * np.linalg.norm(mtx_2))
    result = abs(result)
    return result


def clean_dir(path):
    print("this is danger operation!")
    clean_path = os.path.join(path, "*")
    print("you will clean all files in {}, do you continued?(y/n)".format(clean_path))
    key = str(input())
    if key == "y":
        print("cleaning the files in {}".format(clean_path))
        os.system("rm -r {}".format(clean_path))
        print("clean finished!")
    else:
        print("cancel operation !")


def trans_numpy_cv2(data):
    data1 = sigmoid(data)
    min_data = np.min(data1)
    data1 = data1 - min_data
    max_data = np.max(data1)
    data1 = data1 / max_data * 255
    result = data1.astype(np.uint8)
    result = cv2.merge([result])
    return result

# if __name__ == '__main__':
#     print(get_label_data("/home/cbd109-3/Users/data/yh/Program/Python/SEEG/data/seeg/zero_data/test"))
