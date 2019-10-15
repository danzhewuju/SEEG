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


def matrix_normalization_recorder(data, resize_shape=(130, 200)):  # 这个是矩阵初始化的记录版
    '''
        矩阵的归一化，主要是讲不通形状的矩阵变换为特定形状的矩阵, 矩阵的归一化主要是更改序列
        也就是主要更改行
        eg:(188, 200)->(130, 200)   归一化的表示
        :param data:
        :param resize_shape:
        :return:
        '''
    recoder = []  # [-1/1, 10, 11, 12, 14]  第一位为标识位， -1 标识删除了的信道， +1 标识增加了的信道， 后面对应的是增加或者删除的信道
    recoder.append(0)
    data_shape = data.shape  # 这个必须要求的是numpy的文件格式
    if data_shape[0] != resize_shape[0]:
        if resize_shape[0] > data_shape[0]:  # 做插入处理
            '''
            扩大原来的矩阵
            '''
            d = resize_shape[0] - data_shape[0]
            channels_add = random.sample(range(1, data_shape[0] - 1), d)
            recoder[0] = 1
            recoder += channels_add
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
                recoder[0] = -1
                recoder += channels_del
                data = np.delete(data, channels_del, axis=0)
    return data, recoder


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
    paths = glob.glob(os.path.join(path, '*.' + suffix))
    return paths


def get_matrix_max_location(mtx_data, k, reverse=True):
    '''

    :param mtx_data: 矩阵的数据
    :param k: 获取前K 个最大、最小
    :param reverse: True : 最大， False: 最小
    :return:[(0, 0), (2, 1), (2, 2), (1, 1), (1, 2)] 结果是按照顺序进行排序
    '''
    d_f = mtx_data.flatten()
    if reverse:
        index_id = d_f.argsort()[-k:]
    else:
        index_id = d_f.argsort()[k:]
    x_index, y_index = np.unravel_index(index_id, mtx_data.shape)
    location = list(zip(x_index, y_index))  # 此时只是选取了最大几个，数据之间是没有顺序的
    location_dic = {}
    for x, y in location:
        location_dic[(x, y)] = mtx_data[x][y]

    location_dic = sorted(location_dic.items(), key=lambda x: -x[1]) if reverse else sorted(location_dic.items(),
                                                                                            key=lambda x: x[1])
    result = [x[0] for x in location_dic]
    return result


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
    result = cv2.merge([result])  # 将信道进行整合
    return result


def time_add(h, m, s, seconds_add):
    '''

    :param h: 小时
    :param m: 分钟
    :param s: 秒
    :param seconds_add: 需要增加的时间秒数
    :return: 返回的是绝对的时间
    '''
    s += seconds_add
    s_m = s // 60
    s = s % 60

    m += s_m
    m_h = m // 60
    m = m % 60

    h += m_h

    return int(h), int(m), s


class IndicatorCalculation():  # 包含二分类中各种指标
    def __init__(self, prediction=None, ground_truth=None):
        if prediction is not None and ground_truth is not None:
            self.prediction = prediction  # [0, 1, 0, 1, 1, 0]
            self.ground_truth = ground_truth  # [0, 1, 0, 0, 1 ]

    def __tp(self):
        TP = 0
        for i in range(len(self.prediction)):
            TP += 1 if self.prediction[i] == 1 and self.ground_truth[i] == 1 else 0
        return TP

    def __fp(self):
        FP = 0
        for i in range(len(self.prediction)):
            FP += 1 if self.prediction[i] == 0 and self.ground_truth[i] == 1 else 0
        return FP

    def __fn(self):
        FN = 0
        for i in range(len(self.prediction)):
            FN += 1 if self.prediction[i] == 1 and self.ground_truth[i] == 0 else 0
        return FN

    def __tn(self):
        TN = 0
        for i in range(len(self.prediction)):
            TN += 1 if self.prediction[i] == 0 and self.ground_truth[i] == 0 else 0
        return TN

    def set_values(self, prediction, ground_truth):
        self.prediction = prediction
        self.ground_truth = ground_truth

    def get_accuracy(self):
        return (self.__tp() + self.__tn()) / (self.__tn() + self.__tp() + self.__fn() + self.__fp())

    def get_recall(self):
        return self.__tp() / (self.__tp() + self.__fp())

    def get_precision(self):
        return self.__tp() / (self.__tp() + self.__fn())

    def get_f1score(self):
        return (2 * self.get_recall() * self.get_precision()) / (self.get_recall() + self.get_precision())


if __name__ == '__main__':
    for i in range(10):
        a = np.random.randint(0, 10, (3, 3))
        print(a)
        print(get_matrix_max_location(a, 1))
