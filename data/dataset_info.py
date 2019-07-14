#!/usr/bin/Python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/14 16:19
# @Author  : Alex
# @Site    : 
# @File    : dataset_info.py
# function statistical seeg dataset information
# @Software: PyCharm


import argparse
import os
import random
import numpy as np
from collections import  Counter


class StatisticSeegDataset:
    def __init__(self, path):
        self.root_path = path

    def dataset_statistics_information(self, path):
        '''
        function: 统计相关的数据信息，数据均衡问题
        :param path:  统计数据的路径
        :return:
        '''
        tree_file = {}  # 生成目录的结构树
        tree_file["root_path"] = path
        for index, dirs in enumerate(os.listdir(path)):
            now_dir_name = {}
            path_dir = os.path.join(path, dirs)
            for name in os.listdir(path_dir):
                p = os.path.join(path_dir, name)
                patient_files = [p for p in os.listdir(p)]
                now_dir_name[name] = patient_files
            tree_file[dirs] = now_dir_name
        self.tree_file = tree_file
        return tree_file

    def get_information(self):  # 获取相关的信息
        self.normal_sleep = self.tree_file["sleep"]
        self.pre_seizure = self.tree_file["preseizure"]
        self.normal_number = {}
        self.preseizure_number = {}

        print("---" * 5 + "NORMAL_SLEEP INFORMATION" + "---" * 5)
        count = []
        for meta_data in self.normal_sleep.items():
            print("name :{}; data_number:{}".format(meta_data[0], len(meta_data[1])))
            count.append(len(meta_data[1]))
            self.normal_number[meta_data[0]] = len(meta_data[1])

        print("average number:{} ".format(sum(count) // len(count)))
        self.normal_number["average number"] = sum(count) // len(count)

        print("---" * 5 + "Pre_Seizure INFORMATION" + "---" * 5)
        count = []
        for meta_data in self.pre_seizure.items():
            print("name :{}; data_number:{}".format(meta_data[0], len(meta_data[1])))
            count.append(len(meta_data[1]))
            self.preseizure_number[meta_data[0]] = len(meta_data[1])
        print("average number:{} ".format(sum(count) // len(count)))
        self.preseizure_number["average number"] = sum(count) // len(count)

        # return self.normal_sleep, self.pre_seizure


def up_sampling(paths, upsampling_size):  # 数据的上采样过方法，只通过连接列表来做
    '''

    :param paths: 采样的对象，采样的数据的路径
    :param upsampling_size: 设置的重采样的大小
    :return:
    '''
    raw_data_size = len(paths)
    result = {}
    if raw_data_size > upsampling_size:
        print("Raw data size {} bigger than up sampling size {}, down sampling ".format(raw_data_size, upsampling_size))
        data_index = random.sample(range(raw_data_size), upsampling_size)
        for d in data_index:
            result[paths[d]] = paths[d]
    else:
        print("Up sampling, the repetition rate is :{:.2f}%".format((upsampling_size-raw_data_size)*100/raw_data_size))
        data_index = np.random.randint(0, raw_data_size, upsampling_size-raw_data_size)
        data_index = list(range(raw_data_size)) + data_index.tolist()
        bit_map = np.zeros(raw_data_size) # 位图， 查看重采样的重复个数
        for d in data_index:
            path = paths[d]
            bit_map[d] += 1 # 修改位图
            pre_name = path.split('.')[0] # 获得名称的前缀
            if bit_map[d] > 1:
                pre_name = pre_name + "-ups-{}".format(bit_map[d])
                full_path = pre_name + ".npy"
            else:
                full_path = path
            result[full_path] = path
        sa = Counter(bit_map)
        print(sa)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="statistical seeg dataset information!")
    parser.add_argument('-rd', '--root_dir', default='../data/seizure/split')
    args = parser.parse_args()
    root_dir = args.root_dir  # 获得根目录的路径
    sta_info = StatisticSeegDataset(root_dir)
    sta_info.dataset_statistics_information(root_dir)
    sta_info.get_information()
    result = up_sampling(sta_info.pre_seizure["LK"], 20000)
    print("{}".format(len(result)))
