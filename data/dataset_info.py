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


def up_sampling(paths):  # 数据的上采样过方法，只通过连接列表来做
    result = []
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CNN parameter setting!")
    parser.add_argument('-rd', '--root_dir', default='../data/seizure/split')
    args = parser.parse_args()
    root_dir = args.root_dir  # 获得根目录的路径
    sta_info = StatisticSeegDataset(root_dir)
    sta_info.dataset_statistics_information(root_dir)
    sta_info.get_information()
    # count = []
    # print("---" * 5 + "NORMAL_SLEEP INFORMATION" + "---" * 5)
    # for meta_data in normal_sleep.items():
    #     print("name :{}; data_number:{}".format(meta_data[0], len(meta_data[1])))
    #     count.append(len(meta_data[1]))
    # print("average number:{} ".format(sum(count) // len(count)))
    # print("---" * 5 + "Pre_Seizure INFORMATION" + "---" * 5)
    #
    # count = []
    # for meta_data in pre_seizure.items():
    #     print("name :{}; data_number:{}".format(meta_data[0], len(meta_data[1])))
    #     count.append(len(meta_data[1]))
    # print("average number:{} ".format(sum(count) // len(count)))

    # test(root_dir)
