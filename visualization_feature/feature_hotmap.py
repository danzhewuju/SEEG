#!/usr/bin/Python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/30 19:38
# @Author  : Alex
# @Site    : 
# @File    : feature_hotmap.py
# @Software: PyCharm

from tqdm import tqdm
import sys

from grad_cam import *

sys.path.append('../')

from util.util_file import *
import json


def select_examplea(name="LK", number=1000, path_dir="../data/data_slice/split/preseizure",
                    json_path="./json_path/LK_preseizure_path.json"):
    '''

    :param name: 患者名称
    :param number: 选择重采样的样本
    :param path_dir: 所在的路径
    :param json_path: json 文件存储的位置
    :return:
    :function 重采样，随机选择样本
    '''
    path_dict = get_all_file_path(path_dir, "npy")
    path_LK = path_dict[name]
    select_data = random.sample(path_LK, number)
    json_str = json.dumps(select_data)
    f = open(json_path, 'w')
    f.write(json_str)
    f.close()
    print("select {} patients from {}".format(number, name))


def example_similarity(path_json, json_path="./json_path/LK_preseizure_sorted.json"):
    '''

    :param path_json:  重采样中的选择的样本数
    :param json_path:  选择比较有代表性的样本
    :return:
    '''
    with open(path_json) as f:
        result_score = []
        json_data = json.load(f)
        length = len(json_data)
        for i in tqdm(range(length)):
            data1 = np.load(json_data[i])
            score = 0.0
            for j in range(length):
                data2 = np.load(json_data[j])
                score += mtx_similarity(data1, data2)
            score /= length
            result_score.append(score)
        dict_map = dict(zip(range(length), result_score))
        dict_map_sorted = sorted(dict_map.items(), key=lambda x: -x[1])
        print(dict_map_sorted)
        sample_high_score = [json_data[x] for x, y in dict_map_sorted]
        ff = open(json_path, 'w')
        json.dump(sample_high_score, ff)
        ff.close()
    print("all information has been written in json file")


if __name__ == '__main__':
    # # 样本的选择
    # select_examplea()
    # example_similarity("./json_path/LK_preseizure_path.json")  # 相似度的计算并排序

    # 根据计算的相似度来进行热力图热点计算
    f = open('./json_path/LK_preseizure_sorted.json', 'r')
    clean_dir('./heatmap')
    clean_dir('./raw_data_signal')
    count = 300
    data = f.read()
    json_data = json.loads(data)
    for i in range(count):
        get_feature_map(json_data[i], location_name="top_similarity_signal.txt") # 将相关的进行信号进行处理
