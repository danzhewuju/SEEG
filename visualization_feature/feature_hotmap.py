#!/usr/bin/Python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/30 19:38
# @Author  : Alex
# @Site    : 
# @File    : feature_hotmap.py
# @Software: PyCharm

import random

from grad_cam import *


def select_examplea(name="LK", number=10):
    path_dir = "../data/data_slice/split/preseizure"
    path_dict = get_all_file_path(path_dir, "npy")
    path_LK = path_dict[name]
    select_data = random.sample(path_LK, number)
    json_str = json.dumps(select_data)
    f = open('./json_path/LK_path.json', 'w')
    f.write(json_str)
    print("select {} patients from {}".format(number, name))


if __name__ == '__main__':
    f = open('./json_path/LK_path.json', 'r')
    data = f.read()
    json_data = json.loads(data)
    for i in range(len(json_data)):
        get_feature_map(json_data[i])

    # get_feature_map()
