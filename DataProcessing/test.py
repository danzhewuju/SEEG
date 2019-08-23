#!/usr/bin/Python
"""
@File    : test.py
@Time    : 2019/8/23 20:10
@Author  : Alex
@Software: PyCharm
@Function:
"""

import json
import os

if __name__ == '__main__':
    config = json.load(open("../DataProcessing/config/fig.json"))
    print(config["transferdata.save_split_data_test.path_dir"])
    # json_str = json.load(open('../DataProcessing/config/fig.json'))
    # path_dir = json_str["transferdata.save_split_data_test.path_dir"]
    # print(path_dir)

