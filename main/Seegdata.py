#!/usr/bin/python

import matplotlib.pyplot as plt
import numpy as np
from util import *


class seegdata:
    def __init__(self, path_normal='../data/seizure/split/normal', path_cases='../data/seizure/split/cases'):
        self.path_cases = path_cases
        self.path_normal = path_normal

    def get_split_npy_data(self):
        map_cases = get_all_file_path(self.path_cases, 'npy')
        map_normal = get_all_file_path(self.path_normal, 'npy')
        # print(map_cases)
        # print(map_normal)
        data_map_cases = {}
        for d_map in map_cases.items():
            data = [np.load(x) for x in d_map[1]]
            data_map_cases[d_map[0]] = data
        data_map_normal = {}
        for d_map in map_normal.items():
            data = [np.load(x) for x in d_map[1]]
            data_map_normal[d_map[0]] = data
        # data_tmp = data_map_cases.items(0)
        channel_num = data_map_cases[list(data_map_cases.keys())[0]][0].shape[0]  # 获得信道的个数
        self.channel_number = channel_num
        self.data_map_normal = data_map_normal
        self.data_map_cases = data_map_cases


if __name__ == '__main__':
    seeg = seegdata()
    seeg.get_split_npy_data()
    print(seeg.channel_number)