#!/usr/bin/python
import sys
sys.path.append('../')
from util import *


class seegdata:
    def __init__(self, path_dir="../data/data_slice/split"):
        self.path_dir = path_dir

    def set_path_dir(self, new_path):
        self.path_dir = new_path

    def get_split_npy_data(self, path_normal='../data/data_slice/split/preseizure',
                           path_cases='../data/data_slice/split/cases'):
        self.path_cases = path_cases
        self.path_normal = path_normal  # 癫痫发作的前段时间
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

    def get_all_path_by_keyword(self, keyword):  # ./split/keyword
        name_dir = os.listdir(self.path_dir)
        if keyword in name_dir:
            temp_path = os.path.join(self.path_dir, keyword)
            path_all = get_all_file_path(temp_path, 'npy')
            return path_all
        else:
            print("please check your keyword, keyword is not exist!")
            return None


# if __name__ == '__main__':
#     path_dir = "../data/data_slice/split"
#     seeg = seegdata()
#     p = seeg.get_all_path_by_keyword('sleep')
#     data_path = []
#     for dp in p.values():
#         for p in dp:
#             data_path.append(p)
#     print(data_path.__len__())

