#!/usr/bin/python
# ---------------------------------
# 文件工具方法， 主要包含常见的文件处理方法
#
# ---------------------------------
import glob
import os
import argparse


# parser = argparse.ArgumentParser(description="File path function")
# parser.add_argument("-p", "--path", type=str, default="/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data processed")
# args = parser.parse_args()


def get_all_file_path(path="/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data processed"):  # 主要是获取某文件夹下面所有的文件列表
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
        file_p = glob.glob(os.path.join(p, '*.fif'))
        file_map[names[index]] = file_p
    return file_map
