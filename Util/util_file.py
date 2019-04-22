#!/usr/bin/python
# ---------------------------------
# 文件工具方法， 主要包含常见的文件处理方法
#
# ---------------------------------
import glob
import os


def get_all_file_path(path="../data/data_path.txt"): # 主要是获取某文件夹下面所有的文件列表
    '''
    :param path: 存储对应文件的路径
    :return: 文件夹下面对应的文件路径
    '''

    f = open(path, 'r', encoding="UTF-8")
    paths = [line.rsplit()[0] for line in
             f.readlines()]  # ['E:\\数据集\\表型数据集\\SEEG_Data\\LK处理数据', 'E:\\数据集\\表型数据集\\SEEG_Data\\SGH处理数据']
    f.close()
    print("dir : {}".format(paths))
    LK_path = glob.glob(os.path.join(paths[0], '*.fif'))
    SGH_path = glob.glob(os.path.join(paths[1], '*.fif'))
    print("LK files: {}, count: {}".format(LK_path, len(LK_path)))
    print("SGH files: {}, count: {}".format(SGH_path, len(SGH_path)))
    file_map = {"LK": LK_path, "GHB": SGH_path}
    return file_map
