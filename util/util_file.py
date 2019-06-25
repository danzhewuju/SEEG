#!/usr/bin/python
# ---------------------------------
# 文件工具方法， 主要包含常见的文件处理方法
#
# ---------------------------------
import glob
import os

import numpy as np
import torchvision.transforms as transforms
from PIL import Image


# parser = argparse.ArgumentParser(description="File path function")
# parser.add_argument("-p", "--path", type=str, default="/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data processed")
# args = parser.parse_args()


def get_all_file_path(path="/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data processed",
                      suffix='fif'):  # 主要是获取某文件夹下面所有的文件列表
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


def matrix_normalization(data, resize_shape=(131, 200)):
    '''
    矩阵的归一化，主要是讲不通形状的矩阵变换为特定形状的矩阵
    eg:(188, 200)->(131, 200)   归一化的表示
    :param data:
    :param resize_shape:
    :return:
    '''
    data_t = Image.fromarray(data.astype(np.uint8))
    transforms_data = transforms.Compose(
        [transforms.Resize(resize_shape)]
    )
    result = transforms_data(data_t)
    result = np.array(result)
    return result
