#!/usr/bin/python
import numpy as np
from main.pre_Processing import *
import glob
import os


def test_1():
    b = []
    for index in range(5):
        a = np.random.randint(0, 100, 10)
        a = a[np.newaxis, :]
        b.append(a)
    # a = np.random.randint(0, 10, 10)
    c = np.array(b)
    c = c.reshape((-1, 10))
    print(c)


def test_2():
    path = "../data/output_data/sleep_frame_eeg.npy"
    data = np.load(path)
    print(data.shape)
    print(data)


def test_3():
    '''

    :return: 图像的展示
    '''
    path_data = "../data/output_data/sleep_frame_eeg.npy"
    sleep_time_frame = read_sleep_date(path_data)
    length = len(sleep_time_frame[0])
    ratio = duration_time / length
    print("length:{} ratio:{}".format(length, ratio))
    x = [ratio * x for x in range(length)]
    draw_plot(x, sleep_time_frame[0])


def test_4(path="../data/data_path.txt"):
    '''

    :param path: 文件的存储目录
    :return:
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


def test_6():
    print("test_n finished!")
    a = {"1": 12}
    print(a)
    a["2"] = 123
    print(a)
    return True


# 模板
def test_n():
    print("test_n finished!")
