#!/usr/bin/python
import numpy as np
from main.pre_Processing import *
import glob
import os
import librosa
import librosa.display
import uuid


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
    path_lk0 = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/split/cases/06f4fad8-65db-11e9-bae7-e0d55ee63f3d-1.npy'
    raw = np.load(path_lk0)
    plt.figure()
    librosa.display.waveplot(raw)
    plt.show()
    librosa.display.waveplot(raw[0])
    plt.show()
    # plt.plot(list(range(100 * 5)), raw[0])
    # for index in range(118):
    #     librosa.display.waveplot(raw[index])  # 波形数据的展示，将数据转化为类似于声波的一种
    #     plt.show()

    # # 模板 def test_n():
    # path_normal = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/split/cases"
    # f = list(os.walk(path_normal))
    # print(len(f[0][2]))


def test_8():   # 关于数据过零率的相关统计信息
    path = '../data/seizure/split'
    paths_filet_map = get_all_file_path(path, "npy")
    path_cases = paths_filet_map['cases']
    path_normal = paths_filet_map['normal']
    data_normal = [np.load(x) for x in path_normal]
    data_cases = [np.load(x) for x in path_cases]
    channel_num = len(data_cases[0])

    avg_counts_cases = []
    for dd in data_cases:
        counts = []
        for d in dd:
            c = librosa.zero_crossings(d)
            counts.append(len([x for x in c if x == True]))
        avg_count = sum(counts) / channel_num
        avg_counts_cases.append(avg_count)
    plt.figure()
    plt.title("zero_crossing cases")
    plt.plot(range(len(avg_counts_cases)), avg_counts_cases)
    print(np.mean(np.array(avg_counts_cases)))
    plt.show()

    avg_counts_normal = []
    for dd in data_normal:
        counts = []
        for d in dd:
            c = librosa.zero_crossings(d)
            counts.append(len([x for x in c if x == True]))
        avg_count = sum(counts) / channel_num
        avg_counts_normal.append(avg_count)
    plt.figure()
    plt.title("zero_crossing normal")
    plt.plot(range(len(avg_counts_normal)), avg_counts_normal)
    print(sum(avg_counts_normal)/len(avg_counts_normal))
    plt.show()
