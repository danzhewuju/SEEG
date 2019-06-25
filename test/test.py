#!/usr/bin/python

import random

import librosa
import librosa.display
import pandas as pd

from main.Seegdata import *


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


def test_8():  # 关于数据过零率的相关统计信息
    path_cases = '../data/seizure/split/cases'
    path_normal = '../data/seizure/split/normal'
    map_cases = get_all_file_path(path_cases, 'npy')
    map_normal = get_all_file_path(path_normal, 'npy')
    # print(map_cases)
    # print(map_normal)
    data_map_cases = {}
    for d_map in map_cases.items():
        data = [np.load(x) for x in d_map[1]]
        data_map_cases[d_map[0]] = data
    data_lk = data_map_cases["LK"]
    data_sgh = data_map_cases["SGH"]
    channel_num = len(data_lk[0])
    avg_counts_lk = []
    for dd in data_lk:
        counts = []
        for d in dd:
            c = librosa.zero_crossings(d)
            counts.append(len([x for x in c if x == True]))
        avg_count = sum(counts) / channel_num
        avg_counts_lk.append(avg_count)
    # plt.visualization_feature()
    # plt.title("zero_crossing LK")
    # plt.plot(range(len(avg_counts_cases)), avg_counts_cases)
    print(np.mean(np.array(avg_counts_lk)))
    # plt.show()

    avg_counts_sgh = []
    for dd in data_sgh:
        counts = []
        for d in dd:
            c = librosa.zero_crossings(d)
            counts.append(len([x for x in c if x == True]))
        avg_count = sum(counts) / channel_num
        avg_counts_sgh.append(avg_count)
    plt.figure()
    plt.title("zero_crossing")
    plt.xlabel("t")
    plt.ylabel('count')
    p_lk = plt.plot(range(len(avg_counts_lk)), avg_counts_lk, label='LK')
    end = len(avg_counts_lk)

    p_sgh = plt.plot(avg_counts_sgh[:end], label='sgh')
    plt.legend()
    print(sum(avg_counts_sgh) / len(avg_counts_sgh))
    plt.show()


def test_9():  # 其他功能的探索
    a = {"1": 12, "2": 23, "3": 321}
    b = list(a.keys())
    print("Test")
    print(b)


def test_10():
    path = "../data/seeg/train/pre_zeizure/e01b382d-697b-11e9-bae7-e0d55ee63f3d-0.npy"
    data = np.load(path)
    print(data)
    print(data.dtype)


def test_11():
    path_dir = '../data/raw_data/Pre_seizure'
    print(os.listdir(path_dir))


def test_12():
    path_channel = '../data/seizure/common_channels.csv'
    path = '../data/raw_data/LK_Pre_seizure/LK_SZ1_pre_seizure_raw.fif'
    raw_data = read_raw(path)
    raw_data.plot()
    print(raw_data.info['ch_names'])
    data = pd.read_csv(path_channel, sep=',')
    d_list = data['channels']
    common_channels = list(d_list)
    new_raw_data = select_channel_data_mne(raw_data, common_channels)
    new_raw_data.reorder_channels(common_channels)
    new_raw_data.plot()
    print(new_raw_data.info['ch_names'])

    # print(d_list)
    # print(data['channels'])


def test_13():
    a = np.random.randint(0, 100, 20)
    b = a.copy()
    c = a.copy()
    random.shuffle(b)
    random.shuffle(c)
    print(a, b, c)


def test_14():
    seeg = seegdata()
    path_dir_seizure = "../data/seizure/split/preseizure"
    seeg.set_path_dir(path_dir_seizure)
    sleep_bwt = seeg.get_all_path_by_keyword('within_warning_time')
    sleep_bwt_label1 = sleep_bwt['LK']  # 发病前的一段时间,警戒线之外
    for p in sleep_bwt_label1:
        d = np.load(p)
        print(d.shape)


def test_15():
    path = './path.csv'
    paths = pd.read_csv(path, sep=',')
    print(paths)
    raw_data_path = paths['path']
    raw_name = paths['name']
    f = open('./channel.txt', 'w', encoding='UTF-8')
    f.writelines('name, channels\n')
    for n, p in zip(raw_name, raw_data_path):
        data = read_edf_raw(p)
        channels = get_channels_names(data)
        # print(channels)
        print("name:{}, channels: {}".format(n, len(channels)))
        dd = n + ',' + str(channels)+'\n'
        f.writelines(dd)
    f.close()
