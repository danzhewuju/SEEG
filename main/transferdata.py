#!/usr/bin/python

'''
将数据按照标签进行转化，按照每一帧进行存储 默认每一帧的时间长度是2秒
'''
from util import *
import pandas as pd

time = 2  # 每一帧的持续时间
resample = 100  # 重采样的频率
high_pass = 0
low_pass = 30


def save_file(path_lk, path_sgh, save_dir, flag):  # 将数据进行存储转化为切片数据
    raw_lk = read_raw(path_lk)  # 数据的读取
    raw_lk.resample(100, npad="auto")  # resample 100hz
    raw_sgh = read_raw(path_sgh)
    raw_sgh.resample(100, npad="auto")  # resample 100hz
    raw_sgh_ch_names = get_channels_names(raw_sgh)
    raw_lk_ch_names = get_channels_names(raw_lk)
    common_channels = get_common_channels(raw_lk_ch_names, raw_sgh_ch_names)  # 公用信道的寻找

    raw_lk = select_channel_data_mne(raw_lk, common_channels)  # 选择出公共信道的数据
    raw_sgh = select_channel_data_mne(raw_sgh, common_channels)

    raw_lk_split = data_split(raw_lk, time)
    raw_sgh_split = data_split(raw_sgh, time)
    print(len(raw_lk_split), len(raw_sgh_split))
    save_split_data(raw_lk_split, save_dir, flag)
    save_split_data(raw_sgh_split, save_dir, flag)


def save_split_data_test(raw_data, name, flag, time=time):
    '''

    :param raw_data:  原始的raw数据，保证mne可读格式
    :param name: 个人的名称，用于创建文件夹
    :param flag: flag 标志
    :param time: 切片的大小
    :return:
    '''
    path_dir = "../data/seizure/split"
    if flag == 0:
        dir = 'normal'  # 癫痫发作前的状态
    else:
        if flag == 1:
            dir = "cases"  # 癫痫正在发作
        else:
            dir = "sleep"  # 正常睡眠状态
    path_dir = os.path.join(path_dir, dir)
    if os.path.exists(path_dir) is not True:
        os.makedirs(path_dir)
    path_person = os.path.join(path_dir, name)
    if os.path.exists(path_person) is not True:
        os.makedirs(path_person)

    raw_split_data = data_split(raw_data, time)  # 进行五秒的切片
    print("split time {}".format(len(raw_split_data)))
    save_split_data(raw_split_data, path_person, flag)
    return True


def data_save(path_read, name, flag, common_channels):
    raw = read_raw(path_read)
    raw = filter_hz(raw, high_pass, low_pass)
    raw.resample(resample, npad="auto")  # resample 100hz
    raw = select_channel_data_mne(raw, common_channels)
    raw.reorder_channels(common_channels)  # 更改信道的顺序
    save_split_data_test(raw, name, flag, time)


def generate_data(path, flag, name, path_commom_channel):
    '''

    :param path: 文件的路径
    :param flag: 标志 0 ： 表示癫痫发作前的状态， 1：表示癫痫正在发作 2:表示正常睡觉时的状态
    :param name:病人的名称
    :path_common_channel : 公共的信道名称
    :return:
    '''
    data = pd.read_csv(path_commom_channel, sep=',')
    d_list = data['channels']
    common_channels = list(d_list)
    data_save(path, name, flag, common_channels)


if __name__ == '__main__':
    path_commom_channel = "../data/seizure/common_channels.csv"

    # path_raw = "../data/raw_data/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-0.fif"
    # path_raw = '../data/raw_data/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-1.fif'
    path_raw = '../data/raw_data/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-2-0.fif'
    name = "LK"
    flag = 2  # 正常睡眠时间

    # path_raw = '../data/seizure/LK_label0_raw.fif'
    # name = "LK"
    # flag = 0  # 癫痫发作前的睡眠时间

    generate_data(path_raw, flag, name, path_commom_channel)

    # path_dir = "../data/raw_data/LK_Pre_seizure"
    # flag = 0
    # for p in os.listdir(path_dir):
    #     path_raw = os.path.join(path_dir, p)
    #     name = "LK"
    #     generate_data(path_raw, flag, name, path_commom_channel)
