#!/usr/bin/python3

import mne
import numpy as np
import os
import uuid
import matplotlib.pyplot as plt


def read_raw(path):
    raw = mne.io.read_raw_fif(path, preload=True)
    return raw


def get_channels_names(raw):
    channel_names = raw.info['ch_names']
    return channel_names


def filter_hz(raw, high_pass, low_pass):  # 对数据进行滤波处理 对于（high_pass, low_pass）范围波形进行选择
    raw.filter(high_pass, low_pass)
    return raw


def save_numpy_info(data, path):  # 存储numpy的数据
    np.save(path, data)
    print("Successfully save!")
    return True


def rewrite(raw, include_names, save_path):  # 对数据进行重写,主要是包含某些特殊的信道分离重写
    '''

    :param raw: 读取的原始数据
    :param include_names: 包含信道的名称
    :param save_path: 保存的路径
    :return: 返回只包含对应信道的数据
    '''
    want_meg = True
    want_eeg = False
    want_stim = False
    picks = mne.pick_types(raw.info, meg=want_meg, eeg=want_eeg, stim=want_stim,
                           include=include_names, exclude='bads')
    print("include channel names:{}".format(include_names))

    raw.save(save_path, picks=picks, overwrite=True)
    # raw.save("SEEG.fif", picks=picks_seeg, overwrite=True)
    print("successfully written!")


def get_common_channels(ch_names1, ch_names2):  # 寻找两个数据的公共信道
    '''

    :param ch_names1: raw1 ch_names list
    :param ch_names2: raw2 ch_names list
    :return: common ch_names list
    '''
    common_channels = [x for x in ch_names1 if x in ch_names2]
    return common_channels


def data_connection(raw1, raw2):  # 数据的拼接
    '''

    :param raw1: raw data1
    :param raw2: raw data2
    :return:  data connection raw1:raw2
    '''
    raw1.append(raw2)
    return raw1


def select_channel_data(raw, select_channel_names):  # 根据某些信道的名称进行数据选择,直接选择这个信道的数据
    '''

    :param raw:  raw data
    :return: channel data
    '''
    ch_names = get_channels_names(raw)
    pick_channel_No = mne.pick_channels(ch_names=ch_names, include=select_channel_names)
    data, time = raw[pick_channel_No, :]
    return data


def select_channel_data_mne(raw, select_channel_name):
    chan_name = select_channel_name
    specific_chans = raw.copy().pick_channels(chan_name)
    # specific_chans.plot(block=True)
    return specific_chans


def data_split(raw, time_step):  # 数据的切片处理
    data_split = []
    end = max(raw.times)
    epoch = int(end // time_step)
    fz = int(len(raw) / end)  # 采样频率
    for index in range(epoch - 1):
        start = index * fz * time_step
        stop = (index + 1) * fz * time_step
        data, time = raw[:, start:stop]
        data_split.append(data)
    return data_split


def save_split_data(data_split, path, flag):  # 切片数据的保存
    '''

    :param data_split:  被切片的数据
    :param path:   所存储的文件夹,也就是存储文件的上一级文件夹
    :param flag:   对应数据的标识
    :return:
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    for d in data_split:
        name = str(uuid.uuid1()) + "-" + str(flag)
        path_all = os.path.join(path, name)
        save_numpy_info(d, path_all)
    print("File save successfully {}".format(path))
    return True
