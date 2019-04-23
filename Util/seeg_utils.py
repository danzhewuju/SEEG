#!/usr/bin/python3

import mne
import numpy as np
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


def data_connection(raw1, raw2):
    '''

    :param raw1: raw data1
    :param raw2: raw data2
    :return:  data connection raw1:raw2
    '''
    raw1.append(raw2)
    return raw1


def select_channel_data(raw, ):  # 根据某些信道的名称进行数据选择
    '''

    :param raw:  raw data
    :return: channel data
    '''
