#!/usr/bin/python3

import mne
import numpy as np
import os
import uuid
import pyedflib
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
    if os.path.exists(path):
        print("File is exist!!!")
        return None
    else:
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


def pre_process(raw, seeg=True, eeg=False): # seeg及eeg预处理
    '''

    :param raw:  原数据
    :param seeg:   是否为seeg数据
    :param seeg:   是否为eeg数据
    :return:
    '''
    sfreq = raw.info['sfreq']
    nyq = sfreq / 2.
    if seeg:
        high_pass = 0.5
    elif eeg:
        high_pass = 1.
    raw.notch_filter(np.arange(50, nyq, 50), filter_length='auto', phase='zero') # 滤去线路噪声
    raw.filter(high_pass, None, fir_design='firwin') # 滤去slow drifts
    return raw


def split_edf(filename, NEpochs=1): # 把太大的edf文件分成NEpochs个小edf文件
    '''

    :param filename:  源文件名称
    :param NEpochs:   要划分的数量
    :return:
    '''
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    oridir = os.getcwd()
    if dirname != "": # pyedflib只能读取当前工作目录的文件
        os.chdir(dirname)
    f = pyedflib.EdfReader(basename)
    os.chdir(oridir) # 路径换回去
    NSamples = int(f.getNSamples()[0] / NEpochs)
    NChannels = f.signals_in_file
    fileOutPrefix = basename + '_'

    channels_info = list()
    for ch in range(NChannels):
        ch_dict = dict()
        ch_dict['label'] = f.getLabel(ch)
        ch_dict['dimension'] = f.getPhysicalDimension(ch)
        ch_dict['sample_rate'] = f.getSampleFrequency(ch)
        ch_dict['physical_max'] = f.getPhysicalMaximum(ch)
        ch_dict['physical_min'] = f.getPhysicalMinimum(ch)
        ch_dict['digital_max'] = f.getDigitalMaximum(ch)
        ch_dict['digital_min'] = f.getDigitalMinimum(ch)
        ch_dict['transducer'] = f.getTransducer(ch)
        ch_dict['prefilter'] = f.getPrefilter(ch)
        channels_info.append(ch_dict)

    for i in range(NEpochs):
        print("File %d starts" % i)
        fileOut = os.path.join('.', fileOutPrefix + str(i) + '.edf')
        fout = pyedflib.EdfWriter(fileOut, NChannels, file_type=pyedflib.FILETYPE_EDFPLUS)
        data_list = list()
        for ch in range(NChannels):
            data_list.append(f.readSignal(ch)[i * NSamples: (i + 1) * NSamples - 1])
        fout.setSignalHeaders(channels_info)
        fout.writeSamples(data_list)
        fout.close()
        del fout
        del data_list
        print("File %d done" % i)


def save_raw_as_edf(raw, fout_name): # 把raw数据存为edf格式
    '''

    :param raw:  raw格式数据
    :param fout_name:   输出的文件名
    :return:
    '''
    NChannels = raw.info['nchan']
    channels_info = list()
    for i in range(NChannels):
        '''默认参数来自edfwriter.py'''
        ch_dict = dict()
        ch_dict['label'] = raw.info['chs'][i]['ch_name']
        ch_dict['dimension'] = 'mV'
        ch_dict['sample_rate'] = raw.info['sfreq']
        ch_dict['physical_max'] = 1.0
        ch_dict['physical_min'] = -1.0
        ch_dict['digital_max'] = 32767
        ch_dict['digital_min'] = -32767
        ch_dict['transducer'] = 'trans1'
        ch_dict['prefilter'] = "pre1"
        channels_info.append(ch_dict)

    fileOut = os.path.join('.', fout_name + '.edf')
    fout = pyedflib.EdfWriter(fileOut, NChannels, file_type=pyedflib.FILETYPE_EDFPLUS)
    data_list, _ = raw[:, :]
    print(data_list)
    fout.setSignalHeaders(channels_info)
    fout.writeSamples(data_list)
    fout.close()
    print("Done!")
    del fout
    del data_list