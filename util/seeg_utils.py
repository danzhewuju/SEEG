#!/usr/bin/python3

import os
import uuid

import mne
import numpy as np
import pandas as pd
import pyedflib
import scipy.io as sio
from mne.time_frequency import *
import matplotlib.pyplot as plt


def read_raw(path):
    raw = mne.io.read_raw_fif(path, preload=True)
    return raw


def read_edf_raw(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    return raw


def get_channels_names(raw):
    channel_names = raw.info['ch_names']
    return channel_names


def get_recorder_time(data):
    '''
    :param data: raw data
    :return: 这个文件记录的时间长度
    '''
    time = data.times[-1]
    return time


def filter_hz(raw, high_pass, low_pass):  # 对数据进行滤波处理 对于（high_pass, low_pass）范围波形进行选择
    raw.filter(high_pass, low_pass)
    return raw


def save_numpy_info(data, path):  # 存储numpy的数据
    if os.path.exists(path):
        print("File is exist!!!")
        return False
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
    return True


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


def select_channel_data_mne(raw, select_channel_name):  # 根据信道的顺序，重新选择信道
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


def get_duration_raw_data(raw, start, stop):
    '''

    :param raw: 原始数据
    :param start: 开始的时间点
    :param stop: 终止的时间点
    :return:
    '''
    end = max(raw.times)
    if stop > end:
        print("over range!!!")
        return None
    else:
        duration_data = raw.crop(start, stop)
        return duration_data


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


def seeg_preprocess(fin, fout, seeg_chan_name):
    '''
    SEEG滤波
    :param fin:  源数据文件名
    :param fout:   输出文件名（***以_raw.fif结尾***）
    :param seeg_chan_name：   需要滤波的信道名列表
    :return:
    '''
    raw = mne.io.read_raw_edf(fin, preload=True)
    specific_chans = raw.pick_channels(seeg_chan_name)
    del raw
    if len(specific_chans.info['ch_names']) != len(seeg_chan_name):
        print("channels number not matched")
        return
    sfreq = specific_chans.info['sfreq']  # 采样频率
    nyq = sfreq / 2.  # 奈奎斯特频率
    specific_chans.notch_filter(np.arange(50, nyq, 50), filter_length='auto',
                                phase='zero')
    specific_chans.filter(0.5, None, fir_design='firwin')
    specific_chans.save(fout)
    del specific_chans


def eeg_preprocess(fin, fout, seeg_chan_name):
    '''
    EEG滤波
    :param fin:  源数据文件名
    :param fout:   输出文件名（***以_raw.fif结尾***）
    :param seeg_chan_name：   需要滤波的信道名列表
    :return:
    '''
    raw = mne.io.read_raw_edf(fin, preload=True)
    specific_chans = raw.copy().pick_channels(seeg_chan_name)
    del raw
    if len(specific_chans.info['ch_names']) != len(seeg_chan_name):
        print("channels number not matched")
        return
    sfreq = specific_chans.info['sfreq']  # 采样频率
    nyq = sfreq / 2.  # 奈奎斯特频率
    specific_chans.notch_filter(np.arange(50, nyq, 50), filter_length='auto',
                                phase='zero')
    specific_chans.filter(1., None, fir_design='firwin')
    specific_chans.save(fout)
    del specific_chans


def seeg_npy_plot(data, channels, save_path):
    '''

    :param data: numpy 格式的数据
    :param cahnnels: 所选择的信道list
    :return:
    '''
    k = len(channels)
    plt.figure(0)
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    for i in range(k):
        plt.subplot(k, 1, i + 1)
        plt.title("channel:{}".format(channels[i]))
        plt.plot(data[channels[i]])
    plt.savefig(save_path)
    plt.close(0)
    # plt.show()
    return True


def split_edf(filename, NEpochs=1):  # 把太大的edf文件分成NEpochs个小edf文件
    '''

    :param filename:  源文件名称
    :param NEpochs:   要划分的数量
    :return:
    '''
    dirname = os.path.dirname(filename)
    basename = os.path.basename(filename)
    oridir = os.getcwd()
    if dirname != "":  # pyedflib只能读取当前工作目录的文件
        os.chdir(dirname)
    f = pyedflib.EdfReader(basename)
    os.chdir(oridir)  # 路径换回去
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
            if ch == NChannels - 1:
                data_list.append(f.readSignal(ch)[i * NSamples:])
            else:
                data_list.append(f.readSignal(ch)[i * NSamples: (i + 1) * NSamples - 1])
        fout.setSignalHeaders(channels_info)
        fout.writeSamples(data_list)
        fout.close()
        del fout
        del data_list
        print("File %d done" % i)


def save_raw_as_edf(raw, fout_name):  # 把raw数据存为edf格式
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


def make_whole_as_epoch(raw, e_id=666):
    '''
    将一整个raw作为一个epoch返回
    :param raw:  raw类型对象
    :param e_id:  整数类型，指定event的id，不能与已有id重复
    :return:  Epochs对象
    '''
    data, _ = raw[:, :]
    event_id = {'Added': e_id}  # 人为增加一个event
    event = [[0, 0, e_id]]  # 在第一个样本处标记event为id
    epoch = mne.EpochsArray([data], raw.info, event, 0, event_id)
    return epoch


def tfr_analyze(epochs, freqs, resample=None, decim=1):
    '''
    freqs:type为ndarray，指定一个离散的频率数组
    :param epochs:  待分析的Epochs对象
    :param freqs:  ndarray类型，包含感兴趣的所有频率，例np.arange(80,100,0.5)
    :param resample:  整数类型，指明重采样频率，通过对数据重采样减轻内存压力
    :param decim:  整数类型，只抽取时频变换后的部分结果，减轻内存压力
    :return:  AverageTFR对象，包含时频变换后的数据和信息
    '''
    if resample is not None:
        epochs.resample(resample, npad='auto')  # 重采样，减少内存消耗
    n_cycles = freqs / 2.
    # 使用小波变换进行时频变换
    # decim参数指定对转换过的结果后再次重采样的频率，例如若指定为5，则频率变为原来的5分之一
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=True, decim=decim)
    power.info['sfreq'] /= decim
    return power


def tfr_extract(power, tmin=0, tmax=None):
    '''
    提取tfr_analyze返回的数据中感兴趣的时间段
    :param power:  AverageTFR对象，时频变换的输出
    :param tmin:  时间起点(包含在区间内)
    :param tmax:  时间终点(不包含在区间内)
    :return:  ndarray, shape(n_channels, n_freqs, n_times)
    '''
    sfreq = power.info['sfreq']
    start = int(tmin * sfreq)
    if tmax is None:
        return np.array([[[k for k in power.data[i][j][start:]] for j in range(len(power.data[i]))] for i in
                         range(len(power.data))])
    else:
        end = int(tmax * sfreq)
        return np.array([[[k for k in power.data[i][j][start: end]] for j in range(len(power.data[i]))] for i in
                         range(len(power.data))])


def get_cost_matrix(elec_pos):
    '''
    获取代价矩阵（不同电极之间的距离）
    :param elec_pos:  含有信道名以及坐标的字典
    :return:  cost_matrix：   代价矩阵
    '''
    n = len(elec_pos)
    cost_matrix = [[0 for _ in range(n)] for _ in range(n)]
    i = 0
    while i < n:
        j = i + 1
        while j < n:
            cost_matrix[i][j] = np.linalg.norm(elec_pos[i]['pos'] - elec_pos[j]['pos'])
            cost_matrix[j][i] = cost_matrix[i][j]
            j += 1
        i += 1
    return cost_matrix


def least_traversal(elec_pos):
    '''
    枚举所有起点计算出最小代价的遍历路径
    :param elec_pos:  含有信道名以及坐标的字典
    :return:  min_cost：   最小代价
    :return:  min_path：   对应路径
    '''
    cost_matrix = get_cost_matrix(elec_pos)
    n = len(elec_pos)
    maximum = 9999999
    min_cost = maximum
    min_path = None
    for start in range(n):
        visited = [False for _ in range(n)]
        n_visited = 0
        cur = start
        cost = 0
        path = [elec_pos[cur]['name']]
        while n_visited < n - 1:
            visited[cur] = True
            n_visited += 1
            min_d = maximum
            min_i = 0
            for i in range(n):
                d = cost_matrix[cur][i]
                if d < min_d and not visited[i]:
                    min_d = d
                    min_i = i
            cost += min_d
            path.append(elec_pos[min_i]['name'])
            cur = min_i
        if cost < min_cost:
            min_cost = cost
            min_path = path
    return min_cost, min_path


def retrieve_chs_from_mat(patient_name):
    '''
    提取.mat文件中的信道名和坐标信息
    :param patient_name:   目标病人名（须保证文件名为patient_name.mat）
    :return:  elec_pos：   含有信道名以及坐标的字典
    '''
    pos_info = sio.loadmat(patient_name + ".mat")
    elec_pos = list()
    for i in range(pos_info['elec_Info_Final'][0][0][1][0].size):  # name为字符串,pos为ndarray格式
        elec_pos.append({'name': pos_info['elec_Info_Final'][0][0][0][0][i][0],
                         'pos': pos_info['elec_Info_Final'][0][0][1][0][i][0]})
    return elec_pos


def get_path(patient_name):
    '''
    获取当前病人的信道排列并保存在.csv文件中
    :param patient_name:   目标病人名
    '''
    _, path = least_traversal(retrieve_chs_from_mat(patient_name))
    print(path)
    path_len = len(path)
    print(path_len)
    to_csv = [[i for i in range(path_len)], path]
    to_csv = [[row[i] for row in to_csv] for i in range(path_len)]
    col = ['ID', 'chan_name']
    csv_frame = pd.DataFrame(columns=col, data=to_csv)
    csv_frame.to_csv('./' + patient_name + '_seq.csv', encoding='utf-8')
