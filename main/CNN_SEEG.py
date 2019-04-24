#!/usr/bin/python
from util import *
import argparse

parser = argparse.ArgumentParser(description="CNN parameter setting!")
parser.add_argument('-t', '--time', default=5)  # 每一帧的长度
parser.add_argument('-s', '--sample', default=100)  # 对其进行重采样
args = parser.parse_args()

file_map = get_all_file_path()
# print(file_map['SGH'])
path_LK_Seizure = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data processed/LK/LK_Seizure01_9min6Sec_seeg_raw.fif"
path_SGH_Seizure_1 = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data processed/SGH/SGH_Seizure01_19min9Sec_seeg_raw-1.fif"
path_SGH_Seizure = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data processed/SGH/SGH_Seizure01_19min9Sec_seeg_raw.fif"


def data_precess_input():  # 主要是数据的预处理，作为输入
    raw_LK = read_raw(path_LK_Seizure)
    raw_SGH_1 = read_raw(path_SGH_Seizure_1)
    raw_SGH = read_raw(path_SGH_Seizure)
    raw_SGH = data_connection(raw_SGH, raw_SGH_1)
    # 相关数据的展示
    plt.figure()
    raw_LK.plot()
    plt.show()
    LK_channels_names = get_channels_names(raw_LK)
    SGH_channels_names = get_channels_names(raw_SGH)
    print(get_channels_names(raw_LK))
    print(get_channels_names(raw_SGH_1))
    common_channels = get_common_channels(LK_channels_names, SGH_channels_names)  # 获得的是公共的信道
    print(common_channels)
    print(max(raw_LK.times) / 60)
    raw_SGH = data_connection(raw_SGH, raw_SGH_1)
    print(max(raw_SGH.times) / 60)
    SGH_data = select_channel_data(raw_SGH, common_channels)
    LK_data = select_channel_data(raw_LK, common_channels)
    print(SGH_data.shape)
    print(LK_data.shape)  # 数据的特征
    # test_data = select_channel_data_mne(raw_LK, ['POL E3'])
    # plt.figure()
    # test_data.plot()
    # plt.show()
