#!/usr/bin/python

'''
将数据按照标签进行转化，按照每一帧进行存储
'''
from util import *

path_lk0 = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/LK_label0_raw.fif'
path_sgh0 = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/SGH_label0_raw.fif"

path_lk1 = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/LK_label1_raw.fif'
path_sgh1 = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/SGH_label1_raw.fif"
save_normal = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/split/normal'
save_cases = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/split/cases'
time = 5  # 每一帧的持续时间


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


if __name__ == '__main__':
    # save_file(path_lk0, path_sgh0, save_normal, flag=0)
    save_file(path_lk1, path_sgh1, save_cases, flag=1)
