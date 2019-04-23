#!/usr/bin/python
from Util import *

path_lk0 = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/LK_label0_raw.fif'
path_sgh0 = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/SGH_label0_raw.fif"

path_lk1 = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/LK_label1_raw.fif'
path_sgh1 = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/SGH_label1_raw.fif"
save_normal = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/split/normal'
save_cases = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/split/cases'
time = 5  # 每一帧的持续时间


def save_file():  # 将数据进行存储转化为切片数据
    raw_lk0 = read_raw(path_lk0)  # 数据的读取
    raw_lk0.resample(100, npad="auto")  # resample 100hz
    raw_sgh0 = read_raw(path_sgh0)
    raw_sgh0.resample(100, npad="auto")  # resample 100hz
    raw_sgh0_ch_names = get_channels_names(raw_sgh0)
    raw_lk0_ch_names = get_channels_names(raw_lk0)
    common_channels = get_common_channels(raw_lk0_ch_names, raw_sgh0_ch_names)  # 公用信道的寻找

    raw_lk0 = select_channel_data_mne(raw_lk0, common_channels)  # 选择出公共信道的数据
    raw_sgh0 = select_channel_data_mne(raw_sgh0, common_channels)

    raw_lk0_split = data_split(raw_lk0, time)
    raw_sgh0_split = data_split(raw_sgh0, time)
    print(len(raw_lk0_split), len(raw_sgh0_split))
    save_split_data(raw_lk0, save_normal, 0)
    save_split_data(raw_sgh0, save_normal, 0)


if __name__ == '__main__':
    save_file()
