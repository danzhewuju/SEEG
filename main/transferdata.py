#!/usr/bin/python

'''
将数据按照标签进行转化，按照每一帧进行存储 默认每一帧的时间长度是5秒
'''
from util import *

path_lk0 = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/LK_label0_raw.fif'
path_sgh0 = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/SGH_label0_raw.fif"

path_lk1 = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/LK_label1_raw.fif'
path_sgh1 = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/SGH_label1_raw.fif"
save_normal = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/split/normal'
save_cases = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/seizure/split/cases'
time = 5  # 每一帧的持续时间
resample = 100  # 重采样的频率


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
        dir = 'normal'
    else:
        dir = "cases"
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


if __name__ == '__main__':
    # --------------------------------------------------
    # 主要生成全部切片，所有正常人和病人的切片数据放在一个文件夹
    # save_file(path_lk0, path_sgh0, save_normal, flag=0)
    # save_file(path_lk1, path_sgh1, save_cases, flag=1)
    # --------------------------------------------------

    # ---------------------------------------------------
    # 操作流程
    # 1. 读取数据
    # 2. 重采样
    # 3. 信道选择
    # 3. 数据切片
    # 4. 数据保存

    raw_lk0 = read_raw(path_lk0)  # 数据的读取
    raw_lk0.resample(resample, npad="auto")  # resample 100hz
    raw_sgh0 = read_raw(path_sgh0)
    raw_sgh0.resample(resample, npad="auto")  # resample 100hz

    raw_lk1 = read_raw(path_lk1)  # 数据的读取
    raw_lk1.resample(resample, npad="auto")  # resample 100hz
    raw_sgh1 = read_raw(path_sgh1)
    raw_sgh1.resample(resample, npad="auto")  # resample 100hz

    raw_sgh_ch_names = get_channels_names(raw_sgh0)
    raw_lk_ch_names = get_channels_names(raw_lk0)
    common_channels = get_common_channels(raw_lk_ch_names, raw_sgh_ch_names)  # 公用信道的寻找

    raw_lk0 = select_channel_data_mne(raw_lk0, common_channels)
    raw_lk1 = select_channel_data_mne(raw_lk1, common_channels)
    raw_sgh0 = select_channel_data_mne(raw_sgh0, common_channels)
    raw_sgh1 = select_channel_data_mne(raw_sgh1, common_channels)

    save_split_data_test(raw_lk0, "LK", flag=0)
    save_split_data_test(raw_lk1, "LK", flag=1)
    save_split_data_test(raw_sgh0, 'SGH', flag=0)
    save_split_data_test(raw_sgh1, 'SGH', flag=1)
