#!/usr/bin/python

'''
将数据按照标签进行转化，按照每一帧进行存储 默认每一帧的时间长度是2秒
'''
import pandas as pd

from util import *

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


def save_split_data_test(raw_data, name, flag, time=time, flag_duration=0):
    '''

    :param raw_data:  原始的raw数据，保证mne可读格式
    :param name: 个人的名称，用于创建文件夹
    :param flag: flag 标志
    :param time: 切片的大小
    :return:
    '''
    path_dir = "../data/seizure/split"
    if flag == 0:  # 癫痫发作前的状态, 这个状态可以进一步的细分，主要表现为发作前设置的预警时间
        if flag_duration == 0:  # 默认是不进行细分
            dir = 'preseizure'
        else:
            if flag_duration == 1:
                dir = 'preseizure/within_warning_time'  # 在预警的时间线之内
            else:
                dir = 'preseizure/before_warning_time'  # 在预警的时间线之前
    else:
        if flag == 1:
            dir = "cases"  # 癫痫正在发作
        else:
            if flag == 2:
                dir = "sleep"  # 正常睡眠状态
            else:
                dir = "awake"  # 代表的是清醒的状态

    path_dir = os.path.join(path_dir, dir)
    if os.path.exists(path_dir) is not True:
        os.makedirs(path_dir)
        print("create dir:{}".format(path_dir))
    path_person = os.path.join(path_dir, name)
    if os.path.exists(path_person) is not True:
        os.makedirs(path_person)
        print("create dir:{}".format(path_person))

    raw_split_data = data_split(raw_data, time)  # 进行2秒的切片
    print("split time {}".format(len(raw_split_data)))
    save_split_data(raw_split_data, path_person, flag)
    return True


def data_save(path_read, name, flag, common_channels, flag_duration=0):
    raw = read_raw(path_read)
    raw = filter_hz(raw, high_pass, low_pass)
    raw.resample(resample, npad="auto")  # resample 100hz
    raw = select_channel_data_mne(raw, common_channels)
    raw.reorder_channels(common_channels)  # 更改信道的顺序
    save_split_data_test(raw, name, flag, time, flag_duration=flag_duration)


def generate_data(path, flag, name, path_commom_channel, flag_duration=0):
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
    data_save(path, name, flag, common_channels, flag_duration=flag_duration)


def sleep_normal_handle():
    '''

    :return:
    流程操作， 将正常的睡眠进行切片划分
    '''
    # path_commom_channel = "../data/seizure/channels_info/LK_common_channels.csv"
    # path_raw_normal_sleep = ["../data/raw_data/LK/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-0.fif",
    #                          '../data/raw_data/LK/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-1.fif',
    #                          '../data/raw_data/LK/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-2-0.fif',
    #                          '../data/raw_data/LK/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-4-0.fif',
    #                          '../data/raw_data/LK/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-6-0.fif'
    #
    #                          ]  # 数据太多，因此只是选取部分的数据进行处理
    # name = "LK"
    # flag = 2  # 正常睡眠时间
    #
    # for path_raw in path_raw_normal_sleep:
    #     generate_data(path_raw, flag, name, path_commom_channel)
    #
    # print("{}正常睡眠的数据处理完成！".format(name))

    path_commom_channel = "../data/seizure/channels_info/BDP_seq.csv"
    path_raw_normal_sleep = ['../data/raw_data/BDP/BDP_SLEEP/BDP_Sleep_raw.fif']
    name = "BDP"
    flag = 2
    for path_raw in path_raw_normal_sleep:
        generate_data(path_raw, flag, name, path_commom_channel)
    print("{}正常睡眠的数据处理完成！".format(name))

    path_commom_channel = "../data/seizure/channels_info/SYF_seq.csv"
    path_raw_normal_sleep = ['../data/raw_data/SYF/SYF_SLEEP/SYF_Sleep_raw.fif']
    name = "SYF"
    flag = 2
    for path_raw in path_raw_normal_sleep:
        generate_data(path_raw, flag, name, path_commom_channel)
    print("{}正常睡眠的数据处理完成！".format(name))

    path_commom_channel = "../data/seizure/channels_info/WSH_seq.csv"
    path_raw_normal_sleep = ['../data/raw_data/WSH/WSH_SLEEP/WSH_Sleep_raw.fif']
    name = "WSH"
    flag = 2
    for path_raw in path_raw_normal_sleep:
        generate_data(path_raw, flag, name, path_commom_channel)
    print("{}正常睡眠的数据处理完成！".format(name))
    return True


def pre_seizure_biclass_handle():
    '''

    :return:
    处理流程过的函数，主要是处理癫痫发作前的是睡眠状态

    '''
    # path_commom_channel = "../data/seizure/channels_info/LK_common_channels.csv"
    #
    # path_dir = "../data/raw_data/LK/LK_Pre_seizure"
    # flag = 0
    # for p in os.listdir(path_dir):
    #     path_raw = os.path.join(path_dir, p)
    #     name = "LK"
    #     generate_data(path_raw, flag, name, path_commom_channel)
    # print("癫痫发作前的睡眠处理完成！！！")

    path_commom_channel = "../data/seizure/channels_info/ZK_seq.csv"
    path_dir = "../data/raw_data/ZK/ZK_Pre_seizure"
    flag = 0
    for p in os.listdir(path_dir):
        path_raw = os.path.join(path_dir, p)
        name = "ZK"
        generate_data(path_raw, flag, name, path_commom_channel)

    path_commom_channel = "../data/seizure/channels_info/WSH_seq.csv"
    path_dir = "../data/raw_data/WSH/WSH_Pre_seizure"
    flag = 0
    for p in os.listdir(path_dir):
        path_raw = os.path.join(path_dir, p)
        name = "WSH"
        generate_data(path_raw, flag, name, path_commom_channel)

    path_commom_channel = "../data/seizure/channels_info/SYF_seq.csv"
    path_dir = "../data/raw_data/SYF/SYF_Pre_seizure"
    flag = 0
    for p in os.listdir(path_dir):
        path_raw = os.path.join(path_dir, p)
        name = "SYF"
        generate_data(path_raw, flag, name, path_commom_channel)

    path_commom_channel = "../data/seizure/channels_info/BDP_seq.csv"
    path_dir = "../data/raw_data/BDP/BDP_Pre_seizure"
    flag = 0  # 指明了存储位置
    for p in os.listdir(path_dir):
        path_raw = os.path.join(path_dir, p)
        name = "BDP"
        generate_data(path_raw, flag, name, path_commom_channel)

    path_commom_channel = "../data/seizure/channels_info/BDP_seq.csv"
    path_dir = "../data/raw_data/BDP/BDP_SLEEP"
    flag = 2  # 指明了存储位置
    for p in os.listdir(path_dir):
        path_raw = os.path.join(path_dir, p)
        name = "BDP"
        generate_data(path_raw, flag, name, path_commom_channel)

    # path_commom_channel = "../data/seizure/channels_info/WSH_seq.csv"
    # path_dir = "../data/raw_data/WSH/WSH_Pre_seizure"
    # flag = 0
    # for p in os.listdir(path_dir):
    #     path_raw = os.path.join(path_dir, p)
    #     name = "WSH"
    #     generate_data(path_raw, flag, name, path_commom_channel)
    #
    # path_commom_channel = "../data/seizure/channels_info/SJ_seq.csv"
    # path_dir = "../data/raw_data/SJ/SJ_Pre_seizure"
    # flag = 0
    # for p in os.listdir(path_dir):
    #     path_raw = os.path.join(path_dir, p)
    #     name = "SJ"
    #     generate_data(path_raw, flag, name, path_commom_channel)
    #
    # path_commom_channel = "../data/seizure/channels_info/JWJ_seq.csv"
    # path_dir = "../data/raw_data/JWJ/JWJ_Pre_seizure"
    # flag = 0
    # for p in os.listdir(path_dir):
    #     path_raw = os.path.join(path_dir, p)
    #     name = "JWJ"
    #     generate_data(path_raw, flag, name, path_commom_channel)

    return True


def awake_handle(path_commom_channel):
    path_dir = "../data/raw_data/LK/LK_Awake"
    flag = 3
    for p in os.listdir(path_dir):
        path_raw = os.path.join(path_dir, p)
        name = "LK"
        generate_data(path_raw, flag, name, path_commom_channel)


def pre_seizure_multiclass_handle(path_commom_channel):
    '''
     增加了一个模块， 这个模块进一步的细分了癫痫发作之前的睡眠阶段，在睡眠阶段设置了预警时间
    :return:
    '''

    path_dir = "../data/raw_data/LK/multiPre_seizure/before_warning_time"
    if os.path.exists(path_dir) is True:
        rm_dir = os.path.join('../data/seizure/split/preseizure/before_warning_time', '*')
        os.system('rm -r {}'.format(rm_dir))
        print("all files are removed!")
    flag = 0
    flag_duration = 2
    for p in os.listdir(path_dir):
        path_raw = os.path.join(path_dir, p)
        name = "LK"
        generate_data(path_raw, flag, name, path_commom_channel, flag_duration)

    path_dir = "../data/raw_data/LK/multiPre_seizure/within_warning_time"
    if os.path.exists(path_dir) is True:
        rm_dir = os.path.join('../data/seizure/split/preseizure/within_warning_time', '*')
        os.system('rm -r {}'.format(rm_dir))
        print("all files are removed!")
    flag = 0
    flag_duration = 1
    for p in os.listdir(path_dir):
        path_raw = os.path.join(path_dir, p)
        name = "LK"
        generate_data(path_raw, flag, name, path_commom_channel, flag_duration)


if __name__ == '__main__':
    # 正常的睡眠时间
    sleep_normal_handle()
    # pre_seizure_biclass_handle()
    # sleep_normal_handle()
