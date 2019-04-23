#!/usr/bin/python
from Util import *

path = '../data/LK_eye_close_eeg_raw_new.fif'
duration_time = 30  # 每一帧持续的时间是30s，这是一个参数，由你的神经网络来决定
select_channel_name = ['EEG C3-Ref-1']  # 选择信道的名称

'''
1.对数据进行重采样到100hz
2.选择select_channel_name 信道的数据
3.按照duration的信息进行分段
'''


def get_sleep_frame(path):           # 获得某一个信道的切片并对其进行处理
    raw = read_raw(path)
    raw.resample(100, npad="auto")  # resample 100hz
    raw.plot(duration=30)
    plt.show()
    ch_names = get_channels_names(raw)
    print(ch_names)
    picks_eeg = mne.pick_channels(ch_names=ch_names, include=select_channel_name)
    s = 0  # 开始时间
    end = max(raw.times)  # 持续的记录时间
    epoch = int(end // duration_time)  # 划分为多少切片

    fz = int(len(raw) / end)  # 采样频率
    print("end:{},  epoch:{}times,  fz:{}hz".format(end, epoch, fz))
    sleep_time_frame = []
    for index in range(epoch - 1):
        start = index * fz * duration_time
        stop = (index + 1) * fz * duration_time
        data, time = raw[picks_eeg, start:stop]
        sleep_time_frame.append(data)
    tmp = np.array(sleep_time_frame)
    sleep_time_frame = tmp.reshape((-1, fz * duration_time))  # 数据标准化的处理
    save_path = "../data/output_data/sleep_frame_eeg.npy"
    save_numpy_info(sleep_time_frame, save_path)
    return sleep_time_frame


def read_sleep_date(path):
    '''
    :param path: 读取文件的路径
    :return: 返回读取的数据
    '''
    data = np.load(path)
    return data
