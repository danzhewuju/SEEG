import re
import sys
from scipy import fftpack
import numpy as np
from handel import feature_analysis
import matplotlib.pyplot as plt
from util.util_file import matrix_normalization

sys.path.append("../")
import json
import os
import re


def test_fft(path):
    Fs = 50.0;  # sampling rate采样率
    Ts = 1.0 / Fs;  # sampling interval 采样区间
    t = np.arange(0, 1, Ts)  # time vector,这里Ts也是步长

    ff = 100;  # frequency of the signal
    data = np.load(path)
    y = data[0]

    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq1 = frq[range(int(n / 2))]  # one side frequency range

    YY = np.fft.fft(y)  # 未归一化
    Y = np.fft.fft(y) / n  # fft computing and normalization 归一化
    Y1 = Y[range(int(n / 2))]

    fig, ax = plt.subplots(4, 1)

    ax[0].plot(t, y)
    ax[0].set_xlabel('Time')
    ax[0].set_ylabel('Amplitude')

    ax[1].plot(frq, abs(YY), 'r')  # plotting the spectrum
    ax[1].set_xlabel('Freq (Hz)')
    ax[1].set_ylabel('|Y(freq)|')

    ax[2].plot(frq, abs(Y), 'G')  # plotting the spectrum
    ax[2].set_xlabel('Freq (Hz)')
    ax[2].set_ylabel('|Y(freq)|')

    ax[3].plot(frq1, abs(Y1), 'B')  # plotting the spectrum
    ax[3].set_xlabel('Freq (Hz)')
    ax[3].set_ylabel('|Y(freq)|')

    plt.show()


def test_data_shape(path):
    data = np.load(path)
    print(data.shape)

def draw_seeg_picture(data, sampling=500, x_axis='Time [sec]', y_axis='Channel'):
    '''

    :param data: SEEG读取的信号， 进行可视化的读取
    :return:
    '''
    width = data.shape[1]
    height = data.shape[0]
    dpi = 100
    plt.figure(figsize=(4, 3), dpi=200)
    # my_x_ticks = np.arange(0, width // sampling, 1.0 / sampling)  # 原始数据有width个数据，故此处为设置从0开始，间隔为1/sampling
    # plt.xticks(my_x_ticks)
    plt.xlabel(x_axis)
    # plt.ylabel(y_axis)
    # plt.axis('off')
    plt.imshow(data, aspect='auto')
    plt.show()
    plt.close()


def draw():
    path = '/home/cbd109-3/Users/data/yh/Program/Python/SEEG/visualization_feature/raw_data_time_sequentially/preseizure/BDP/filter/pre_1/e076f2ea-2552-11ea-9699-e0d55e6ff654-0.npy'
    data = np.load(path, allow_pickle=True)
    data = matrix_normalization(data)
    draw_seeg_picture(data, sampling=100)




if __name__ == '__main__':
    # path = "./log/BDP/preseizure/BDP-random_sample.npy"
    # # test_fft(path)
    # feature_analysis(feature_data_path=path)
    # path = "./log/BDP/preseizure/BDP-feature.npy"
#     # test_data_shape(path)
#     path = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/visualization_feature/raw_data_time_sequentially/preseizure/BDP/filter/pre_1/e076f2d4-2552-11ea-9699-e0d55e6ff654-0.npy"
#     data = np.load(path)
#     print(data.shape)
    draw()
