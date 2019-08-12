#!/usr/bin/Python
# -*- coding: utf-8 -*-
# @Time    : 2019/8/6 14:59
# @Author  : Alex
# @Site    : 
# @File    : handel.py
# @Software: PyCharm

import json
import sys
sys.path.append('../')
from util import *
import re
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os


def get_hotmap_dic(path_hotmap, path_b_raw_data):
    path_all_a = get_first_dir_path(path_hotmap)
    path_all_name_hot = []
    for p in path_all_a:
        d = re.sub('-loc(.+).jpg', '', p)
        raw_path = d + ".npy"
        name = re.findall('examples/(.+)', raw_path)[0]
        path_all_name_hot.append(name[:-4])
    dict_hot = dict(zip(path_all_a, path_all_name_hot))
    path_all_b = get_first_dir_path(path_b_raw_data)
    path_all_name_raw = []
    for p in path_all_b:
        name = re.findall('raw_data_signal/(.+)', p)[0]
        path_all_name_raw.append(name[:-4])
    name_save = [x+".jpg" for x in  path_all_name_raw]
    dict_raw = dict(zip(path_all_b, path_all_name_raw))
    re1 = dict(sorted(dict_hot.items(), key=lambda x: x[1]))
    re2 = dict(sorted(dict_raw.items(), key=lambda x: x[1]))
    # test_v_1 = list(re1.values())
    # test_v_2 = list(re2.values())
    # print(test_v_2 == test_v_1)
    # for i in range(len(test_v_1)):
    #     p1 = test_v_1[i]
    #     p2 = test_v_2[i]
    #     if p1 != p2:
    #         print(p1, p2)
    path_hotmap_r = re1.keys()
    path_raw_r = re2.keys()
    dict_result = dict(zip(path_hotmap_r, path_raw_r))
    return dict_result, name_save


def create_raw_data_signal(top_k=100, image_dir="./examples"):
    # hot_map_paths = get_first_dir_path('./examples', suffix='jpg')
    # t = re.findall("loc-(.+).jpg", hot_map_paths[0])[0]
    # print(t)
    #
    # channels_tmp = list(map(int, t.split("-")))
    # channels = list(set(channels_tmp))
    # channels.sort()
    # print(channels)

    with open('./json_path/LK_preseizure_sorted.json') as f:
        json_path = json.load(f)

        raw_data_name = [re.findall("/.+/(.+)", p)[0] for p in json_path]
        dict_name_path = dict(zip(raw_data_name, json_path))
        images_path = get_first_dir_path(image_dir)
        selected_raw_path = []
        for i_p in images_path:
            path_tmp = i_p
            channels_str = re.findall('-loc-(.+).jpg', path_tmp)[0]
            channels_number = map(int, channels_str.split('-'))
            channels_number = list(set(channels_number))
            if len(channels_number) > 3:
                channels_number = channels_number[:3]
            else:
                if len(channels_number) == 1:
                    min_no = channels_number[0] if channels_number[0] > 0 else 1
                    channels_number.append(min_no-1)
                    channels_number.append(min_no+1)
                    channels_number.sort()

            channels_number.sort()

            d = re.sub('-loc(.+).jpg', '', path_tmp)
            raw_path = d + ".npy"
            name = re.findall('examples/(.+)', raw_path)[0]
            selected_raw_path.append((name, channels_number))

        for index, (name, channels_number) in tqdm(enumerate(selected_raw_path)):
            new_name = name[:-4] + '.jpg'
            save_path = os.path.join('./raw_data_signal', new_name)
            data_p = dict_name_path[name]
            raw_data = np.load(data_p)
            # plt.subplot(121)
            seeg_npy_plot(raw_data, channels_number, save_path)
            # plt.subplot(122)
            # image = plt.imread(images_path[index])
            # plt.imshow(image)
            # plt.savefig(save_path)
            # # plt.show()
        print("All files has been written!")

        # 进行路径的转换

        # data_path = "../data/data_slice/split/preseizure/LK/17673574-a894-11e9-bc3a-338334ea1429-0.npy"
        # data = np.load(data_path)
        #
        # seeg_npy_plot(data, [23, 24, 25])
        #
        # data_path = "../data/data_slice/split/preseizure/LK/176811ba-a894-11e9-bc3a-338334ea1429-0.npy"
        # data = np.load(data_path)
        # seeg_npy_plot(data, [23, 24, 25])
        #
        # data_path = "../data/data_slice/split/preseizure/LK/12e74ffc-a894-11e9-bc3a-338334ea1429-0.npy"
        # data = np.load(data_path)
        # seeg_npy_plot(data, [23, 24, 25])
        #
        # data_path = "../data/data_slice/split/preseizure/LK/06d1da02-a894-11e9-bc3a-338334ea1429-0.npy"
        # data = np.load(data_path)
        # seeg_npy_plot(data, [72, 73, 74])
        #
        # data_path = "../data/data_slice/split/preseizure/LK/17647370-a894-11e9-bc3a-338334ea1429-0.npy"
        # data = np.load(data_path)
        # seeg_npy_plot(data, [3, 4, 5])
        #
        # data_path = "../data/data_slice/split/preseizure/LK/176976a4-a894-11e9-bc3a-338334ea1429-0.npy"
        # data = np.load(data_path)
        # seeg_npy_plot(data, [39, 40, 41])


# def test_FFT():
#     data_path = "../data/data_slice/split/preseizure/LK/17673574-a894-11e9-bc3a-338334ea1429-0.npy"
#     data = np.load(data_path)
#     channel_number = 25
#     data_channel = data[channel_number]
#     sampling_rate = 100  # 采样频率为100Hz
#     fft_size = 200  # FFT处理的取样长度
#     # N点FFT进行精确频谱分析的要求是N个取样点包含整数个取样对象的波形。因此N点FFT能够完美计算频谱对取样对象的要求是n*Fs/N（n*采样频率/FFT长度），
#     # 因此对8KHZ和512点而言，完美采样对象的周期最小要求是8000/512=15.625HZ,所以156.25的n为10,234.375的n为15。
#     xs = data_channel  # 从波形数据中取样fft_size个点进行运算
#     xf = np.fft.rfft(xs) / fft_size  # 利用np.fft.rfft()进行FFT计算，rfft()是为了更方便对实数信号进行变换，由公式可知/fft_size为了正确显示波形能量
#     # rfft函数的返回值是N/2+1个复数，分别表示从0(Hz)到sampling_rate/2(Hz)的分。
#     # 于是可以通过下面的np.linspace计算出返回值中每个下标对应的真正的频率：
#     freqs = np.linspace(0, sampling_rate / 2, fft_size / 2 + 1)
#     # np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
#     # 在指定的间隔内返回均匀间隔的数字
#     xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
#     # 最后我们计算每个频率分量的幅值，并通过 20*np.log10()将其转换为以db单位的值。为了防止0幅值的成分造成log10无法计算，我们调用np.clip对xf的幅值进行上下限处理
#
#     # 绘图显示结果
#     plt.figure(figsize=(8, 4))
#     plt.subplot(211)
#     plt.plot(data_channel[:fft_size], xs)
#     plt.xlabel(u"Time(S)")
#     plt.title(u"156.25Hz and 234.375Hz WaveForm And Freq")
#     plt.subplot(212)
#     plt.plot(freqs, xfp)
#     plt.xlabel(u"Freq(Hz)")
#     plt.subplots_adjust(hspace=0.4)
#     plt.show()

def image_connection(data_signal_dir, raw_data_dir, save_dir = "./contact_image"):
    if os.path.exists(save_dir) is not True:
        os.mkdir(save_dir)
    signal_dir_t = os.listdir(data_signal_dir)
    signal_dir_path = [os.path.join(data_signal_dir, x) for x in signal_dir_t]
    dict_hot_raw, save_name = get_hotmap_dic(data_signal_dir, raw_data_dir)
    for p in tqdm(signal_dir_path):
        index = signal_dir_path.index(p)
        name = save_name[index]
        save_path = os.path.join(save_dir, name)
        path_test_1 = p
        path_test_2 = dict_hot_raw[p]
        plt.figure(0)
        imag_test = Image.open(path_test_1)
        dst_1 = imag_test.transpose(Image.ROTATE_90)
        dst_2 = Image.open(path_test_2)

        f = plt.figure(figsize=(7, 12))
        ax = f.add_subplot(211)
        ax.axis('off')
        ax2 = f.add_subplot(212)
        ax2.axis('off')
        ax.imshow(dst_1)
        ax2.imshow(dst_2)
        # plt.subplot(2, 1, 1)
        # plt.imshow(dst_1)
        #
        #
        # plt.subplot(2, 1, 2)
        # plt.imshow(dst_2)
        # plt.axis('off')
        plt.savefig(save_path)
        # plt.show()
        plt.close(0)
        # plt.show()


if __name__ == '__main__':
    path_a = "./examples"
    path_b = "./raw_data_signal"
    create_raw_data_signal(top_k=100)
    image_connection('./examples', './raw_data_signal')
    # get_hotmap_dic(path_a, path_b)
