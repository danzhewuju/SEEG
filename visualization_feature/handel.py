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
from DataProcessing.transferdata import *
import re
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os
from grad_cam import *


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
    name_save = [x + ".jpg" for x in path_all_name_raw]
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


def create_raw_data_signal(image_dir="./examples"):
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
                    channels_number.append(min_no - 1)
                    channels_number.append(min_no + 1)
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
            seeg_npy_plot(raw_data, channels_number, save_path)

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


def image_connection(data_signal_dir, raw_data_dir, save_dir="./contact_image"):
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


def time_heat_map(path="./raw_data_time_sequentially/preseizure/LK"):
    '''

    :return:
    构造时间序列的热力图
    '''
    heat_map_dir = "./examples"
    path_data = get_first_dir_path(path, 'npy')
    path_data.sort()  # 根据uuid 按照时间序列进行排序
    count = 30  # 拼接的时间
    clean_dir(heat_map_dir)  # 清除文件夹里面所有文件
    for p in path_data:
        get_feature_map(p)
    heat_map_path = get_first_dir_path(heat_map_dir)
    heat_map_path.sort()
    test_1 = Image.open(heat_map_path[0])
    dst_1 = test_1.transpose(Image.ROTATE_90)
    size = dst_1.size
    plt.figure(figsize=(2*count, 3))
    result = Image.new(dst_1.mode, (size[0] * count, size[1]))
    for i in range(count):
        img = Image.open(heat_map_path[i])
        img_t = img.transpose(Image.ROTATE_90)
        result.paste(img_t, box=(i * size[0], 0))
    # result.save("./60s.png")
    plt.imshow(result)
    plt.show()


def image_contact_process():  # 流程处理函数
    path_a = "./examples"
    path_b = "./raw_data_signal"
    create_raw_data_signal()  # 生成原始信号中梯队最高的信道的信号图像
    image_connection('./examples', './raw_data_signal')


def raw_data_without_filter_process():
    '''
    1.医生需要原始的数据，需要未经过切片的原始数据，因此此时需要重写相关函数。不经过滤波
    :return:
    '''
    # 1.癫痫发作前的原始数据的重写
    path_commom_channel = "../data/data_slice/channels_info/LK_seq.csv"
    path_dir = "../data/raw_data/LK/LK_Pre_seizure"
    flag = 0
    for index, p in enumerate(os.listdir(path_dir)):
        if index < 1:
            path_raw = os.path.join(path_dir, p)
            name = "LK"
            generate_data(path_raw, flag, name, path_commom_channel, isfilter=False)
    print("癫痫发作前的睡眠处理完成！！！")

    # 2.正常数据的重写
    path_commom_channel = "../data/data_slice/channels_info/LK_seq.csv"
    path_raw_normal_sleep = ["../data/raw_data/LK/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-0.fif",
                             '../data/raw_data/LK/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-1.fif',
                             '../data/raw_data/LK/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-2-0.fif',
                             '../data/raw_data/LK/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-4-0.fif',
                             '../data/raw_data/LK/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-6-0.fif'

                             ]  # 数据太多，因此只是选取部分的数据进行处理
    name = "LK"
    flag = 2  # 正常睡眠时间

    for index, path_raw in enumerate(path_raw_normal_sleep):
        if index < 1:
            generate_data(path_raw, flag, name, path_commom_channel, isfilter=False)

    print("{}正常睡眠的数据处理完成！".format(name))


if __name__ == '__main__':
    # 1. 将两个原信号连接在一起
    # image_contact_process()

    # 2. 生成未滤波数据的切片
    # raw_data_without_filter_process()

    # 3. 拼接热力图， 将热力图按照时间序列进行拼接
    time_heat_map()

