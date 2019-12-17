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
from DataProcessing.transferdata import generate_data, resample
import re
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import os
from grad_cam import patient_test, classification, get_feature_map, get_feature_map_dynamic
from util.util_file import dir_create_check
import pandas as pd
from functools import partial
from Util import cal_acc_visualization


def get_hotmap_dic(path_hotmap, path_b_raw_data):
    path_all_a = get_first_dir_path(path_hotmap)
    path_all_name_hot = []
    for p in path_all_a:
        d = re.sub('-loc(.+).jpg', '', p)
        raw_path = d + ".npy"
        name = re.findall('heatmap/(.+)', raw_path)[0]
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


def create_raw_data_signal_by_similarity(image_dir="./heatmap"):
    # hot_map_paths = get_first_dir_path('./heatmap', suffix='jpg')
    # t = re.findall("loc-(.+).jpg", hot_map_paths[0])[0]
    # print(t)
    #
    # channels_tmp = list(map(int, t.split("-")))
    # channels = list(set(channels_tmp))
    # channels.sort()
    # print(channels)

    with open('./json_path/BDP_preseizure_sorted.json') as f:
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
            name = re.findall('heatmap/(.+)', raw_path)[0]
            selected_raw_path.append((name, channels_number))

        for index, (name, channels_number) in tqdm(enumerate(selected_raw_path)):
            new_name = name[:-4] + '.jpg'
            save_path = os.path.join('./raw_data_signal', new_name)
            data_p = dict_name_path[name]
            raw_data = np.load(data_p)
            seeg_npy_plot(raw_data, channels_number, save_path)

        print("All files has been written!")


# 将原始的数据和信道相结合
def create_raw_data_signal_by_time(image_dir="./log/{}/{}/heatmap".format(patient_test, classification)):
    # patient_test = "ZK"

    raw_path_list = get_first_dir_path("./raw_data_time_sequentially/{}/".format(classification) + patient_test, "npy")
    raw_data_name = [re.findall("/.+/(.+)", p)[0] for p in raw_path_list]
    dict_name_path = dict(zip(raw_data_name, raw_path_list))

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
        name = re.findall('heatmap/(.+)', raw_path)[0]
        selected_raw_path.append((name, channels_number))

    for index, (name, channels_number) in enumerate(tqdm(selected_raw_path)):
        new_name = name[:-4] + '.jpg'
        name_npy = name[:-4] + '.npy'
        save_path = os.path.join('./log/{}/{}/raw_data_signal'.format(patient_test, classification), new_name)
        path_tmp = "./log/{}/{}/raw_data_signal_data_storage".format(patient_test, classification)
        dir_create_check(path_tmp)
        save_path_npy = os.path.join(path_tmp, name_npy)
        data_p = dict_name_path[name]
        raw_data = np.load(data_p)  # 此时相当于原始数据，信道数已经不一致了
        # 需要重新映射channel number 的序号

        seeg_npy_plot(raw_data, channels_number, save_path, save_path_npy=save_path_npy)

    print("All files has been written!")


def image_connection(data_signal_dir, raw_data_dir,
                     save_dir="./log/{}/{}/contact_image".format(patient_test, classification)):
    clean_dir("./log/{}/{}/contact_image".format(patient_test, classification))
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
        dst_2 = Image.open(path_test_2)

        f = plt.figure(figsize=(7, 12))
        ax = f.add_subplot(211)
        ax2 = f.add_subplot(212)
        ax.imshow(imag_test)
        ax2.imshow(dst_2)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close(0)


def time_heat_map():
    '''

    :return:
    构造时间序列的热力图
    '''
    max_lenth = 2083
    clean_dir("./log/{}/{}".format(patient_test, classification))
    # patient_test = "ZK"
    path = "./raw_data_time_sequentially/{}/".format(classification) + patient_test
    file_name = "{0}/{1}/{0}_SZ1_pre_seizure_raw".format(patient_test,
                                                         classification) if classification == "preseizure" else "{0}/{1}/{0}_Sleep_raw".format(
        patient_test, classification)
    # 指定了这个文件来让医生进行验证
    file_name = file_name + ".txt"
    heat_map_dir = "./log/{}/{}/heatmap".format(patient_test, classification)
    path_data = get_first_dir_path(path, 'npy')
    path_data.sort()  # 根据uuid 按照时间序列进行排序
    count = min(len(path_data), max_lenth)  # 拼接的时间
    selected_path = path_data[:count]
    dir_create_check(heat_map_dir)
    # clean_dir(heat_map_dir)  # 清除文件夹里面所有文件
    for index, p in enumerate(selected_path):
        get_feature_map(p, file_name)
    heat_map_path = get_first_dir_path(heat_map_dir)
    heat_map_path.sort()
    test_1 = Image.open(heat_map_path[0])
    size = test_1.size
    plt.figure(figsize=(2 * count, 3))
    result = Image.new(test_1.mode, (size[0] * count, size[1]))
    prediction = pd.read_csv("./log/{}/{}/heatmap.csv".format(patient_test, classification), sep=',')
    for i in range(count):
        if prediction.loc[i]['ground truth'] != prediction.loc[i]['prediction']:
            img = Image.new('RGBA', (int(200), int(130)), color=(100, 100, 100, 255))
        else:
            img = Image.open(heat_map_path[i])
        result.paste(img, box=(i * size[0], 0))
    result.save("./log/{}/{}/heatmap{}.png".format(patient_test, classification, count * 2))
    # plt.imshow(result)
    # plt.show()


def image_contact_process_by_similarity():  # 流程处理函数
    path_a = "./log/{}/{}/heatmap".format(patient_test, classification)
    path_b = "./log/{}/{}/raw_data_signal".format(patient_test, classification)

    create_raw_data_signal_by_similarity()  # 生成原始信号中梯队最高的信道的信号图像
    image_connection(path_a, path_b)


def image_contact_process_by_time():
    path_a = "./log/{}/{}/heatmap".format(patient_test, classification)
    path_b = "./log/{}/{}/raw_data_signal".format(patient_test, classification)
    dir_create_check(path_a)
    dir_create_check(path_b)

    create_raw_data_signal_by_time()  # 生成原始信号中梯队最高的信道的信号图像
    image_connection(path_a, path_b)


def raw_data_slice():
    '''
    1.医生需要原始的数据，需要未经过切片的原始数据，因此此时需要重写相关函数。不经过滤波,数据是没有按照时间顺序来排序
    :return:
    '''
    # patient_test = "ZK"
    flag_dic = {"preseizure": 0, "sleep": 2}
    config = "./json_path/config.json"
    config_json = json.load(open(config))
    path_commom_channel = config_json["handel.dynamic_detection__path_channel_list_" + patient_test]

    # 1.癫痫发作前的原始数据的重写
    # 2.正常数据的重写
    # execute
    flag = flag_dic[classification]
    clean_dir("./raw_data_time_sequentially/{}/{}".format(classification, patient_test))  # 删除文件夹下面已有的旧的文件

    path_dir = config_json["person_raw_data_{}_".format(classification) + patient_test]
    print("processing data from {}".format(path_dir))
    path_raw = path_dir
    name = patient_test
    generate_data(path_raw, flag, name, path_commom_channel, isfilter=True)

    print("癫痫发作前的睡眠处理完成！！！") if flag == 0 else print("正常睡眠数据处理完成！！！")


def sequentially_signal(config="./json_path/config.json"):  # 时间序列的热点分析
    config_json = json.load(open(config))
    # patient_test = "ZK"
    path = config_json['handel.sequentially__path_{}_{}'.format(classification, patient_test)].format(patient_test,
                                                                                                      classification)

    channel_list_path = config_json['handel.sequentially__path_channel_list_' + patient_test]
    channel_pandas = pd.read_csv(channel_list_path)
    channel_list = channel_pandas['chan_name']  # 获得与信道对应的index-channel的列表

    start_time = config_json['handel.sequentially__start_time_{}_{}'.format(classification, patient_test)]
    start_time_list = [int(x) for x in start_time.split(":")]

    save_signal_info = config_json['handel.sequentially__save_dir'].format(patient_test, classification)

    data = pd.read_csv(path, sep=',')
    location_time = data['time_location']
    location_spatial = data['spatial_location']

    channel_time = {}
    h, m, s = start_time_list[0], start_time_list[1], start_time_list[2]
    for index in range(len(location_time)):
        time_loc = int(location_time[index].split('-')[0])
        second_add = time_loc / 100  # 需要累积之前的时间,将时间的换算单位转化为:s
        second_add = round(second_add, 2)
        # 进行时间的累积计算
        h, m, s = time_add(h, m, s, second_add)
        s = round(s, 2)

        time_point = "{}:{}:{}".format(str(h), str(m), str(s))

        space_loc = int(location_spatial[index].split('-')[0])
        channel_name = channel_list[space_loc]  # 这个操作只对LK有效，因为其他病人的信道数并不存在相应的对应关系

        if channel_name not in channel_time.keys():
            channel_time[channel_name] = time_point
        else:
            tmp = channel_time[channel_name]
            result = tmp + ", " + time_point
            channel_time[channel_name] = result

    fp_signal_info = open(save_signal_info, 'w')
    fp_signal_info.write("文件路径信息：{} \t 起始时间:{}\n".format(path, start_time))
    for name, time in channel_time.items():
        line = name + ":\t" + time + "\n"
        fp_signal_info.write(line)
    fp_signal_info.close()
    print("All information has been written in {}".format(save_signal_info))


def dynamic_detection():
    '''

    :param raw_data_path: raw data path
    :return: heat_map
    动态热力图的检测，实现热点中间对齐的功能，需要包含完整的热点信息
    '''
    clean_dir("./log/{}/{}/heatmap".format(patient_test, classification))  # 清空heatmap 文件夹下面所有文件

    config_info = json.load(open("./json_path/config.json"))  # 读取配置文件信息
    raw_data_path = config_info['person_raw_data_{}_'.format(classification) + patient_test]
    channel_info_path = config_info['handel.dynamic_detection__path_channel_list_' + patient_test]  # 信道的排序信息表
    channel_names = pd.read_csv(channel_info_path, sep=',')
    channel_list = channel_names['chan_name'].tolist()
    raw_data = read_raw(raw_data_path)

    data = filter_hz(raw_data, 0, 30)  # 对数据进行滤波处理
    data.resample(resample, npad="auto")  # resample 100hz
    data = select_channel_data_mne(data, channel_list)  # 选择需要的信道
    data.reorder_channels(channel_list)  # 更改信道的顺序

    start_time = 0  # 开始的时间为0
    end_time = get_recorder_time(raw_data)  # 这个文件的全部时间
    time_gap = 2  # 文件的时间间隙
    flag_head = start_time  # 移动的标尺头部
    flag_tail = start_time + time_gap  # 移动标尺的尾部
    flag_head_new = flag_head
    flag_tail_new = flag_tail
    fz = int(len(raw_data) / end_time)  # 采样频率
    while flag_tail <= end_time:
        data_slice, _ = data[:, flag_head * fz:flag_tail * fz]
        name = "{}-{}.jpg".format(flag_head, flag_tail)
        if end_time - flag_tail < time_gap / 2 or flag_head - start_time < time_gap / 2:
            key_flag = False
        else:
            key_flag = True

        time = get_feature_map_dynamic(data_slice, name, key_flag)

        if time != -1:
            flag_head_new = flag_head + time - (time_gap / 2)
            flag_tail_new = flag_head + time + (time_gap / 2)
            data_slice, _ = data[:, int(flag_head_new * fz):int(flag_tail_new * fz)]
            key_flag = False
            name = "{:.2f}-{:.2f}.jpg".format(flag_head_new, flag_tail_new)
            get_feature_map_dynamic(data_slice, name, key_flag=key_flag)
        flag_head += 2
        flag_tail += 2


def save_feature_data():
    '''

    :return: 用于生成特征数据
    '''
    path_dir = "./log/{}/{}".format(patient_test, classification)  # 所要读取特征的文件夹
    data_dir = "./raw_data_time_sequentially/{}/{}".format(classification, patient_test)
    raw_data_sequence = get_first_dir_path(data_dir, suffix='npy')  # 原始数据的序列
    raw_data_sequence.sort()  # 按照时间序列来获得数据  # 数据
    # 找到文件列表中记录了features
    key_dict = {"preseizure": "pre_seizure", "sleep": "Sleep"}
    _ = os.listdir(path_dir)
    file_name = ""
    for p in _:
        if key_dict[classification] in p:
            file_name = p
            break
    file_path = os.path.join(path_dir, file_name)
    print("Finding file: {} ".format(file_name))
    data_frame = pd.read_csv(file_path)  # 特征的相关信息， 特征信息
    time_location = data_frame['time_location'].tolist()
    spatial_location = data_frame['spatial_location'].tolist()
    feature_data = []  # 将feature data 提取出来
    duration_time = 50  # 设定特征的持续时间50*10=500ms
    for i in range(len(time_location)):  # 保存选取得特征
        data = np.load(raw_data_sequence[i])
        time_start = int(spatial_location[i].split("-")[0])
        location_start = int(spatial_location[i].split("-")[0])
        start_time = max(0, time_start - duration_time // 2)  # 选择特征的区域
        end_time = min(200, time_start + duration_time // 2)
        if end_time - start_time < duration_time:
            data_pad = data[location_start][start_time:end_time]
            data_pad = np.pad(data_pad, (0, duration_time - (end_time - start_time)), 'constant')
            feature_data.append(data_pad)
        else:
            feature_data.append(data[location_start][start_time:end_time])  # 存储特征对应的波形
    save_path = os.path.join(path_dir, "{}-feature.npy".format(patient_test))
    feature_data = np.asarray(feature_data).reshape(len(feature_data), -1)
    np.save(save_path, feature_data)
    print("{} file has been saved!".format(save_path))


def feature_analysis(feature_data_path="./log/{0}/{1}/{0}-feature.npy".format(patient_test, classification)):
    '''

    :return: 主要用于特征的分析，
    '''
    # 是否加入特征的验证系统
    preseizure_name_list = ["BDP", "LK", "SYF", "ZK"]

    feature_data = np.load(feature_data_path)
    feature_score = []
    for p in tqdm(feature_data):
        score = 0
        for n in feature_data:
            score += similarity_dtw(p, n)
        score /= feature_data.shape[0]
        round(score, 6)
        feature_score.append(score)
    data_dic = {"ID": range(0, len(feature_score)), "Score": feature_score}
    data_frame = pd.DataFrame(data_dic)
    save_path = "./log/{0}/{1}/{0}-feature_score.csv".format(patient_test, classification)
    data_frame.to_csv(save_path, index=None)
    print("Similarity of features has finished. The result has been saved in {}".format(save_path))


def switch_patient():
    name_list = ["BDP", "LK", "SYF", "WSH", "ZK"]
    dict_map = dict(zip(range(len(name_list)), name_list))
    josn_path = "./json_path/config.json"
    json_config = "../DataProcessing/config/fig.json"
    with open(josn_path, 'r') as f:
        data = json.load(f)
        for k, v in dict_map.items():
            print("{}.{}".format(k, v))
        print("请输入你要切换的病人序号：", end="")
        key = int(input())
        selected_patient = dict_map[key]
        data["patient_test"] = selected_patient

    with  open(josn_path, 'w') as f:
        json.dump(data, f)
        print("{} patient_test 已经切换为:{}".format(josn_path, selected_patient))

    with open(json_config, 'r') as f:
        data = json.load(f)
        data['patient_test'] = selected_patient
    with open(json_config, 'w') as f:
        json.dump(data, f)
        print("{} patient_test 已近被切换为：{}".format(json_config, selected_patient))
    menu()


def random_sample():
    '''
    实现随机采样的方法
    :return:
    '''
    data_path = "./raw_data_time_sequentially/{}/{}".format(classification, patient_test)
    data_list = get_first_dir_path(data_path, suffix='npy')
    data_list.sort()
    data_random_sample = []
    duration = 50
    for p in data_list:
        data = np.load(p, allow_pickle=True)
        channel_no = random.randint(0, len(data) - 1)
        star = random.randint(0, data.shape[1] - duration - 1)
        data_sample = data[channel_no][star:star + duration]
        data_random_sample.append(data_sample)
    data_random_sample = np.array(data_random_sample).reshape(len(data_list), duration)
    save_path = "./log/{0}/{1}/{0}-random_sample.npy".format(patient_test, classification)
    np.save(save_path, data_random_sample)
    print("Sampling finished")


def menu():
    def display_line(messages):
        star = lambda x: "*" * x
        print(star(30))
        for i in range(len(messages)):
            print("{}".format(messages[i]))
        print(star(30))
        print("请输入你的操作：", end="")

    def display():
        messages = ["0.程序退出", "1.数据切片的生成", "2.拼接热力图", "3.计算时间序列", "4.时间片段和热力图的结合", "5.特征数据提取", "6.计算评价指标",
                    "7.特征相似度计算", "8.数据的随机采样", "9.病人切换"]
        display_line(messages)

    config = json.load(open("./json_path/config.json", 'r'))  # 需要指定训练所使用的数据
    patient_test = config['patient_test']  # "BDP"
    classification = config['classification']  # "sleep", "preseizure"
    print("patient_test is : {}, classification is : {}".format(patient_test, classification))
    path_dir = "./log/{}/{}".format(patient_test, classification)
    dir_create_check(path_dir)
    handle_menu = {1: partial(raw_data_slice), 2: partial(time_heat_map), 3: partial(sequentially_signal),
                   4: partial(image_contact_process_by_time), 0: partial(exit), 5: partial(save_feature_data),
                   6: partial(cal_acc_visualization), 7: partial(feature_analysis), 8: partial(random_sample),
                   9: partial(switch_patient)}

    while True:
        print("现在的测试病人是：{}, 测试状态是：{}".format(patient_test, classification))
        display()
        key = int(input())
        handle_menu[key]()


if __name__ == '__main__':
    # TODO: list
    # 1.0 需要运行 feature_hotmap.py 文件, 保证文件夹heatmao, raw_data_signal 里面存在照片
    # 1. 将两个原信号连接在一起,一个是热力信号，一个是原始的波形信号
    # image_contact_process_by_similarity()

    # 2.1 生成未滤波数据的切片, 可以设置是否选择滤波处理
    # raw_data_slice()

    # 2.2. 拼接热力图， 将热力图按照时间序列进行拼接,拼接我60s
    # time_heat_map()

    # 2.3 按照绝对时间来计算序列
    # sequentially_signal()

    # 2.3 将按照时间的片段信号和热力图进行结合
    # image_contact_process_by_time()

    # 3.1 从整体的文件进行热力分析， 以及热力图分割，读取完整的文件，防止热力图被分割
    # dynamic_detection()
    menu()
    # switch_patient()
