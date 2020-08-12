#!/usr/bin/Python
"""
@File    : Util.py
@Time    : 2019/9/9 20:48
@Author  : Alex
@Software: PyCharm
@Function:
"""
import pandas as pd
import os
import json

config = json.load(open("./json_path/config.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']
classification = config['classification']


def cal_acc_visualization():
    path = "./log/{}/{}/heatmap.csv".format(patient_test, classification)
    if os.path.exists(path) is False:
        print("{} file is not existed!".format(path))
        exit(0)
    data = pd.read_csv(path, sep=',')
    data_p = data['prediction'].tolist()
    data_g = data['ground truth'].tolist()
    count = 0
    for p in range(len(data_p)):
        if data_p[p] == data_g[p]:
            count += 1
    acc = count / len(data_p)
    print("accuracy: {:.2f}%".format(acc * 100))


def cal_time_info(path_dir, label, save_file):
    '''
    用于计算切片的时间信息， 每一个切片会对应一个起始时间
    :param path: 所在的文件夹
    :param label: 需要预设的标签
    :param save_file: 保存的文件位置
    :return:
    '''
    path_list = os.listdir(path_dir)
    path_list.sort()
    start_time = [2 * x for x in range(len(path_list))]
    labels = [label] * len(path_list)
    data_dict = {'id': path_list, 'start time': start_time, 'labels': labels}
    dataframe = pd.DataFrame(data_dict)
    if not os.path.exists(save_file):
        dataframe.to_csv(save_file, index=False)
        print("{} file has been saved!".format(save_file))
    else:
        old_data = pd.read_csv(save_file)
        new_data_frame = pd.concat([old_data, dataframe], axis=0, ignore_index=True)
        new_data_frame.to_csv(save_file, index=False)
        print("New data has been added in {} file.".format(save_file))
    return


if __name__ == '__main__':
    #     cal_acc_visualization()  # 计算数据的准确率，主要是用于 cal 的计算
    label = "non_seizure"
    dict_label = {"pre_seizure": 'pre_seizure', "non_seizure": 'sleep_normal'}
    path_dir = "./valpatient_data/{}/{}".format(patient_test, dict_label[label])
    save_file = "./log/{0}/{0}_time_index.csv".format(patient_test)
    cal_time_info(path_dir, label, save_file)  # 用于计算切片的时间信息， 每一个切片会对应一个起始时间
