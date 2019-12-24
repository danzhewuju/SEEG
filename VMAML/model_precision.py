#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/19 15:02
# @Author  : Alex
# @Site    : 
# @File    : model_precision.py
# @Software: PyCharm

import argparse
import os
import sys
from scipy.special import softmax
import pickle
import time

import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader, Dataset
from Seeg_VMAML import VAE
from util.util_file import matrix_normalization

sys.path.append('../')
import random
from MAML.Mamlnet import Seegnet
from VMAML.vmeta import *
from util.util_file import Pyemail, LogRecord
import json
from functools import partial

config = json.load(open("../DataProcessing/config/fig.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=4000)
argparser.add_argument('--n_way', type=int, help='n way', default=2)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
argparser.add_argument('--imgsz', type=int, help='imgsz', default=100)
argparser.add_argument('--imgc', type=int, help='imgc', default=5)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
argparser.add_argument('--vae_lr', type=float, help='meta-level outer learning rate', default=0.002)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=20)
argparser.add_argument('--dataset_dir', type=str, help="training data set",
                       default="../data/seeg/zero_data/{}".format(patient_test))
argparser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
argparser.add_argument('-train_p', '--train_path', default='../data/seeg/zero_data/{}/train'.format(patient_test))
argparser.add_argument('-test_p', '--test_path', default='../data/seeg/zero_data/{}/test'.format(patient_test))
argparser.add_argument('-val_p', '--val_path', default='../data/seeg/zero_data/{}/val'.format(patient_test))

args = argparser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
TEST_PATH = args.test_path
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
VAE_LR = args.vae_lr
CNN_batch_size = 16
device = torch.device("cuda" if args.cuda else "cpu")

# 模型的选择 1.CNN 2.MAML
model_selection = "CNN"

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

config = [
    ('conv2d', [32, 1, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 1, 0]),
    ('flatten', []),
    ('linear', [args.n_way, 7040])]

resize = (130, 200)

x_ = 8
y_ = 12
NUM_CLASS = 2


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(x_ * y_ * 32, 32)  # x_ y_ 和你输入的矩阵有关系
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, NUM_CLASS)  # 取决于最后的个数种类

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)  # 这里面的-1代表的是自适应的意思。
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class Data_info():
    def __init__(self, path_val, label):
        # pre-seizure: 0  non-seizure:1
        names = os.listdir(path_val)
        full_path = [(os.path.join(path_val, x), label, x) for x in names]
        self.full_path = full_path
        self.data_length = len(full_path)
        self.names = names


class MyDataset(Dataset):  # 重写dateset的相关类
    def __init__(self, imgs, transform=None, target_transform=None):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label, name_id = self.imgs[index]
        data = np.load(fn)
        result = matrix_normalization(data, (130, 200))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        return result, label, name_id

    def __len__(self):
        return len(self.imgs)


def save_file_util(dir, name):
    file_path = __file__
    dir_name = os.path.dirname(file_path)
    new_dir = os.path.join(dir_name, dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    save_name = os.path.join(new_dir, name)
    return save_name


def precision_vmaml():
    # 模型世界的状态

    path = "../visualization_feature/raw_data_time_sequentially/{}/{}/filter/".format(state_dic[true_label],
                                                                                      patient_test)
    print("path:{}".format(path))
    data_info = Data_info(path, true_label)
    # print(data_info.full_path)
    # print(sorted(data_info.full_path))
    my_dataset = MyDataset(data_info.full_path)
    maml = Meta(args, config).to(device)
    model_path = str(
        "./models/{}/maml".format(patient_test) + str(args.n_way) + "way_" + str(args.k_spt) + "shot_{}.pkl".format(
            patient_test))
    if os.path.exists(model_path):
        maml.load_state_dict(torch.load(model_path))
    else:
        print("model is not exist!")
    maml_net = maml.net
    pre_result = {}

    for data, label, name_id in my_dataset:
        data = data[np.newaxis, :]
        data = torch.from_numpy(data)
        data = data.to(device)
        with torch.no_grad():
            result = maml_net(data)
            c_result = result.cpu().detach().numpy()
            r = softmax(c_result)
            pre_y = r.argmax(1)
            pre_result[name_id] = pre_y

    save_name = save_file_util("precision", "{}-{}.pkl".format(patient_test, "pre-seizure-precision-0"))
    # save_name = save_file_util("precision", "{}-{}.pkl".format(patient_test, "sleep-precision-1"))
    with open(save_name, 'wb') as f:
        pickle.dump(pre_result, f)
        print("save success!")
    sum_length = len(pre_result)
    count = 0
    for name_id, pre in pre_result.items():
        if pre == true_label:
            count += 1
    print("Accuracy: {}".format(count / sum_length))


def precision_cnn():
    # data_path = "../visualization_feature/raw_data_time_sequentially/{}/{}/filter".format(state_dic[true_label],
    #                                                                                             patient_test)
    data_path = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/data/seeg/zero_data/BDP/val/sleep_normal"
    print("path:{}".format(data_path))
    data_info = Data_info(data_path, true_label)
    my_dataset = MyDataset(data_info.full_path)
    dataloader = DataLoader(my_dataset, batch_size=CNN_batch_size, shuffle=False)
    cnn_model = CNN().cuda(0)
    model_path = "../RelationNet/models/cnn_model/model-cnn_{}.ckpt".format(patient_test)
    if os.path.exists(model_path):
        cnn_model.load_state_dict(torch.load(model_path))
        print("CNN model loaded success! {}".format(model_path))
    else:
        print("modle is not exist!")
        exit(0)
    pre_result = {}
    with torch.no_grad():
        for data, label, name_id in dataloader:
            data = data.cuda(device)

            outputs = cnn_model(data)
            # print(outputs)
            # c_result = outputs.cpu().detach().numpy()
            # r = softmax(c_result, axis=1)
            # pre_y = r.argmax(1)[0]
            _, predicted = torch.max(outputs.data, 1)
            pre_y = predicted

            pre_result[name_id[0]] = pre_y

    save_name = save_file_util("precision", "{}-{}.pkl".format(patient_test, "pre-seizure-precision-0"))
    # save_name = save_file_util("precision", "{}-{}.pkl".format(patient_test, "sleep-precision-1"))
    with open(save_name, 'wb') as f:
        pickle.dump(pre_result, f)
        print("save success!")
    sum_length = len(pre_result)
    count = 0
    for name_id, pre in pre_result.items():
        if pre == true_label:
            count += 1
    print("Accuracy: {}".format(count / sum_length))


if __name__ == '__main__':
    true_label = 0
    state_dic = {0: "sleep", 1: "preseizure"}
    # run_dict = {"VMAML": partial(precision_vmaml), "CNN": partial(precision_cnn)}
    # run_dict[model_selection]
    if model_selection == "CNN":
        precision_cnn()
    else:
        precision_vmaml()
