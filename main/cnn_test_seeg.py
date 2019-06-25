#!/usr/bin/Python
'''
author: Alex
function: test cnn model
'''

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util.util_file import *

parser = argparse.ArgumentParser(description="CNN parameter setting!")
parser.add_argument('-t', '--time', default=2)  # 每一帧的长度
parser.add_argument('-s', '--sample', default=100)  # 对其进行重采样
parser.add_argument('-train_p', '--train_path', default='../data/seeg/train')
parser.add_argument('-test_p', '--test_path', default='../data/seeg/test')
parser.add_argument('-val_p', '--val_path', default='../data/seeg/val')
parser.add_argument('-m_p', '--model_path', default='./models/model-cnn.ckpt')
parser.add_argument('-g', '--GPU', type=int, default=0)
parser.add_argument('-n', '--class_number', type=int, default=2)
parser.add_argument('-b', '--batch_size', type=int, default=16)
parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epoch', type=int, default=10)
args = parser.parse_args()

# hyper parameter setting

TEST_PATH = args.test_path
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path  # 验证数据的文件夹
MODLE_PATH = args.model_path
GPU = args.GPU  # 使用哪个GPU
NUM_CLASS = args.class_number  # 分类的个数
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCH = args.epoch

# input = 130*200
x_ = 8
y_ = 12


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
        # plt.imshow(out)
        # plt.show()
        out = out.reshape(out.size(0), -1)  # 这里面的-1代表的是自适应的意思。
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class Data_info():
    def __init__(self, path_val):  # verification dataset
        index_name_val = os.listdir(path_val)
        data_val = []
        for index, name in enumerate(index_name_val):
            path = os.path.join(path_val, name)
            dir_names = os.listdir(path)
            for n in dir_names:
                full_path = os.path.join(path, n)
                data_val.append((full_path, index))

        self.val = data_val
        self.val_length = len(data_val)


class MyDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas

    def __getitem__(self, item):
        d_p, label = self.datas[item]
        data = np.load(d_p)
        data = matrix_normalization(data, (130, 200))
        data = data.astype('float32')
        data = data[np.newaxis, :]
        return data, label

    def __len__(self):
        return len(self.datas)


def run():
    start_time = time.time()  # 开始时间
    data_info = Data_info(VAL_PATH)
    val_data = MyDataset(data_info.val)  # 标准数据集的构造?
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN().cuda(GPU)  # 保持和之前的神经网络相同的结构特征?
    model.load_state_dict(torch.load(MODLE_PATH))
    print("Loading {} model!".format(MODLE_PATH))

    total = data_info.val_length
    correct = 0

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance) 对于单个图片的测试
    with torch.no_grad():
        for (data, labels) in val_loader:
            data = data.cuda(GPU)
            labels = labels.cuda(GPU)

            outputs = model(data)  # 直接获得模型的结果
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the {} test seegs: {} %'.format(total,
                                                                         100 * correct / total))
    end_time = time.time()
    run_time = end_time - start_time
    print("Running Time {:.4f}".format(run_time))


if __name__ == '__main__':
    run()
