#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/24 13:32
# @Author  : Alex
# @Site    : 
# @File    : ConVae.py
# @Software: PyCharm

from __future__ import print_function

import argparse
import os

# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from util.util_file import matrix_normalization
import json

config = json.load(open("../DataProcessing/config/fig.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']
print("patient_test is {}".format(patient_test))

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-train_p', '--train_path', default='../data/seeg/zero_data/{}/train'.format(patient_test))
parser.add_argument('-test_p', '--test_path', default='../data/seeg/zero_data/{}/test'.format(patient_test))
parser.add_argument('-val_p', '--val_path', default='../data/seeg/zero_data/{}/val'.format(patient_test))
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
TEST_PATH = args.test_path
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
torch.manual_seed(args.seed)

device = torch.device("cuda")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

resize = (130, 200)


# def data_normalization(data):  # (32,1,128,200)->(32,1,130,200)
#     result = []
#     data_tmp = data.cpu().detach().numpy()
#     # data_tmp = data
#     length = data_tmp.shape[0]
#     for i in range(length):
#         data_r = data_tmp[i][0]
#         data_result = matrix_normalization(data_r, resize_shape=(128, 200))
#         data_result = data_result[np.newaxis, :]
#         result.append(data_result)
#     result_r = np.array(result)
#     # result_r = torch.from_numpy(result_r)
#     result_r = torch.from_numpy(result_r)
#     return result_r


# 数据的输入输出

class Data_info():
    def __init__(self, path_train, path_test):
        index_name_train = os.listdir(path_train)
        index_name_test = os.listdir(path_test)
        data_train = []
        data_test = []
        preseizure = []
        sleep_normal = []
        for index, name in enumerate(index_name_train):
            path = os.path.join(path_train, name)
            dir_names = os.listdir(path)
            for n in dir_names:
                full_path = os.path.join(path, n)
                if name == "pre_zeizure":
                    preseizure.append((full_path, index))
                else:
                    sleep_normal.append((full_path, index))
                data_train.append((full_path, index))

        for index, name in enumerate(index_name_test):
            path = os.path.join(path_test, name)
            dir_names = os.listdir(path)
            for n in dir_names:
                full_path = os.path.join(path, n)
                if name == "pre_zeizure":
                    preseizure.append((full_path, index))
                else:
                    sleep_normal.append((full_path, index))
                data_test.append((full_path, index))

        # t = time.time()
        # random.seed(t)
        # random.shuffle(data_train)
        # t = time.time()
        # random.seed(t)
        # random.shuffle(data_test)

        self.data_train = data_train
        self.data_test = data_test
        self.train_length = len(data_train)
        self.test_length = len(data_test)
        self.preseizure = preseizure
        self.sleep_normal = sleep_normal


class MyDataset(Dataset):  # 重写dateset的相关类
    def __init__(self, imgs, transform=None, target_transform=None):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        data = np.load(fn)
        result = matrix_normalization(data, (128, 200))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        return result, label

    def __len__(self):
        return len(self.imgs)


datas = Data_info(path_train=TRAIN_PATH, path_test=TEST_PATH)
all_data = datas.data_train + datas.data_test

positive_loader = DataLoader(MyDataset(datas.preseizure), batch_size=32, shuffle=True)  # 作为训练集
negative_loader = DataLoader(MyDataset(datas.sleep_normal), batch_size=32, shuffle=True)  # 作为测试集

all_loader = MyDataset(all_data)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 44, 68
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16,22, 34
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 12, 18
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 10, 16
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=2, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=3),  # b, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


model = VAE().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_negative(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(negative_loader):
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(datas.sleep_normal),
                       100. * batch_idx / len(negative_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / datas.train_length))
    name = str("./models/model-vae-negative_normalsleep.ckpt")
    torch.save(model.state_dict(), name)
    print("model has been saved!")


def train_positive(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(positive_loader):
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(datas.preseizure),
                       100. * batch_idx / len(positive_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.10f}'.format(
        epoch, train_loss / datas.train_length))
    name = str("./models/model-vae-positive_preseizure.ckpt")
    torch.save(model.state_dict(), name)
    print("model has been saved!")


def train_all_data(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(all_loader):
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                epoch, batch_idx * len(data), len(all_data),
                       100. * batch_idx / len(all_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / datas.train_length))
    name = str("./models/model-vae.ckpt")
    torch.save(model.state_dict(), name)
    print("model has been saved!")


def trans_data(vae_model, data, shape=(128, 200)):
    data_tmp = torch.from_numpy(data)
    data_input = data_tmp.cuda(0)
    data_r = vae_model(data_input)
    recon_batch = data_r.cpu()
    result = recon_batch.detach().numpy()
    result = result.reshape(shape)
    return result


#
# def show_eeg(data):
#     plt.imshow(data)
#     plt.show()


if __name__ == "__main__":
    for epoch in range(1, args.epochs + 1):
        # 1.训练正态编码器
        # train_positive(epoch)
        # 2. 训练负态编码器
        train_negative(epoch)
        # 3.用全部数据训练编码器， 暂未使用
        # train_all_data(epoch)
