#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/10/9 18:16
# @Author  : Alex
# @Site    : 
# @File    : Vae_test.py
# @Software: PyCharm

from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append("../")
from util.util_file import matrix_normalization
from tqdm import tqdm

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-train_p', '--train_path', default='../data/seeg/zero_data/train')
parser.add_argument('-test_p', '--test_path', default='../data/seeg/zero_data/test')
parser.add_argument('-val_p', '--val_path', default='../data/seeg/zero_data/val')
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
resize = (130, 200)
device = torch.device("cpu")


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
                if name == "pre_seizure":
                    preseizure.append((full_path, index))
                else:
                    sleep_normal.append((full_path, index))
                data_train.append((full_path, index))

        for index, name in enumerate(index_name_test):
            path = os.path.join(path_test, name)
            dir_names = os.listdir(path)
            for n in dir_names:
                full_path = os.path.join(path, n)
                if name == "pre_seizure":
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
        result = matrix_normalization(data, (130, 200))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        return result, label

    def __len__(self):
        return len(self.imgs)


datas = Data_info(path_train=TRAIN_PATH, path_test=TEST_PATH)
all_data = datas.data_train + datas.data_test
positive_loader = MyDataset(datas.preseizure)  # 作为训练集
Positive_loader = DataLoader(positive_loader, batch_size=args.batch_size,shuffle=True)
# negative_loader = MyDataset(datas.sleep_normal)  # 作为测试集


# all_loader = MyDataset(all_data)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, resize[0] * resize[1]), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return abs(BCE + KLD)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.fc1 = nn.Linear(resize[0] * resize[1], 400)
        self.fc1 = nn.Linear(resize[0] * resize[1], resize[0] * resize[1])
        self.fc12 = nn.Linear(resize[0] * resize[1], resize[0] * resize[1])
        self.fc2 = nn.Linear(resize[0] * resize[1], resize[0] * resize[1])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc12(h1))
        return self.fc1(h1), self.fc12(h1)

    def encode_same_size(self, x):
        h = F.relu(self.fc1(x))
        output = h.reshape(resize[0], resize[1])
        return output

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc2(z))
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, resize[0] * resize[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_positive(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(Positive_loader):
        # data = torch.from_numpy(data)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(datas.preseizure),
                       100. * batch_idx / len(positive_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / datas.train_length))
    name = str("./models/model-vae-positive_preseizure.ckpt")
    torch.save(model.state_dict(), name)
    print("model has been saved!")



if __name__ == '__main__':
    train_positive(10)
