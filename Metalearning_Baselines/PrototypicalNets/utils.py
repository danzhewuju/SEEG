import os
import shutil
import time
import pprint
from torch.utils.data import Dataset
import torch
import random
import numpy as np
import scipy as sp
import scipy.stats
import sklearn.metrics as metrics
import logging
import sys

sys.path.append("../../")
from util.util_file import IndicatorCalculation


class MyDataset(Dataset):  # 重写dateset的相关类
    def __init__(self, imgs, transform=None, target_transform=None):
        self.imgs = imgs
        data, labels = zip(*imgs)
        self.data = list(data)
        self.labels = list(labels)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        data = np.load(fn)
        result = matrix_normalization(data, (130, 200))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        # result = trans_data(vae_model, result)
        return result, label

    def __len__(self):
        return len(self.imgs)


def matrix_normalization(data, resize_shape=(130, 200)):
    '''
    矩阵的归一化，主要是讲不通形状的矩阵变换为特定形状的矩阵, 矩阵的归一化主要是更改序列
    也就是主要更改行
    eg:(188, 200)->(130, 200)   归一化的表示
    :param data:
    :param resize_shape:
    :return:
    '''
    data_shape = data.shape  # 这个必须要求的是numpy的文件格式
    if data_shape[0] != resize_shape[0]:
        if resize_shape[0] > data_shape[0]:  # 做插入处理
            '''
            扩大原来的矩阵
            '''
            d = resize_shape[0] - data_shape[0]
            channels_add = random.sample(range(1, data_shape[0] - 1), d)
            fake_channel = []  # 添加信道列表的值
            for c in channels_add:
                tmp = (data[c - 1] + data[c]) * 1.0 / 2
                fake_channel.append(tmp)
            data = np.insert(data, channels_add, fake_channel, axis=0)
        else:
            if resize_shape[0] < data_shape[0]:  # 做删除处理
                '''
                删除掉原来的矩阵
                '''
                d = data_shape[0] - resize_shape[0]
                channels_del = random.sample(range(1, data_shape[0] - 1), d)
                data = np.delete(data, channels_del, axis=0)
    return data


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


class Data_info():
    def __init__(self, path_train, path_test):
        index_name_train = os.listdir(path_train)
        index_name_test = os.listdir(path_test)
        data_train = []
        data_test = []
        for index, name in enumerate(index_name_train):
            path = os.path.join(path_train, name)
            dir_names = os.listdir(path)
            for n in dir_names:
                full_path = os.path.join(path, n)
                data_train.append((full_path, index))

        for index, name in enumerate(index_name_test):
            path = os.path.join(path_test, name)
            dir_names = os.listdir(path)
            for n in dir_names:
                full_path = os.path.join(path, n)
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


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label) ** 2).sum() / len(pred) / 2


class logger():
    def __init__(self, message, format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                 name="log.txt"):
        logging.basicConfig(level=logging.INFO, filename=name, filemode='a', format=format)
        print(message)
        logging.info(message)
