#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 21:00
# @Author  : Alex
# @Site    : 
# @File    : config.py
# @Software: PyCharm

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
import smtplib
from email.mime.text import MIMEText
from email.header import Header


class Pyemail:
    def set_SMTP(self):
        # 第三方 SMTP 服务
        self.mail_host = "smtp.qq.com"  # 设置服务器
        self.mail_user = "danyuhao@qq.com"  # 用户名
        self.mail_pass = "guwoxifmcgribdfj"  # 口令

    def set_sender(self, sender='danyuhao@qq.com'):
        self.sender = sender

    def set_receivers(self, *kwg):
        receivers = [x for x in kwg]
        self.receivers = receivers

    def send_info(self):
        try:
            smtpObj = smtplib.SMTP()
            smtpObj.connect(self.mail_host, 25)  # 25 为 SMTP 端口号
            smtpObj.login(self.mail_user, self.mail_pass)
            smtpObj.sendmail(self.sender, self.receivers, self.message.as_string())
            print("邮件发送成功")
        except smtplib.SMTPException:
            print("Error: 无法发送邮件")

    def __init__(self, tital, message_info):
        self.set_SMTP()
        self.set_sender()
        self.set_receivers("danyuhao@qq.com")
        self.message = MIMEText(message_info, 'plain', 'utf-8')
        self.message['From'] = Header("Lab", 'utf-8')
        self.message['To'] = Header("Alex", 'utf-8')
        self.subject = tital
        self.message['Subject'] = Header(self.subject, 'utf-8')
        self.send_info()


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


def get_label_data(path):  # get data include label
    '''

    :param path:
    :return: {"path":1, "path2":2}
    '''
    class_name = os.listdir(path)
    data_name = []
    data_label = []
    for i, name in enumerate(class_name):
        new_path = os.path.join(path, name)
        data_file = os.listdir(new_path)
        path_file = [os.path.join(new_path, x) for x in data_file]
        data_name += path_file
        data_label += [i] * len(data_file)
    result_data_label = dict(zip(data_name, data_label))
    return result_data_label


class Data_info():
    def __init__(self, *kwg):
        index_name_train = os.listdir(kwg[0])
        # index_name_test = os.listdir(path_test)
        data_dir = []
        # data_test = []
        for index, name in enumerate(index_name_train):
            path = os.path.join(kwg[0], name)
            dir_names = os.listdir(path)
            for n in dir_names:
                full_path = os.path.join(path, n)
                data_dir.append((full_path, index))

        self.data = data_dir
        self.data_length = len(data_dir)


class IndicatorCalculation():  # 包含二分类中各种指标
    '''
    tp, fp
    fn, tn

    '''

    def __init__(self, prediction=None, ground_truth=None):
        if prediction is not None and ground_truth is not None:
            self.prediction = prediction  # [0, 1, 0, 1, 1, 0]
            self.ground_truth = ground_truth  # [0, 1, 0, 0, 1 ]

    @staticmethod
    def __division_detection(number):  # division detection if divisor is zero, the result is zero
        return 0 if number == 0 else number

    def __tp(self):
        TP = 0
        for i in range(len(self.prediction)):
            TP += 1 if self.prediction[i] == 1 and self.ground_truth[i] == 1 else 0
        return TP

    def __fp(self):
        FP = 0
        for i in range(len(self.prediction)):
            FP += 1 if self.prediction[i] == 1 and self.ground_truth[i] == 0 else 0
        return FP

    def __fn(self):
        FN = 0
        for i in range(len(self.prediction)):
            FN += 1 if self.prediction[i] == 0 and self.ground_truth[i] == 1 else 0
        return FN

    def __tn(self):
        TN = 0
        for i in range(len(self.prediction)):
            TN += 1 if self.prediction[i] == 0 and self.ground_truth[i] == 0 else 0
        return TN

    def set_values(self, prediction, ground_truth):
        self.prediction = prediction
        self.ground_truth = ground_truth

    def get_accuracy(self):
        return (self.__tp() + self.__tn()) / (self.__tn() + self.__tp() + self.__fn() + self.__fp())

    def get_recall(self):
        divisor = self.__division_detection(self.__tp() + self.__fn())
        if divisor == 0:
            return 0
        else:
            return self.__tp() / divisor

    def get_precision(self):
        divisor = self.__division_detection(self.__tp() + self.__fp())
        if divisor == 0:
            return 0
        else:
            return self.__tp() / divisor

    def get_f1score(self):
        if (self.get_recall() is None) or (self.get_precision() is None) or (
                (self.get_recall() + self.get_precision()) == 0):
            return 0
        else:
            return (2 * self.__tp()) / (2 * self.__tp() + self.__fn() + self.__fp())

    def get_auc(self):
        if sum(self.ground_truth) == 0 or sum(self.ground_truth) == len(self.ground_truth):  # 防止选择的样本中全为0，或者全为1
            return 0
        else:
            y_predict = np.array(self.prediction)
            y_real = np.array(self.ground_truth)
            auc_score = metrics.roc_auc_score(y_real, y_predict)
        return auc_score


class logger():
    def __init__(self, message, format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                 name="log.txt"):
        logging.basicConfig(level=logging.INFO, filename=name, filemode='a', format=format)
        print(message)
        logging.info(message)
