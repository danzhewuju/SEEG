#!/usr/bin/python
# ---------------------------------
# 文件工具方法， 主要包含常见的文件处理方法
#  以及常见的文件存储的方法
# ---------------------------------
import glob
import os
import random

import cv2
import numpy as np
from sklearn import metrics
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from dtw import dtw
import math
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import wasserstein_distance
import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return math.sinh(x) / math.cosh(x)


def get_all_file_path(path, suffix='fif'):  # 主要是获取某文件夹下面所有的文件列表
    '''
    :param path: 存储对应文件的路径
    :return: 文件夹下面对应的文件路径
    '''
    names = []
    file_map = {}
    path_dir = []
    dir = os.listdir(path)
    names = dir
    for d in dir:
        path_dir.append(os.path.join(path, d))

    for index, p in enumerate(path_dir):
        new_suffix = '*.' + suffix
        file_p = glob.glob(os.path.join(p, new_suffix))
        file_map[names[index]] = file_p
    return file_map


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


def matrix_normalization_recorder(data, resize_shape=(130, 200)):  # 这个是矩阵初始化的记录版
    '''
        矩阵的归一化，主要是讲不通形状的矩阵变换为特定形状的矩阵, 矩阵的归一化主要是更改序列
        也就是主要更改行
        eg:(188, 200)->(130, 200)   归一化的表示
        :param data:
        :param resize_shape:
        :return:
        '''
    recoder = []  # [-1/1/0, 10, 11, 12, 14]  第一位为标识位， -1 标识删除了的信道， +1 标识增加了的信道，0, 表示对信道没有变化 后面对应的是增加或者删除的信道
    recoder.append(0)
    data_shape = data.shape  # 这个必须要求的是numpy的文件格式
    if data_shape[0] != resize_shape[0]:
        if resize_shape[0] > data_shape[0]:  # 做插入处理
            '''
            扩大原来的矩阵
            '''
            d = resize_shape[0] - data_shape[0]
            channels_add = random.sample(range(1, data_shape[0] - 1), d)
            recoder[0] = 1
            recoder += channels_add
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
                recoder[0] = -1
                recoder += channels_del
                data = np.delete(data, channels_del, axis=0)
    return data, recoder


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


def get_first_dir_path(path, suffix="jpg"):
    paths = glob.glob(os.path.join(path, '*.' + suffix))
    return paths


def get_matrix_max_location(mtx_data, k, reverse=True):
    '''

    :param mtx_data: 矩阵的数据
    :param k: 获取前K 个最大、最小
    :param reverse: True : 最大，逆序 False: 最小，否则正序
    :return:[(0, 0), (2, 1), (2, 2), (1, 1), (1, 2)] 结果是按照顺序进行排序
    '''
    d_f = mtx_data.flatten()
    if reverse:
        index_id = d_f.argsort()[-k:]
    else:
        index_id = d_f.argsort()[k:]
    x_index, y_index = np.unravel_index(index_id, mtx_data.shape)
    location = list(zip(x_index, y_index))  # 此时只是选取了最大几个，数据之间是没有顺序的
    location_dic = {}
    for x, y in location:
        location_dic[(x, y)] = mtx_data[x][y]

    location_dic = sorted(location_dic.items(), key=lambda x: -x[1]) if reverse else sorted(location_dic.items(),
                                                                                            key=lambda x: x[1])
    result = [x[0] for x in location_dic]
    return result


def mtx_similarity(mtx_a, mtx_b):
    '''

    :param mtx_a:
    :param mtx_b:
    :return:  计算矩阵的相似度
    '''
    mtx_1 = mtx_a.flatten()  # 展开为一维
    mtx_2 = mtx_b.flatten()
    min_length = min(len(mtx_1), len(mtx_2))
    mtx_1 = mtx_1[:min_length]
    mtx_2 = mtx_2[:min_length]
    result = np.dot(mtx_1, mtx_2.T) / (np.linalg.norm(mtx_1) * np.linalg.norm(mtx_2))
    result = abs(result)
    return result


def clean_dir(path):
    print("this is danger operation!")
    clean_path = os.path.join(path, "*")
    print("you will clean all files in {}, do you continued?(y/n)".format(clean_path))
    key = str(input())
    if key == "y" or key == "Y":
        print("cleaning the files in {}".format(clean_path))
        os.system("rm -r {}".format(clean_path))
        print("clean finished!")
    else:
        print("cancel operation !")
        exit(0)


def trans_numpy_cv2(data):
    data1 = sigmoid(data)
    min_data = np.min(data1)
    data1 = data1 - min_data
    max_data = np.max(data1)
    data1 = data1 / max_data * 255
    result = data1.astype(np.uint8)
    result = cv2.merge([result])  # 将信道进行整合
    return result


def time_add(h, m, s, seconds_add):
    '''

    :param h: 小时
    :param m: 分钟
    :param s: 秒
    :param seconds_add: 需要增加的时间秒数
    :return: 返回的是绝对的时间
    '''
    s += seconds_add
    s_m = s // 60
    s = s % 60

    m += s_m
    m_h = m // 60
    m = m % 60

    h += m_h
    h %= 24

    return int(h), int(m), s


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

    def get_auc(self, y_pre=None, y_real=None):
        # if type(self.prediction) is not np.ndarray:
        #     self.prediction = np.asarray(self.prediction)
        #     self.ground_truth = np.asarray(self.ground_truth)
        if y_real is  None and y_pre is  None:
            y_predict = self.prediction.cpu()
            y_real = self.ground_truth.cpu()
        else:
            y_predict = y_pre.cpu()
            y_real = y_real.cpu()
        auc_score = metrics.roc_auc_score(y_real, y_predict)
        return auc_score


def dir_create_check(path_dir):
    if os.path.exists(path_dir) is False:
        os.makedirs(path_dir)
        print("{} has been created!".format(path_dir))
    else:
        print("{} has existed!".format(path_dir))


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


def similarity_dtw(s1, s2):
    '''

    :param s1:  序列1
    :param s2:  序列2
    :return:
    '''
    ratio = 10  # 设定的放缩系数，避免数据的相似度过于集中
    euclidean_norm = lambda x, y: np.abs(ratio * (x - y))
    d, cost_matrix, acc_cost_matrix, path = dtw(s1, s2, dist=euclidean_norm)
    score = 1 - tanh(d)  # 相似度的评分【0,1】 0： 完全不同， 1： 完全相同
    return score


def similarity_EMD(s1, s2):
    k = 1e3
    score = 1 - np.tanh(k * wasserstein_distance(s1, s2))
    return score


def fft_function(data):
    '''
    傅里叶的频谱分析
    :param data:
    :return:
    '''
    fft_y = fft(data)
    N = len(data)
    x = range(int(N / 2))
    y_ = np.abs(fft_y) * 2 / N
    y = y_[range(int(N / 2))]
    plt.figure()
    plt.plot(x, y, 'r')
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    name = "Frequency"
    plt.title(name)
    plt.show()


def histogram_spectrum(data, file_pass=10):
    frequency = []
    for i, d in enumerate(data):
        fft_y = fft(d)
        N = len(d)
        x = range(int(N / 2))
        y_ = np.abs(fft_y) * 2 / N
        y = y_[range(int(N / 2))]
        index = np.argmax(y)
        frequency.append(index)
    print(frequency)
    plt.figure()
    frequency = list(filter(lambda x: x <= 10, frequency))
    count = len(dict(Counter(frequency)))
    plt.hist(frequency, density=True, bins=count)
    plt.xlabel("Frequency")
    plt.ylabel("Density")
    plt.title("Histogram of the spectrum")
    plt.show()


class LogRecord:

    @staticmethod
    def write_log(log_txt, log_path="/home/cbd109-3/Users/data/yh/Program/Python/SEEG/log/Running_result.txt"):
        base_dir = os.path.dirname(log_path)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not os.path.exists(log_path):
            f = open(log_path, 'w')
        else:
            f = open(log_path, 'a')
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        result = "{}\n{}\n".format(time_now, log_txt)
        f.write(result)
        f.close()


def test_list():
    a = [1, 2, 3]
    b = [4, 5, 6]
    print(a + b)


if __name__ == '__main__':
    test_list()
