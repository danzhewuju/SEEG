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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys

sys.path.append('../')
from util.util_file import matrix_normalization
import scipy as sp
import scipy.stats
from VAE.vae import trans_data, VAE
from util.util_file import IndicatorCalculation, LogRecord
import json
from cnn_train_seeg import train_transform

config = json.load(open("../DataProcessing/config/fig.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']
print("patient_test is {}".format(patient_test))

parser = argparse.ArgumentParser(description="CNN parameter setting!")
parser.add_argument('-t', '--time', default=2)  # 每一帧的长度
parser.add_argument('-s', '--sample', default=100)  # 对其进行重采样
parser.add_argument('-train_p', '--train_path', default='../data/seeg/zero_data/{}/train'.format(patient_test))
parser.add_argument('-test_p', '--test_path', default='../data/seeg/zero_data/{}/test'.format(patient_test))
# parser.add_argument('-val_p', '--val_path', default='../visualization_feature/valpatient_data/')
parser.add_argument('-val_p', '--val_path', default='../data/seeg/zero_data/{}/val'.format(patient_test))
parser.add_argument('-m_p', '--model_path', default='./models/cnn_model/model-cnn_{}.ckpt')
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
MODLE_PATH = args.model_path.format(patient_test)
GPU = args.GPU  # 使用哪个GPU
NUM_CLASS = args.class_number  # 分类的个数
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCH = args.epoch

# input = 130*200
x_ = 8
y_ = 12


# 预处理的模型加载
# path = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/VAE/models/model-vae.ckpt"  # 模型所在的位置
# vae_model = VAE().cuda(GPU)
# vae_model.load_state_dict(torch.load(path))
# vae_model.eval()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


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
                data_val.append((full_path, index, n))

        self.val = data_val
        self.val_length = len(data_val)


class MyDataset(Dataset):
    def __init__(self, datas, transform):
        self.datas = datas
        self.transform = transform

    def __getitem__(self, item):
        d_p, label, name_id = self.datas[item]
        data = np.load(d_p)
        if self.transform is not None:
            data = self.transform(data)
        result = matrix_normalization(data, (130, 200))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        # result = trans_data(vae_model, result)
        return result, label, name_id

    def __len__(self):
        return len(self.datas)


def run():
    start_time = time.time()  # 开始时间
    data_info = Data_info(VAL_PATH)
    val_data = MyDataset(data_info.val, transform=train_transform)  # 标准数据集的构造
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN().cuda(GPU)  # 保持和之前的神经网络相同的结构特征?
    model.load_state_dict(torch.load(MODLE_PATH))
    print("Loading model! {}".format(MODLE_PATH))

    total = data_info.val_length
    correct = 0

    # Test the model
    cal = IndicatorCalculation()
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance) 对于单个图片的测试
    total_accuracy = []
    total_precision = []
    total_recall = []
    total_f1_score = []
    total_auc = []
    for i in range(10):

        pre_label = []
        ground_label = []
        scores = np.zeros(0)
        with torch.no_grad():
            for (data, labels, name_id) in val_loader:
                data = data.cuda(GPU)
                labels = labels.cuda(GPU)

                outputs = model(data)  # 直接获得模型的结果
                possible = F.softmax(outputs, dim=1)
                score = possible[:, 1]
                scores = np.append(scores, score.detach().cpu())
                _, predicted = torch.max(outputs.data, 1)
                pre_label += predicted.tolist()
                ground_label += labels.detach().cpu().numpy().tolist()

        cal.set_values(pre_label, ground_label)
        acc_avg = cal.get_accuracy()
        precisions_avg = cal.get_precision()
        recall_avg = cal.get_recall()
        f1score_avg = cal.get_f1score()
        scores = torch.from_numpy(scores)
        ground_label = torch.Tensor(ground_label)
        auc_avg = cal.get_auc(scores, ground_label)

        total_accuracy.append(acc_avg)
        total_precision.append(precisions_avg)
        total_recall.append(recall_avg)
        total_f1_score.append(f1score_avg)
        total_auc.append(auc_avg)

        print('Test Accuracy:{:.5f}, Test Precision:{:.5f}, Test Recall:{:.5f}, Test F1 score:{:.5f}, Test AUC:{:.5f}'.
              format(acc_avg, precisions_avg, recall_avg, f1score_avg, auc_avg))
    average_accuracy, h_a = mean_confidence_interval(total_accuracy)
    average_precision, h_p = mean_confidence_interval(np.array(total_precision))
    average_recall, h_r = mean_confidence_interval(np.array(total_recall))
    average_f1score, h_f = mean_confidence_interval(np.array(total_f1_score))
    average_auc, h_au = mean_confidence_interval(total_auc)

    result = "average accuracy :{}, h:{}\n average precision :{}, h:{}\n average recall :{}, h:{}\n average f1score " \
             ":{}, h:{}\n average AUC:{}, h:{}".format(average_accuracy, h_a, average_precision, h_p, average_recall,
                                  h_r, average_f1score, h_f, average_auc, h_au)
    print(result)
    log = "{}-{}:\n{} ".format(os.path.basename(__file__), patient_test, result)
    LogRecord.write_log(log)
    end_time = time.time()
    run_time = end_time - start_time
    print("Running Time {:.4f}".format(run_time))


def run_test_1():
    true_label = 0
    CNN_batch_size = 1

    class Data_info():
        def __init__(self, path_val, label):
            # pre-seizure: 0  non-seizure:1
            names = os.listdir(path_val)
            full_path = [(os.path.join(path_val, x), label, x) for x in names]
            self.full_path = full_path
            self.data_length = len(full_path)
            self.names = names

    class MyDataset(Dataset):  # 重写dateset的相关类
        def __init__(self, imgs):
            self.imgs = imgs
            # self.transform = transform
            # self.target_transform = target_transform

        def __getitem__(self, index):
            fn, label, name_id = self.imgs[index]
            data = np.load(fn)
            result = matrix_normalization(data, (130, 200))
            result = result.astype('float32')
            result = result[np.newaxis, :]
            return result, label, name_id

        def __len__(self):
            return len(self.imgs)

    data_path = "../data/seeg/zero_data/BDP/val/sleep_normal"
    print("path:{}".format(data_path))
    data_info = Data_info(data_path, true_label)
    my_dataset = MyDataset(data_info.full_path)
    dataloader = DataLoader(my_dataset, batch_size=CNN_batch_size, shuffle=False)

    model = CNN().cuda(0)
    # model.load_state_dict(torch.load(MODLE_PATH))
    model_path = MODLE_PATH
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("CNN model loaded success! {}".format(model_path))
    else:
        print("modle is not exist!")
        exit(0)
    pre_result = {}
    with torch.no_grad():
        for data, label, name_id in dataloader:
            data = data.cuda(0)

            outputs = model(data)
            # print(outputs)
            # c_result = outputs.cpu().detach().numpy()
            # r = softmax(c_result, axis=1)
            # pre_y = r.argmax(1)[0]
            _, predicted = torch.max(outputs.data, 1)
            pre_y = predicted

            pre_result[name_id[0]] = pre_y

    # save_name = save_file_util("precision", "{}-{}.pkl".format(patient_test, "pre-seizure-precision-0"))
    # # save_name = save_file_util("precision", "{}-{}.pkl".format(patient_test, "sleep-precision-1"))
    # with open(save_name, 'wb') as f:
    #     pickle.dump(pre_result, f)
    #     print("save success!")
    sum_length = len(pre_result)
    count = 0
    for name_id, pre in pre_result.items():
        if pre == true_label:
            count += 1
    print("Accuracy: {}".format(count / sum_length))


if __name__ == '__main__':
    run()
    # run_test_1()
