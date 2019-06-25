#!/usr/bin/python
import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from util.util_file import *

'''
一般CNN模型的训练
'''

parser = argparse.ArgumentParser(description="CNN parameter setting!")
parser.add_argument('-t', '--time', default=2)  # 每一帧的长度
parser.add_argument('-s', '--sample', default=100)  # 对其进行重采样
parser.add_argument('-train_p', '--train_path', default='../data/seeg/train')
parser.add_argument('-test_p', '--test_path', default='../data/seeg/test')
parser.add_argument('-val_p', '--val_path', default='../data/seeg/val')
parser.add_argument('-g', '--GPU', type=int, default=0)
parser.add_argument('-n', '--class_number', type=int, default=2)
parser.add_argument('-b', '--batch_size', type=int, default=32)
parser.add_argument('-l', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epoch', type=int, default=10)
args = parser.parse_args()

# hyper parameter setting

TEST_PATH = args.test_path
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
GPU = args.GPU  # 使用哪个GPU
NUM_CLASS = args.class_number  # 分类的个数
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.learning_rate
NUM_EPOCH = args.epoch

# input = 130*200
x_ = 8
y_ = 12


def show_plt(accuracy, loss):  # 画出accuracy,loss的趋势图
    path = './drawing/'
    time_stamp = time.time()
    time_struct = time.localtime(time_stamp)
    time_stamp = "Accuracy-Loss%d-%d-%d %d:%d:%d" % (time_struct[0], time_struct[1], time_struct[2],
                                                     time_struct[3], time_struct[4], time_struct[5])
    name = path + time_stamp + ".png"
    acc = np.asarray(accuracy)
    loss = np.asarray(loss)
    plt.title("Accuracy&&Loss")
    plt.xlabel("epoch")
    plt.ylabel("Acc/loss")
    plt.plot(acc, label="Accuracy")
    plt.plot(loss, label="Loss")
    plt.legend(loc='upper right')
    plt.savefig(name)
    plt.show()

    print("Saved Image!")
    return True


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


class MyDataset(Dataset):  # 重写dateset的相关类
    def __init__(self, imgs, transform=None, target_transform=None):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        data = np.load(fn)
        data = matrix_normalization(data, (130, 200))
        data = data.astype('float32')
        data = data[np.newaxis, :]
        return data, label

    def __len__(self):
        return len(self.imgs)


def run():
    start_time = time.time()
    datas = Data_info(path_train=TRAIN_PATH, path_test=TEST_PATH)
    train_data = MyDataset(datas.data_train)  # 作为训练集
    test_data = MyDataset(datas.data_test)  # 作为测试集
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    model = CNN().cuda(GPU)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train the model

    total_step = len(train_loader)
    Acc_h = []  # 用于画图的数据记录
    Loss_h = []  # 用于画图的数据记录
    correct_h = []
    loss_h = []
    for epoch in range(NUM_EPOCH):
        for i, (images, labels) in enumerate(train_loader):

            images = images.cuda(GPU)
            labels = labels.cuda(GPU)

            # images = images
            # labels = labels

            # Forward pass
            outputs = model(images).cuda(GPU)
            loss = criterion(outputs, labels).cuda(GPU)
            _, prediction = torch.max(outputs.data, 1)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct = (prediction == labels).sum().item()
            correct_rate = correct / BATCH_SIZE

            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'
                      .format(epoch + 1, NUM_EPOCH, i + 1, total_step, loss.item(), correct_rate))
            correct_h.append(correct_rate)
            loss_h.append(loss.item())
        Acc_h.append(np.mean(np.asarray(correct_h)))
        Loss_h.append(np.mean(np.asarray(loss_h)))
        correct_h.clear()
        loss_h.clear()
    show_plt(Acc_h, Loss_h)

    # Test the model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.cuda(GPU)
        labels = labels.cuda(GPU)
        # images = images
        # labels = labels
        outputs = model(images)  # 直接获得模型的结果
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the {} test seegs: {} %'.format(datas.test_length,
                                                                         100 * correct / total))
    Acc = correct / total

    # Save the model checkpoint
    timestamp = str(int(time.time()))
    name = str("./models/model-cnn.ckpt")
    torch.save(model.state_dict(), name)
    end_time = time.time()
    run_time = end_time - start_time
    print("Running Time {:.4f}".format(run_time))


if __name__ == '__main__':
    run()
