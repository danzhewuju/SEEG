# ---------------------
# 数据的预处理的一些方法
# ---------------------

import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
import numpy as np


def mini_data_folders():
    train_folder = '../data/seeg/train'
    test_folder = '../data/seeg/test'
    metatrain_folders = [os.path.join(train_folder, label)
                         for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))]

    metatest_folders = [os.path.join(test_folder, label)
                         for label in os.listdir(test_folder)
                         if os.path.isdir(os.path.join(test_folder, label))]
    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders, metatrain_folders


class MiniDataTask(object):
    def __init__(self, character_folders, num_class, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_class
        self.train_num = train_num
        self.test_num = test_num

        class_folders = random.sample(self.character_folders, self.num_classes)
        labels = np.array(range(len(class_folders)))
        labels = dict(zip(class_folders, labels))
        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in class_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num + test_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

