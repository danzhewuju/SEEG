# code is based on https://github.com/katerakelly/pytorch-maml
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
import random
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import Sampler
import time
import sys

sys.path.append('../')
from util.util_file import *

import json

config = json.load(open("../DataProcessing/config/fig.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']
print("patient_test is {}".format(patient_test))


def imshow(img):
    npimg = img.numpy()
    plt.axis("off")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# class Rotate(object):
#     def __init__(self, angle):
#         self.angle = angle
#
#     def __call__(self, x, mode="reflect"):
#         x = x.rotate(self.angle)
#         return x


def mini_imagenet_folders(model_name):
    train_folder = '../data/seeg/{}/{}/train'.format(model_name, patient_test)
    test_folder = '../data/seeg/{}/{}/val'.format(model_name, patient_test)

    metatrain_folders = [os.path.join(train_folder, label)
                         for label in os.listdir(train_folder)
                         if os.path.isdir(os.path.join(train_folder, label))
                         ]
    metatest_folders = [os.path.join(test_folder, label)
                        for label in os.listdir(test_folder)
                        if os.path.isdir(os.path.join(test_folder, label))
                        ]
    random.seed(1)
    random.shuffle(metatrain_folders)
    random.shuffle(metatest_folders)

    return metatrain_folders, metatest_folders


class SeegnetTask(object):

    def __init__(self, character_folders, num_classes, train_num, test_num):
        self.character_folders = character_folders
        self.num_classes = num_classes
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

        t = time.time()
        random.seed(t)
        random.shuffle(self.test_labels)
        random.seed(t)
        random.shuffle(self.test_roots)

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])


class FewShotDataset(Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        self.transform = transform  # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.split = split
        self.image_roots = self.task.train_roots if self.split == 'train' else self.task.test_roots
        self.labels = self.task.train_labels if self.split == 'train' else self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Seegnet(FewShotDataset):

    def __init__(self, *args, **kwargs):
        super(Seegnet, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = np.load(image_root)
        result = matrix_normalization(image, (130, 200))
        result = result.astype('float32')
        result = torch.from_numpy(result)
        result = result[np.newaxis, :]
        # image = image.convert('RGB')
        # if self.transform is not None:
        #     image = self.transform(image)
        label = self.labels[idx]
        # if self.target_transform is not None:
        #     label = self.target_transform(label)
        return result, label


class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' heatmap each from 'num_cl' pools
        of heatmap of size 'num_per_class' '''

    def __init__(self, num_cl, num_inst, shuffle=True):

        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batches = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
        else:
            batches = [[i + j * self.num_inst for i in range(self.num_inst)] for j in range(self.num_cl)]
        batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]

        if self.shuffle:
            random.shuffle(batches)
            for sublist in batches:
                random.shuffle(sublist)
        batches = [item for sublist in batches for item in sublist]
        return iter(batches)

    def __len__(self):
        return 1


class ClassBalancedSamplerOld(Sampler):
    ''' Samples 'num_inst' heatmap each from 'num_cl' pools
        of heatmap of size 'num_per_class' '''

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1


def get_mini_imagenet_data_loader(task, num_per_class=1, split='train', shuffle=False):
    # normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])
    dataset = Seegnet(task, split=split)
    # dataset = Seegnet(task, split=split, transform=transforms.Compose([transforms.ToTensor(), normalize]))
    if split == 'train':
        sampler = ClassBalancedSamplerOld(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)

    else:
        sampler = ClassBalancedSampler(task.num_classes, task.test_num, shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)
    return loader
