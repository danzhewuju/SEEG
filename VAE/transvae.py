#!/usr/bin/Python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/25 10:09
# @Author  : Alex
# @Site    : 
# @File    : transvae.py
# function transfer data to vae data
# @Software: PyCharm

import torch
import sys

sys.path.append("../")
from util import *
from vae import trans_data, VAE
from tqdm import tqdm
import json

config = json.load(open("../DataProcessing/config/fig.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']
print("patient_test is {}".format(patient_test))

train_path = "../data/seeg/zero_data/{}/train".format(patient_test)
test_path = "../data/seeg/zero_data/{}/test".format(patient_test)
val_path = "../data/seeg/zero_data/{}/val".format(patient_test)

save_train_dir = "../data/seeg/zero_data/{}/train_vae".format(patient_test)
save_test_dir = "../data/seeg/zero_data/{}/test_vae".format(patient_test)
save_val_dir = "../data/seeg/zero_data/{}/val_vae".format(patient_test)


def vae_data(raw_path, save_path):  # train-test dataset and positive/negative
    clean_dir(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    names = os.listdir(raw_path)
    # clean_dir(save_path)
    for n in names:
        if n == "pre_zeizure":
            model_path = "./models/model-vae-positive_preseizure.ckpt"
            save_dir = os.path.join(save_path, n)
        else:
            model_path = "./models/model-vae-negative_normalsleep.ckpt"
            save_dir = os.path.join(save_path, n)
        if os.path.exists(save_dir) is not True:  # if dir don't exist , create it
            os.mkdir(save_dir)
        path_new = os.path.join(raw_path, n)
        file_names = os.listdir(path_new)

        vae_model = VAE().cuda(0)  # lode model
        vae_model.load_state_dict(torch.load(model_path))
        vae_model.eval()
        print("{} has been loaded!".format(model_path))
        for p in file_names:
            name_tmp = p.split("/")[-1]
            save_data_path = os.path.join(save_dir, name_tmp)  # saving data path
            data_path = os.path.join(path_new, p)
            data = np.load(data_path)
            result = matrix_normalization(data, (130, 200))
            result = result.astype('float32')
            result = result[np.newaxis, np.newaxis, :]
            result = trans_data(vae_model, result)
            save_numpy_info(result, save_data_path)
        print("all files:{} had been saved!".format(len(file_names)))


def vae_data_val(raw_path, save_path):  # val dataset
    clean_dir(save_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    names = os.listdir(raw_path)
    model_path_all = "./models/model-vae.ckpt"
    # model_path_p = "./models/model-vae-positive_preseizure.ckpt"
    # model_path_n = "./models/model-vae-negative_normalsleep.ckpt"
    for n in tqdm(names):
        save_dir = os.path.join(save_path, n)
        if os.path.exists(save_dir) is not True:  # if dir don't exist , create it
            os.mkdir(save_dir)
        path_new = os.path.join(raw_path, n)
        file_names = os.listdir(path_new)

        # vae_model_p = VAE().cuda(0)  # lode model
        # vae_model_p.load_state_dict(torch.load(model_path_p))
        # vae_model_p.eval()
        #
        # vae_model_n = VAE().cuda(0)  # lode model
        # vae_model_n.load_state_dict(torch.load(model_path_n))
        # vae_model_n.eval()

        vae_model_all = VAE().cuda(0)  # lode model
        vae_model_all.load_state_dict(torch.load(model_path_all))
        vae_model_all.eval()

        print("{} has been loaded!".format(model_path_all))
        for p in file_names:
            name_tmp = p.split("/")[-1]
            save_data_path = os.path.join(save_dir, name_tmp)  # saving data path
            data_path = os.path.join(path_new, p)
            data = np.load(data_path)
            result = matrix_normalization(data, (130, 200))
            result = result.astype('float32')
            result = result[np.newaxis, :]
            result = trans_data(vae_model_all, result)
            save_numpy_info(result, save_data_path)
        print("all files:{} had been saved!".format(len(file_names)))


if __name__ == '__main__':
    # clean_dir(save_val_dir)
    # 1. 训练集的vae编码
    vae_data(train_path, save_train_dir)  # positive/negative
    # # 2. 测试集的vae编码
    vae_data(test_path, save_test_dir)  # positive/negative
    # # 3. 验证集的vae编码
    # vae_data_val(val_path, save_val_dir)  # validation dataset
