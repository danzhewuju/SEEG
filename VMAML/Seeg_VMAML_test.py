#!/usr/bin/Python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/16 19:03
# @Author  : Alex
# @Site    : 
# @File    : Seeg_VMAML_test.py
# @Software: PyCharm
import argparse
import os
import sys

import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader
from VMAML.Seeg_VMAML import VAE
from task_generator_test import SeegnetTask

sys.path.append('../')
from MAML.Mamlnet import Seegnet
from VMAML.meta import *

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=1000)
argparser.add_argument('--n_way', type=int, help='n way', default=2)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=8)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
argparser.add_argument('--imgsz', type=int, help='imgsz', default=100)
argparser.add_argument('--imgc', type=int, help='imgc', default=5)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
argparser.add_argument('--dataset_dir', type=str, help="training data set", default="../data/seeg/zero_data")

args = argparser.parse_args()

config = [
    ('conv2d', [32, 1, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 2, 0]),
    ('conv2d', [32, 32, 3, 3, 1, 0]),
    ('relu', [True]),
    ('bn', [32]),
    ('max_pool2d', [2, 1, 0]),
    ('flatten', []),
    ('linear', [args.n_way, 7040])
]

resize = (130, 200)
device = torch.device('cuda')
Vae = VAE().to(device)
maml = Meta(args, config).to(device)


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def trans_data_vae(data, label_data):
    shape_data = data.shape
    data_view = data.reshape((-1, resize[0], resize[1]))
    shape_label = label_data.shape
    label_list = label_data.flatten()
    number = shape_label[0] * shape_label[1]
    result = []
    for i in range(number):
        data_tmp = data_view[i]
        data_tmp = torch.from_numpy(data_tmp)
        data_tmp = data_tmp.to(device)
        recon_batch, mu, logvar = Vae(data_tmp)

        # if label_list[i] == 1:  # positive
        #     optimizer_vae_p.zero_grad()
        #     recon_batch, mu, logvar = vae_p(data_tmp)
        #     loss = loss_function(recon_batch, data_tmp, mu, logvar)
        #     loss.backward()
        #     optimizer_vae_p.step()
        # else:
        #     optimizer_vae_n.zero_grad()
        #     recon_batch, mu, logvar = vae_n(data_tmp)
        #     loss = loss_function(recon_batch, data_tmp, mu, logvar)
        #     loss.backward()
        #     optimizer_vae_n.step()
        result_tmp = recon_batch.detach().cpu().numpy()
        result_tmp = result_tmp.reshape(resize)
        data_result = result_tmp[np.newaxis, :]
        result.append(data_result)
    result_t = np.array(result)
    result_r = result_t.reshape(shape_data)
    return result_r


def main():
    torch.manual_seed(222)  # 为cpu设置种子，为了使结果是确定的
    torch.cuda.manual_seed_all(222)  # 为GPU设置种子，为了使结果是确定的
    np.random.seed(222)

    print(args)

    if os.path.exists(
            "./models/" + str("./models/maml" + str(args.n_way) + "way_" + str(args.k_spt) + "shot.pkl")):
        path = "./models/" + str("./models/maml" + str(args.n_way) + "way_" + str(args.k_spt) + "shot.pkl")
        maml.load_state_dict(path)
        print("load model success")

    if os.path.exists("./models/Vae.pkl"):
        path_vae = str("./models/Vae.pkl")
        Vae.load_state_dict(torch.load(path_vae))
        print("loading VAE model successfully!")

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    # mini = Seegnet(args.dataset_dir, mode='train_vae', n_way=args.n_way, k_shot=args.k_spt,
    #                     k_query=args.k_qry,
    #                     batchsz=1000)
    mini_test = Seegnet(args.dataset_dir, mode='val', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=50)
    test_accuracy = []
    for epoch in range(10):
        # fetch meta_batchsz num of episode each time
        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
        accs_all_test = []

        for x_spt, y_spt, x_qry, y_qry in db_test:
            x_spt = trans_data_vae(x_spt.numpy(), y_spt)
            x_qry = trans_data_vae(x_qry.numpy(), y_qry)
            x_spt = torch.from_numpy(x_spt)
            x_qry = torch.from_numpy(x_qry)
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            accs, loss_t = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
            accs_all_test.append(accs)

        # [b, update_step+1]
        accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
        print('Test acc:', accs)
        test_accuracy.append(accs[-1])
    acc_mean, h = mean_confidence_interval(np.array(test_accuracy))
    print("average accuracy :{}, h:{}".format(acc_mean, h))


if __name__ == '__main__':
    main()
