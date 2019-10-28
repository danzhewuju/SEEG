#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/20 8:03
# @Author  : Alex
# @Site    : 
# @File    : Seeg_VMAML_Double_Vae_test.py
# @Software: PyCharm

import argparse
import os
import sys

import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader
from Seeg_VMAML import VAE

sys.path.append('../')
from MAML.Mamlnet import Seegnet
from VMAML.vmeta import *

import json

config = json.load(open("../DataProcessing/config/fig.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']
print("patient_test is {}".format(patient_test))

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=2000)
argparser.add_argument('--n_way', type=int, help='n way', default=2)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
argparser.add_argument('--imgsz', type=int, help='imgsz', default=100)
argparser.add_argument('--imgc', type=int, help='imgc', default=5)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
argparser.add_argument('--dataset_dir', type=str, help="training data set",
                       default="../data/seeg/zero_data/{}".format(patient_test))

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

# 构建两个编码器
vae_p = VAE().to(device)
vae_n = VAE().to(device)
model_vae_p_path = "./models/Vae_negative_{}.pkl".format(patient_test)
model_vae_n_path = "./model/Vae_positive_{}.pkl".format(patient_test)
if os.path.exists(model_vae_n_path) and os.path.exists(model_vae_p_path):
    vae_p.load_state_dict(torch.load(model_vae_p_path))
    vae_n.load_state_dict(torch.load(model_vae_n_path))
    print("loading VAE model successfully!")
else:
    print("VAE model doesn't exist!")


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def trans_data_vae(data, label_data):  # 经过了VAE编码
    shape_data = data.shape
    data_view = data.reshape((-1, resize[0], resize[1]))
    shape_label = label_data.shape
    label_list = label_data.flatten()
    number = shape_label[0] * shape_label[1]
    result = []
    result_p = []
    result_n = []
    for i in range(number):
        data_tmp = data_view[i]
        data_tmp = torch.from_numpy(data_tmp)
        data_tmp = data_tmp.to(device)

        # 1. 经过编码器，并行结构
        recon_batch_p, mu_p, logvar_p = vae_p(data_tmp)
        recon_batch_n, mu_n, logvar_n = vae_n(data_tmp)

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
        result_tmp_n = recon_batch_n.detach().cpu().numpy()
        result_tmp_p = recon_batch_p.detach().cpu().numpy()
        # result_tmp = (result_tmp_p+result_tmp_n)/2
        # result_tmp = result_tmp_p.reshape(resize)
        data_result_p = result_tmp_p.reshape(resize)[np.newaxis, :]
        data_result_n = result_tmp_n.reshape(resize)[np.newaxis, :]
        result_p.append(data_result_p)
        result_n.append(data_result_n)

        # 2.同时经过两个编码器 串行结构
        # recon_batch_p, mu_p, logvar_p = vae_p(data_tmp)
        # recon_batch, _, _ = vae_n(recon_batch_p)
        # recon_batch = recon_batch.detach().cpu().numpy()
        # data_result = recon_batch.reshape(resize)[np.newaxis, :]
        # result.append(data_result)

    result_t_p = np.array(result_p)
    result_t_n = np.array(result_n)
    result_r_p = result_t_p.reshape(shape_data)
    result_r_n = result_t_n.reshape(shape_data)

    # result_t = np.array(result)
    # result_t = result_t.reshape(shape_data)

    return result_r_p, result_r_n
    # return result_t


def main():
    torch.manual_seed(222)  # 为cpu设置种子，为了使结果是确定的
    torch.cuda.manual_seed_all(222)  # 为GPU设置种子，为了使结果是确定的
    np.random.seed(222)

    print(args)
    path = str(
        "./models/{}/maml".format(patient_test) + str(args.n_way) + "way_" + str(args.k_spt) + "shot_{}.pkl".format(
            patient_test))

    if os.path.exists(path):

        maml.load_state_dict(torch.load(path))
        print("load model success")
    else:
        print("model doesn't exist!")
        exit(0)

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
                        batchsz=1000)
    test_accuracy = []
    test_precision = []
    test_recall = []
    test_f1score = []
    test_auc = []
    for epoch in range(10):
        # fetch meta_batchsz num of episode each time
        db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
        accs = []
        precisions = []
        recalls = []
        f1scores = []
        aucs = []

        for x_spt, y_spt, x_qry, y_qry in db_test:
            # 1.未引入VAE模块
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            result, loss_t = maml.finetunning(x_spt, y_spt, x_qry, y_qry)

            # # 2.需要引入VAE编码
            # x_spt_p, x_spt_n = trans_data_vae(x_spt.numpy(), y_spt)
            # x_qry_p, x_qry_n = trans_data_vae(x_qry.numpy(), y_qry)
            # x_spt_p = torch.from_numpy(x_spt_p)
            # x_spt_n = torch.from_numpy(x_spt_n)
            # x_qry_p = torch.from_numpy(x_qry_p)
            # x_qry_n = torch.from_numpy(x_qry_n)
            #
            # x_spt_p, x_spt_n, y_spt, x_qry_p, x_qry_n, y_qry = x_spt_p.squeeze(0).to(device), x_spt_n.squeeze(0).to(
            #     device), y_spt.squeeze(0).to(device), \
            #                                                    x_qry_p.squeeze(0).to(device), x_qry_n.squeeze(0).to(
            #     device), y_qry.squeeze(0).to(device)
            #
            # accs, loss_t = maml.finetunning_double_vae(x_spt_p, x_spt_n, y_spt, x_qry_p, x_qry_n, y_qry)

            # 3. 引入VAE 但是同时经过两个编码器
            # x_spt = trans_data_vae(x_spt.numpy(), y_spt)
            # x_qry = trans_data_vae(x_qry.numpy(), y_qry)
            # x_spt = torch.from_numpy(x_spt)
            # x_qry = torch.from_numpy(x_qry)
            #
            # x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
            #                              x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)
            # accs, loss_t = maml.finetunning(x_spt, y_spt, x_qry, y_qry)

            accs.append(result['accuracy'])
            precisions.append(result['precision'])
            recalls.append(result['recall'])
            f1scores.append(result['f1score'])
            aucs.append(result['auc'])

        # [b, update_step+1]
        acc_avg = np.array(accs).mean()
        precision_avg = np.array(precisions).mean()
        recall_avg = np.array(recalls).mean()
        f1score_avg = np.array(f1scores).mean()
        auc_avg = np.array(aucs).mean()
        print('Test Accuracy:{}, Test Precision:{}, Test Recall:{}, Test F1 score:{}, Test AUC:{}'.
              format(acc_avg, precision_avg, recall_avg, f1score_avg, auc_avg))

        test_accuracy.append(acc_avg)
        test_precision.append(precision_avg)
        test_recall.append(recall_avg)
        test_f1score.append(f1score_avg)
        test_auc.append(auc_avg)
    acc_mean, h = mean_confidence_interval(np.array(test_accuracy))
    pre_mean, h_p = mean_confidence_interval(np.array(test_precision))
    recall_mean, h_r = mean_confidence_interval(np.array(test_recall))
    f1_mean, h_f1 = mean_confidence_interval(np.array(test_f1score))
    auc_mean, h_au = mean_confidence_interval(np.array(test_auc))
    print("average accuracy :{}, h:{}\n average precision :{}, h:{}\n average recall :{}, h:{}"
          "\n average f1score :{}, h:{}\n average AUC :{}, h:{}\n".format(acc_mean, h, pre_mean, h_p, recall_mean, h_r,
                                                                          f1_mean, h_f1, auc_mean, h_au))


if __name__ == '__main__':
    main()
