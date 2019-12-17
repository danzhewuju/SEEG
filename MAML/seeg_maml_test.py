#!/usr/bin/Python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/16 19:03
# @Author  : Alex
# @Site    : 
# @File    : Seeg_VMAML_test.py
# @Software: PyCharm
import argparse
import os

import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader

from Mamlnet import Seegnet
from meta import Meta
import json

config = json.load(open("../DataProcessing/config/fig.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']
print("patient_test is {}".format(patient_test))


def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    torch.manual_seed(222)  # 为cpu设置种子，为了使结果是确定的
    torch.cuda.manual_seed_all(222)  # 为GPU设置种子，为了使结果是确定的
    np.random.seed(222)

    print(args)

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

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    if os.path.exists(
            str("./models/maml" + str(args.n_way) + "way_" + str(args.k_spt) + "shot_{}.pkl".format(patient_test))):
        path = str("./models/maml" + str(args.n_way) + "way_" + str(args.k_spt) + "shot_{}.pkl".format(patient_test))
        maml.load_state_dict(torch.load(path))
        print("load model :{} successfully".format(path))
    else:
        print("model don't existed!")
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
                        batchsz=100)
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
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                         x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

            result, loss_t = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
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
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=2)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=2)
    argparser.add_argument('--dataset_dir', type=str, help="training data set",
                           default="../data/seeg/zero_data/{}".format(patient_test))

    args = argparser.parse_args()
    main()
