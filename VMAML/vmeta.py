from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import os

from VMAML.vlearner import Learner
from util.util_file import IndicatorCalculation
import json
import pickle

config = json.load(open("../DataProcessing/config/fig.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']


class Meta(nn.Module):
    """
    Meta Learner
    """

    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.net = Learner(config)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm / counter

    def forward(self, x_spt, y_spt, x_qry, y_qry, flag):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        for i in range(task_num):

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x_spt[i], vars=None, bn_training=True)
            loss = F.cross_entropy(logits, y_spt[i])
            grad = torch.autograd.grad(loss, self.net.parameters())
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

                logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        if flag:  # 设置交替训练的条件格式
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs, loss_q

    def finetunning_double_vae(self, x_spt_p, x_spt_n, y_spt, x_qry_p, x_qry_n, y_qry):
        """

        :param x_spt_p: 经过VAE_p编码器
        :param x_spt_n: 经过VAE_n编码器
        :param y_spt:
        :param x_qry_p:
        :param x_qry_n:
        :param y_qry:
        :return:
        验证集经过双VAE
        """
        assert len(x_spt_p.shape) == 4  # 用来检查数据类型的断言
        assert len(x_spt_n.shape) == 4

        querysz = x_qry_p.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits_p = net(x_spt_p)
        lohits_n = net(x_spt_n)
        loss = min(F.cross_entropy(logits_p, y_spt), F.cross_entropy(lohits_n, y_spt))
        grad = torch.autograd.grad(loss, net.parameters())  # 自动求导机制
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q_p = net(x_qry_p, net.parameters(), bn_training=True)
            logits_q_n = net(x_qry_n, net.parameters(), bn_training=True)

            # [setsz]
            pred_q_p = F.softmax(logits_q_p, dim=1)
            pred_q_n = F.softmax(logits_q_n, dim=1)
            pred_q = (pred_q_n + pred_q_p) / 2
            pred_q = pred_q.argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q_p = net(x_qry_p, net.parameters(), bn_training=True)
            logits_q_n = net(x_qry_n, net.parameters(), bn_training=True)

            # [setsz]
            pred_q_p = F.softmax(logits_q_p, dim=1)
            pred_q_n = F.softmax(logits_q_n, dim=1)
            pred_q = (pred_q_n + pred_q_p) / 2
            pred_q = pred_q.argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct
        loss_all = 0
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits_q_p = net(x_qry_p, fast_weights, bn_training=True)
            logits_q_n = net(x_qry_n, fast_weights, bn_training=True)

            loss = min(F.cross_entropy(logits_q_n, y_spt), F.cross_entropy(logits_q_p, y_spt))
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q_p = net(x_qry_p, fast_weights, bn_training=True)
            logits_q_n = net(x_qry_n, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = min(F.cross_entropy(logits_q_n, y_spt), F.cross_entropy(logits_q_p, y_spt))
            loss_all += loss_q.cpu().detach().numpy()

            with torch.no_grad():
                logits_q = (logits_q_n + logits_q_p) / 2
                # pred_q_p = F.softmax(logits_q_p, dim=1)
                # pred_q_n = F.softmax(logits_q_n, dim=1)
                # pred_q = (pred_q_n + pred_q_p) / 2
                pred_q = F.softmax(logits_q, dim=1)
                pred_q = pred_q.argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net
        loss_all /= self.update_step_test - 1

        accs = np.array(corrects) / querysz

        return accs, loss_all

    def finetunning(self, x_spt, y_spt, x_qry, y_qry, query_y_id_list=None):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4  # 用来检查数据类型的断言

        querysz = x_qry.size(0)

        corrects = [0 for _ in range(self.update_step_test + 1)]
        precisions = [0 for _ in range(self.update_step_test + 1)]  # precision
        recalls = [0 for _ in range(self.update_step_test + 1)]  # recalls
        f1scores = [0 for _ in range(self.update_step_test + 1)]  # F_1 score
        auc = [0 for _ in range(self.update_step_test + 1)]
        cal = IndicatorCalculation()

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        # net = deepcopy(self.net)
        net = self.net

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            possible = F.softmax(logits_q, dim=1)
            scores = possible[:, 1]
            # scalar
            # scalar
            # correct = torch.eq(pred_q, y_qry).sum().item()
            cal.set_values(pred_q, y_qry)
            corrects[0] = cal.get_accuracy()  # 指标构建
            precisions[0] = cal.get_precision()
            recalls[0] = cal.get_recall()
            f1scores[0] = cal.get_f1score()
            auc[0] = cal.get_auc(scores, y_qry)

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            possible = F.softmax(logits_q, dim=1)
            scores = possible[:, 1]
            # scalar
            # correct = torch.eq(pred_q, y_qry).sum().item()
            cal.set_values(pred_q, y_qry)
            corrects[1] = cal.get_accuracy()
            precisions[1] = cal.get_precision()
            recalls[1] = cal.get_recall()
            f1scores[1] = cal.get_f1score()
            auc[1] = cal.get_auc(scores, y_qry)

        loss_all = 0
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)
            loss_all += loss_q.cpu().detach().numpy()

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                if query_y_id_list is not None and k == self.update_step_test - 1:  # 需要记录最后的预测结果
                    prediction_query = pred_q.detach().cpu().numpy().tolist()
                # correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                possible = F.softmax(logits_q, dim=1)
                scores = possible[:, 1]
                cal.set_values(pred_q, y_qry)
                corrects[k + 1] = cal.get_accuracy()
                precisions[k + 1] = cal.get_precision()
                recalls[k + 1] = cal.get_recall()
                f1scores[k + 1] = cal.get_f1score()
                auc[k + 1] = cal.get_auc(scores, y_qry)

        del net

        # 将预测的结果进行统计
        if query_y_id_list is not None:
            r_path = "./precision/{}_val_prediction.pkl".format(patient_test)
            # 文件存在需要被创建
            record = {} if not os.path.exists(r_path) else np.load(r_path, allow_pickle=True)
            # record = np.load(r_path, allow_pickle=True)
            ground_truth = y_qry.cpu().tolist()
            for index, id in enumerate(query_y_id_list):
                if id not in record.keys():
                    label = prediction_query[index]
                    res = {'ground truth': ground_truth[index], 'prediction': label}
                    record[id] = res

            with open(r_path, 'wb') as f:
                pickle.dump(record, f)

        # index = len(corrects) - 1 # 选取最优的那个结果
        index = corrects.index(max(corrects))  # 选取准确率最高的那个结果
        loss_all /= self.update_step_test - 1
        result = {"accuracy": corrects[index],
                  "precision": precisions[index],
                  "recall": recalls[index],
                  "f1score": f1scores[index],
                  "auc": auc[index],
                  }
        # accs = np.array(corrects) / querysz

        return result, loss_all


def main():
    pass


if __name__ == '__main__':
    main()
