#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/9/20 7:41
# @Author  : Alex
# @Site    : 
# @File    : Seeg_VMAML_Double_Vae.py
# @Software: PyCharm
# function 两个VAE来实现的训练

from __future__ import print_function

import argparse
import sys

import torch.utils.data
from torch.utils.data import DataLoader

sys.path.append('../')

from MAML.Mamlnet import *
from VMAML.vmeta import *
from util.util_file import matrix_normalization
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import json

config = json.load(open("../DataProcessing/config/fig.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']

argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=4000)
argparser.add_argument('--n_way', type=int, help='n way', default=2)
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=5)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
argparser.add_argument('--imgsz', type=int, help='imgsz', default=100)
argparser.add_argument('--imgc', type=int, help='imgc', default=5)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.001)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=8)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
argparser.add_argument('--dataset_dir', type=str, help="training data set",
                       default="../data/seeg/zero_data/{}".format(patient_test))
argparser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
argparser.add_argument('-train_p', '--train_path', default='../data/seeg/zero_data/{}/train'.format(patient_test))
argparser.add_argument('-test_p', '--test_path', default='../data/seeg/zero_data/{}/test'.format(patient_test))
argparser.add_argument('-val_p', '--val_path', default='../data/seeg/zero_data/{}/val'.format(patient_test))

args = argparser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
TEST_PATH = args.test_path
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

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
    ('linear', [args.n_way, 7040])]

resize = (130, 200)


# 数据的输入输出

class Data_info():
    def __init__(self, path_train, path_test):
        index_name_train = os.listdir(path_train)
        index_name_test = os.listdir(path_test)
        data_train = []
        data_test = []
        preseizure = []
        sleep_normal = []
        for index, name in enumerate(index_name_train):
            path = os.path.join(path_train, name)
            dir_names = os.listdir(path)
            for n in dir_names:
                full_path = os.path.join(path, n)
                if name == "pre_zeizure":
                    preseizure.append((full_path, index))
                else:
                    sleep_normal.append((full_path, index))
                data_train.append((full_path, index))

        for index, name in enumerate(index_name_test):
            path = os.path.join(path_test, name)
            dir_names = os.listdir(path)
            for n in dir_names:
                full_path = os.path.join(path, n)
                if name == "pre_zeizure":
                    preseizure.append((full_path, index))
                else:
                    sleep_normal.append((full_path, index))
                data_test.append((full_path, index))

        t = time.time()
        random.seed(t)
        random.shuffle(data_train)
        t = time.time()
        random.seed(t)
        random.shuffle(data_test)

        self.data_train = data_train
        self.data_test = data_test
        self.train_length = len(data_train)
        self.test_length = len(data_test)
        self.preseizure = preseizure
        self.sleep_normal = sleep_normal


class MyDataset(Dataset):  # 重写dateset的相关类
    def __init__(self, imgs, transform=None, target_transform=None):
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        data = np.load(fn)
        result = matrix_normalization(data, (130, 200))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        return result, label

    def __len__(self):
        return len(self.imgs)


print(TRAIN_PATH)
print(TEST_PATH)
datas = Data_info(path_train=TRAIN_PATH, path_test=TEST_PATH)
all_data = datas.data_train + datas.data_test  # 所有的训练集
positive_loader = MyDataset(datas.preseizure)  # 作为训练集
negative_loader = MyDataset(datas.sleep_normal)  # 作为测试集

all_loader = MyDataset(all_data)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(resize[0] * resize[1], 1000)
        self.fc21 = nn.Linear(1000, 20)
        self.fc22 = nn.Linear(1000, 20)
        self.fc3 = nn.Linear(20, 1000)
        self.fc4 = nn.Linear(1000, resize[0] * resize[1])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, resize[0] * resize[1]))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, resize[0] * resize[1]), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def trans_data(vae_model, data, shape=(130, 200)):
    data_tmp = torch.from_numpy(data)
    data_input = data_tmp.cuda(0)
    recon_batch, _, _ = vae_model(data_input)
    recon_batch = recon_batch.cpu()
    result = recon_batch.detach().numpy()
    result = result.reshape(shape)
    return result


def show_eeg(data):
    plt.imshow(data)
    plt.show()


# 构造了两个VAE的编码器
vae_p = VAE().to(device)
vae_n = VAE().to(device)
optimizer_vae_p = optim.Adam(vae_p.parameters(), lr=0.001)
optimizer_vae_n = optim.Adam(vae_n.parameters(), lr=0.001)


# 仅仅使用一个VAE的编码器
# Vae = VAE().to(device)
# optimizer_vae = optim.Adam(Vae.parameters(), lr=0.005)

# vae 模块
def trans_data_vae(data, label_data):
    shape_data = data.shape
    data_view = data.reshape((-1, resize[0], resize[1]))
    shape_label = label_data.shape
    label_list = label_data.flatten()
    number = shape_label[0] * shape_label[1]
    result = []
    loss_all = 0.0

    for i in range(number):
        data_tmp = data_view[i]
        data_tmp = torch.from_numpy(data_tmp)
        data_tmp = data_tmp.to(device)

        if label_list[i] == 1:  # positive
            optimizer_vae_p.zero_grad()
            recon_batch, mu, logvar = vae_p(data_tmp)
            loss = loss_function(recon_batch, data_tmp, mu, logvar)
            loss.backward()
            optimizer_vae_p.step()
        else:
            optimizer_vae_n.zero_grad()
            recon_batch, mu, logvar = vae_n(data_tmp)
            loss = loss_function(recon_batch, data_tmp, mu, logvar)
            loss.backward()
            optimizer_vae_n.step()
        loss_all += loss.item()
        result_tmp = recon_batch.detach().cpu().numpy()
        result_tmp = result_tmp.reshape(resize)
        data_result = result_tmp[np.newaxis, :]
        result.append(data_result)
    result_t = np.array(result)
    result_r = result_t.reshape(shape_data)
    loss_all = loss_all / number
    return result_r, loss_all


# maml 的网络架构, 需要融合vae模块
def maml_framwork():
    torch.manual_seed(222)  # 为cpu设置种子，为了使结果是确定的
    torch.cuda.manual_seed_all(222)  # 为GPU设置种子，为了使结果是确定的
    np.random.seed(222)

    print(args)

    # 引入vae的模块
    # device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    # if os.path.exists(str(
    #         "./models/maml" + str(args.n_way) + "way_" + str(
    #             args.k_spt) + "shot.pkl")):
    #     path = str("./models/maml" + str(args.n_way) + "way_" + str(args.k_spt) + "shot.pkl")
    #     maml.load_state_dict(torch.load(path))
    #     print("loading MAML model success!")
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = Seegnet(args.dataset_dir, mode='train', n_way=args.n_way, k_shot=args.k_spt,
                   k_query=args.k_qry,
                   batchsz=args.epoch)
    mini_test = Seegnet(args.dataset_dir, mode='test', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=100)
    last_accuracy = 0.0
    plt_train_loss = []
    plt_train_acc = []

    plt_test_loss = []
    plt_test_acc = []

    # flag_vae = True  # 设置梯度反向传播的标志位，vae
    # flag_maml = not flag_vae  # 设置梯度反向传播的薄志伟，maml
    for epoch in range(1):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in tqdm(enumerate(db)):

            x_spt_vae, loss_spt = trans_data_vae(x_spt.numpy(), y_spt)
            x_qry_vae, loss_qry = trans_data_vae(x_qry.numpy(), y_qry)
            x_spt_vae = torch.from_numpy(x_spt_vae)
            x_qry_vae = torch.from_numpy(x_qry_vae)
            x_spt_vae, y_spt, x_qry_vae, y_qry = x_spt_vae.to(device), y_spt.to(device), x_qry_vae.to(device), y_qry.to(
                device)

            accs, loss_q = maml(x_spt_vae, y_spt, x_qry_vae, y_qry, True)
            # maml.meta_optim.zero_grad()
            # loss = loss_spt + loss_qry + loss_q
            # loss.backward()
            # maml.meta_optim.step()

            if step % 20 == 0:
                d = loss_q.cpu()
                dd = d.detach().numpy()
                plt_train_loss.append(dd)
                plt_train_acc.append(accs[-1])
                print('step:', step, '\ttraining acc:', accs)

                if step % 50 == 0:  # evaluation
                    db_test = DataLoader(mini_test, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
                    accs_all_test = []
                    loss_all_test = []

                    for x_spt, y_spt, x_qry, y_qry in db_test:
                        x_spt_vae, loss_spt = trans_data_vae(x_spt.numpy(), y_spt)
                        x_qry_vae, loss_qry = trans_data_vae(x_qry.numpy(), y_qry)
                        x_spt_vae = torch.from_numpy(x_spt_vae)
                        x_qry_vae = torch.from_numpy(x_qry_vae)
                        x_spt_vae, y_spt, x_qry_vae, y_qry = x_spt_vae.squeeze(0).to(device), y_spt.squeeze(0).to(
                            device), x_qry_vae.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                        result, loss_test = maml.finetunning(x_spt_vae, y_spt, x_qry_vae, y_qry)
                        acc = result['accuracy']

                        loss_all_test.append(loss_test.item())
                        accs_all_test.append(acc)

                    # [b, update_step+1]
                    # accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                    plt_test_acc.append(acc)
                    avg_loss = np.mean(np.array(loss_all_test))
                    plt_test_loss.append(avg_loss)

                test_accuracy = np.array(accs_all_test).mean()
                print('Test acc:', test_accuracy)
                if test_accuracy >= last_accuracy:
                    # save networks
                    torch.save(maml.state_dict(), str(
                        "./models/maml" + str(args.n_way) + "way_" + str(
                            args.k_spt) + "shot_{}.pkl".format(patient_test)))
                    last_accuracy = test_accuracy

                    torch.save(vae_p.state_dict(), "./models/Vae_positive_{}.pkl".format(patient_test))
                    print("VAE positive model save successfully!")
                    torch.save(vae_n.state_dict(), "./models/Vae_negative_{}.pkl".format(patient_test))
                    print("VAE negative model save successfully!")
                    print("{} and {} model have saved!!!".format("maml", "vae"))
    plt.figure()
    plt.title("testing info")
    plt.xlabel("episode")
    plt.ylabel("Acc/loss")
    plt.plot(plt_test_loss, label='Loss')
    plt.plot(plt_test_acc, label='Acc')
    plt.legend(loc='upper right')
    plt.savefig('./drawing/test.png')
    # plt.show()

    plt.figure()
    plt.title("training info")
    plt.xlabel("episode")
    plt.ylabel("Acc/loss")
    plt.plot(plt_train_loss, label='Loss')
    plt.plot(plt_train_acc, label='Acc')
    plt.legend(loc='upper right')
    plt.savefig('./drawing/train.png')
    # plt.show()


if __name__ == "__main__":
    maml_framwork()
# for epoch in range(1, args.epochs + 1):
#     train_positive(epoch)
#     train_negative(epoch)
