from __future__ import print_function

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset
import sys
sys.path.append("../")
from util.util_file import matrix_normalization
from tqdm import tqdm

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('-train_p', '--train_path', default='../data/seeg/zero_data/train')
parser.add_argument('-test_p', '--test_path', default='../data/seeg/zero_data/test')
parser.add_argument('-val_p', '--val_path', default='../data/seeg/zero_data/val')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
TEST_PATH = args.test_path
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

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


datas = Data_info(path_train=TRAIN_PATH, path_test=TEST_PATH)
all_data = datas.data_train + datas.data_test
positive_loader = MyDataset(datas.preseizure)  # 作为训练集
negative_loader = MyDataset(datas.sleep_normal)  # 作为测试集

all_loader = MyDataset(all_data)


# train_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=True, download=True,
#                    transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)
# test_loader = torch.utils.data.DataLoader(
#     datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#     batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # self.fc1 = nn.Linear(resize[0] * resize[1], 400)
        self.fc1 = nn.Linear(resize[0]*resize[1], 2000)
        self.fc12 = nn.Linear(2000, 200)
        self.fc21 = nn.Linear(200, 20)
        self.fc22 = nn.Linear(200, 20)
        self.fc3 = nn.Linear(20, 200)
        self.fc4 = nn.Linear(200, resize[0] * resize[1])

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc12(h1))
        return self.fc21(h1), self.fc22(h1)

    def encode_same_size(self, x):
        h = F.relu(self.fc1(x))
        output = h.reshape(resize[0], resize[1])
        return output

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


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, resize[0] * resize[1]), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return abs(BCE+KLD)


def train_negative(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(negative_loader):
        data = torch.from_numpy(data)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(datas.sleep_normal),
                       100. * batch_idx / len(negative_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / datas.train_length))
    name = str("./models/model-vae-negative_normalsleep.ckpt")
    torch.save(model.state_dict(), name)
    print("model has been saved!")


def train_positive(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(positive_loader):
        data = torch.from_numpy(data)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(datas.preseizure),
                       100. * batch_idx / len(positive_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / datas.train_length))
    name = str("./models/model-vae-positive_preseizure.ckpt")
    torch.save(model.state_dict(), name)
    print("model has been saved!")


def train_all_data(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(all_loader):
        data = torch.from_numpy(data)
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(all_data),
                       100. * batch_idx / len(all_loader),
                       loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / datas.train_length))
    name = str("./models/model-vae.ckpt")
    torch.save(model.state_dict(), name)
    print("model has been saved!")


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


if __name__ == "__main__":
    for epoch in tqdm(range(1, args.epochs + 1)):
        # 1.训练正态编码器
        train_positive(epoch)
        # 2. 训练负态编码器
        train_negative(epoch)
        # 3.用全部数据训练编码器， 暂未使用
        # train_all_data(epoch)
