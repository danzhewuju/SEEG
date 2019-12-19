# ----------------------------
# seeg train few shot learning
#
#
# ----------------------------

import argparse
import math

import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import task_generator as tg
from util.util_file import *
import json

config = json.load(open("../DataProcessing/config/fig.json", 'r'))  # 需要指定训练所使用的数据
patient_test = config['patient_test']
print("patient_test is {}".format(patient_test))

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f", "--feature_dim", type=int, default=64)
parser.add_argument("-r", "--relation_dim", type=int, default=8)
parser.add_argument("-w", "--class_num", type=int, default=2)
parser.add_argument("-s", "--sample_num_per_class", type=int, default=10)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=10)
parser.add_argument("-e", "--episode", type=int, default=4000)
parser.add_argument("-t", "--test_episode", type=int, default=50)
parser.add_argument("-l", "--learning_rate", type=float, default=0.001)

parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-mn", '--model_name', type=str, default="zero_data")
args = parser.parse_args()

# Hyper Parameters
FEATURE_DIM = args.feature_dim
RELATION_DIM = args.relation_dim
CLASS_NUM = args.class_num
SAMPLE_NUM_PER_CLASS = args.sample_num_per_class
BATCH_NUM_PER_CLASS = args.batch_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
MODEL_NAME = args.model_name
print("running on data set :{}".format(MODEL_NAME))

# 118
# x_ = 28
# y_ = 48
# f1_line = 50

x_ = 31
y_ = 48
f1_line = 60


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return m, h


# ---------------------------------------------------------
# model framework
class CNNEncoder(nn.Module):
    """docstring for ClassName"""

    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = out.view(out.size(0),-1)
        return out  # 64


class RelationNetwork(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size * f1_line, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


def weights_init(m):  # init weights
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())


# ---------------------------------------------------------


def main():
    # 设置随机数种子
    setup_seed(1)
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    metatrain_folders, metatest_folders = tg.mini_data_folders(MODEL_NAME)

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    feature_encoder.cuda(GPU)  # 利用gpu进行运算
    relation_network.cuda(GPU)  # 利用gpu进行运算

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=10000, gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=10000, gamma=0.5)

    # if os.path.exists(str("./models/seegnet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
    #         SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
    #     feature_encoder.load_state_dict(torch.load(str(
    #         "./models/seegnet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
    #             SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
    #     print("load feature encoder success")
    # if os.path.exists(str("./models/seegnet_relation_network_" + str(CLASS_NUM) + "way_" + str(
    #         SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
    #     relation_network.load_state_dict(torch.load(str(
    #         "./models/seegnet_relation_network_" + str(CLASS_NUM) + "way_" + str(
    #             SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
    #     print("load relation network success")

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0
    plt_test_acc = []
    plt_test_loss = []
    plt_train_acc = []
    plt_train_loss = []

    for episode in range(EPISODE):  # default=1000

        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        task = tg.MiniDataTask(metatrain_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train",
                                                             shuffle=False)
        batch_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=BATCH_NUM_PER_CLASS, split="test",
                                                            shuffle=True)

        # sample datas
        samples, sample_labels = sample_dataloader.__iter__().next()  # 25*3*84*84
        batches, batch_labels = batch_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(samples).cuda(GPU))  # 25*64*19*19
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, x_, y_)
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        batch_features = feature_encoder(Variable(batches).cuda(GPU))  # 20x64*5*5

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(BATCH_NUM_PER_CLASS * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, x_, y_)
        relations = relation_network(relation_pairs).view(-1, CLASS_NUM)

        mse = nn.MSELoss().cuda(GPU)
        one_hot_labels = Variable(
            torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1, 1), 1).cuda(GPU))
        loss = mse(relations, one_hot_labels)

        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        _, predict_labels = torch.max(relations.data, 1)

        batch_labels = batch_labels.cuda(GPU)

        rewards = [1 if predict_labels[j] == batch_labels[j] else 0 for j in range(BATCH_NUM_PER_CLASS * CLASS_NUM)]
        train_acc = sum(rewards) / (BATCH_NUM_PER_CLASS * CLASS_NUM)
        plt_train_acc.append(train_acc)
        plt_train_loss.append(loss.item())

        if (episode + 1) % 10 == 0:
            print("episode:", episode + 1, "loss", loss.item())

        if episode % TEST_EPISODE == 0:

            # test
            print("Testing...")
            accuracies = []
            for i in range(TEST_EPISODE):
                total_rewards = 0
                task = tg.MiniDataTask(metatest_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, 15)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS,
                                                                     split="train", shuffle=False)
                num_per_class = 20
                test_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=num_per_class, split="test",
                                                                   shuffle=False)

                sample_images, sample_labels = sample_dataloader.__iter__().next()
                test_num = 0
                for test_images, test_labels in test_dataloader:
                    batch_size = test_labels.shape[0]
                    # calculate features
                    sample_features = feature_encoder(Variable(sample_images).cuda(GPU))  # 5x64
                    sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, x_, y_)
                    sample_features = torch.sum(sample_features, 1).squeeze(1)
                    test_features = feature_encoder(Variable(test_images).cuda(GPU))  # 20x64

                    # calculate relations
                    # each batch sample link to every samples to calculate relations
                    # to form a 100x128 matrix for relation network
                    sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

                    test_features_ext = test_features.unsqueeze(0).repeat(1 * CLASS_NUM, 1, 1, 1, 1)
                    test_features_ext = torch.transpose(test_features_ext, 0, 1)
                    relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2,
                                                                                                 x_, y_)
                    relations = relation_network(relation_pairs).view(-1, CLASS_NUM)

                    _, predict_labels = torch.max(relations.data, 1)

                    test_labels = test_labels.cuda(GPU)

                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)

                    test_num += batch_size
                accuracy = total_rewards / 1.0 / test_num
                accuracies.append(accuracy)

            test_accuracy, h = mean_confidence_interval(accuracies)
            plt_test_acc.append(test_accuracy)
            plt_test_loss.append(loss.item())

            if test_accuracy > last_accuracy:
                # save networks
                torch.save(feature_encoder.state_dict(), str(
                    "./models/seegnet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(
                        SAMPLE_NUM_PER_CLASS) + "shot_{}.pkl".format(patient_test)))
                torch.save(relation_network.state_dict(), str(
                    "./models/seegnet_relation_network_" + str(CLASS_NUM) + "way_" + str(
                        SAMPLE_NUM_PER_CLASS) + "shot_{}.pkl".format(patient_test)))

                print("save networks for episode:", episode)

                last_accuracy = test_accuracy

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


if __name__ == '__main__':
    main()
