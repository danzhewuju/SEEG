import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import sys
sys.path.append('../')
from MAML.learner import *
import json
from VMAML.vmeta import Meta
from util import matrix_normalization_recorder


from util.util_file import matrix_normalization, trans_numpy_cv2, get_matrix_max_location

flag_model = "MAML"  # 切换内核，有两种模式：CNN, VMAML
print("using model:{}".format(flag_model))
x_ = 8
y_ = 12
NUM_CLASS = 2
shape = (200, 130)  # cv2的图片坐标和numpy的坐标并不一致， cv2:(x, y)  PIL:(y,x)

'''
feature extraction by maml model !
'''
config_maml = [
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
    ('linear', [2, 7040])
]


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     # nn.ReLU(inplace=True),
        #     # nn.Linear(4096, num_classes),
        # )
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(x_ * y_ * 32, 32)  # x_ y_ 和你输入的矩阵有关系
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, NUM_CLASS)  # 取决于最后的个数种类

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.reshape(out.size(0), -1)  # 这里面的-1代表的是自适应的意思。
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []

        # CNN
        if flag_model == "CNN":
            for name, module in self.model._modules.items():  # 特定的maml
                if name == 'fc1':
                    x = x.reshape(1, -1)
                x = module(x)
                if name in self.target_layers:
                    x.register_hook(self.save_gradient)
                    outputs += [x]
            return outputs, x
        else:
            # MAML MAML 热力图的计算和CNN热力图的计算不太一样
            for name, module in self.model._modules.items():  # 特定的maml
                x_r = module(x)
                x = module.feature_heat_map
                x.register_hook(self.save_gradient)
                outputs += [x]
            return outputs, x_r


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        # output = self.model.classifier(output)
        return target_activations, output


def preprocess_image(img):
    means = [0.456]
    stds = [0.224]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(1):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img.copy())
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask, save_path):
    # img = np.transpose(img, (1, 0, 2))
    # mask = mask.T
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_path, np.uint8(255 * cam))


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda(0))
        else:
            features, output = self.extractor(input)

        # if index == None:
        #     index = np.argmax(output.cpu().data.numpy())
        index_p = np.argmax(output.cpu().data.numpy())
        if os.path.exists('./log/') is not True:
            os.mkdir('./log/')
        if os.path.exists("./log/heatmap.csv") is not True:
            f = open("./log/heatmap.csv", 'w')
            f.writelines("ground truth,prediction\n")
        else:
            f = open("./log/heatmap.csv", 'a')

        str = "{},{}\n".format(index, index_p)
        f.writelines(str)
        f.close()

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, shape)  # 将重新划分大小
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    '''

    :return: args parameters setting
    '''
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=4000)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=50)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--dataset_dir', type=str, help="training data set", default="../data/seeg/zero_data")
    argparser.add_argument('--use-cuda', action='store_true', default=False,
                           help='Use NVIDIA GPU acceleration')
    args = argparser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


def get_feature_map(path_data, location_name):
    '''

    :param path_data: raw data path
    :param location_name: raw data original file name ex: LK_SZ1_pre_seizure_raw.txt
    :return:
    '''
    args = get_args()
    config = json.load(open('./json_path/config.json'))

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    device = torch.device('cuda')
    if flag_model == "CNN":
        model = CNN().cuda(device) if args.use_cuda else CNN()  # 模型架构的调整， 1.CNN, 2. MAML
        model_path = config['grad_cam.get_feature_map__model_path_cnn']
    else:
        model = Meta(args, config_maml).cuda(device) if args.use_cuda else Meta(args, config_maml)
        model_path = config['grad_cam.get_feature_map__model_path_maml']

    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print("load {} model success!".format(model_path))
    grad_cam = GradCam(model=model, target_layer_names=["layer4"], use_cuda=args.use_cuda)

    # img = cv2.imread(args.image_path, 1)
    data = np.load(path_data)
    # if os.path.exists("./log/random_operate_channel_record.csv"):
    #     f = open("./log/random_operate_channel_record.csv", 'a')
    # else:
    #     f = open("./log/random_operate_channel_record.csv", 'w')
    #     f.write("id,operate,channel\n")
    data_numpy, recoder = matrix_normalization_recorder(data)  # 需要记录出随机初始化的过程
    # operate = "add" if recoder[0] == 1 else "del"
    # content = path_data + "," + operate + "," + "-".join(recoder) + "\n"
    # f.write(content)
    # f.close()

    img = trans_numpy_cv2(data_numpy)

    img = np.float32(cv2.resize(img, shape)) / 255
    img = img[:, :, np.newaxis]
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = 0  # 目标函数

    mask = grad_cam(input, target_index)
    location = get_matrix_max_location(mask, 5)  # 获得最大梯度的位置，包含时间位置和物理位置
    # 记录随机化过程操作删除的信道信息
    channel_number = []
    for i in range(5):
        channel_number.append(location[i][0])  # 计算的出来的经过采样后的数据
    recorder_old_channel_index = []
    flag_op_channel = recoder[0]
    recoder = recoder[1:]
    if flag_op_channel != 0:
        for p in channel_number:
            count = 0
            for t in recoder:
                if t <= p:
                    count += 1
            if flag_op_channel == -1:
                recorder_old_channel_index.append(p + count)  # 原来的随机过程进行了删除，现在需要复原
            if flag_op_channel == 1:
                recorder_old_channel_index.append(p - count)  # 原来的随机过程进行了采样

    location_full_path = os.path.join("./log", location_name)
    if os.path.exists(location_full_path):
        fp = open(location_full_path, 'a')
    else:
        fp = open(location_full_path, 'w')
        header = "time_location,spatial_location\n"
        fp.write(header)  # 头部的信息
    location_spatial = [str(x) for x in recorder_old_channel_index]  # 记录的物理信道的位置不再是随机采样处理后的位置，而是之前的位置
    location_spatial_str = "-".join(location_spatial)
    location_time = [str(x[1]) for x in location]
    location_time_str = "-".join(location_time)
    fp.write("{},{}\n".format(location_time_str, location_spatial_str))
    fp.close()

    channel_location = "-loc" + ("-{}" * 5).format(recorder_old_channel_index[0], recorder_old_channel_index[1],
                                                   recorder_old_channel_index[2], recorder_old_channel_index[3],
                                                   recorder_old_channel_index[4])

    name = path_data.split("/")[-1][:-4] + channel_location + ".jpg"
    save_path = os.path.join("./heatmap", name)
    show_cam_on_image(img, mask, save_path)  # 将热力图写回到原来的图片


def get_feature_map_dynamic(data, name, key_flag=True):
    '''

    :param path_data: raw data path
    :param location_name: raw data original file name ex: LK_SZ1_pre_seizure_raw.txt
    :return:
    '''
    args = get_args()
    config = json.load(open('./json_path/config.json'))

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    device = torch.device('cuda')
    if flag_model == "CNN":
        model = CNN().cuda(device) if args.use_cuda else CNN()  # 模型架构的调整， 1.CNN, 2. MAML
        model_path = config['grad_cam.get_feature_map__model_path_cnn']
    else:
        model = Meta(args, config_maml).cuda(device) if args.use_cuda else Meta(args, config_maml)
        model_path = config['grad_cam.get_feature_map__model_path_maml']

    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
    print("load {} model success!".format(model_path))
    grad_cam = GradCam(model=model, target_layer_names=["layer4"], use_cuda=args.use_cuda)

    # img = cv2.imread(args.image_path, 1)
    data_numpy = matrix_normalization(data)
    img = trans_numpy_cv2(data_numpy)

    img = np.float32(cv2.resize(img, shape)) / 255
    img = img[:, :, np.newaxis]
    input = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = 0  # 癫痫发作前

    mask = grad_cam(input, target_index)
    location = get_matrix_max_location(mask, 1)  # 获得最大梯度的位置，包含时间位置和物理位置
    time = location[0][0]
    if key_flag:
        if time < 50 or time > 150:
            # 信号又截断的可能，需要返回重新定位
            print("Thermal signal is cut off!")

            return time / 100
        else:
            # 不在保存相关的信道信息，仅仅只对照片进行保留

            save_path = os.path.join("./heatmap", name)
            show_cam_on_image(img, mask, save_path)  # 将热力图写回到原来的图片
            return -1  # 返回安全吗
    else:
        # 不在保存相关的信道信息，仅仅只对照片进行保留

        save_path = os.path.join("./heatmap", name)
        show_cam_on_image(img, mask, save_path)  # 将热力图写回到原来的图片
        return -1  # 返回安全吗
