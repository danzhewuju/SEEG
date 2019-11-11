##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Albert Berenguel
## Computer Vision Center (CVC). Universitat Autonoma de Barcelona
## Email: aberenguel@cvc.uab.es
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from datasets import miniImagenetOneShot
from option import Options
from experiments.OneShotMiniImageNetBuilder import miniImageNetBuilder
import tqdm
from config import mean_confidence_interval, logger, Pyemail
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default="1")
args = parser.parse_args()
# from logger import Logger

'''
:param batch_size: Experiment batch_size
:param classes_per_set: Integer indicating the number of classes per set
:param samples_per_class: Integer indicating samples per class
        e.g. For a 20-way, 1-shot learning task, use classes_per_set=20 and samples_per_class=1
             For a 5-way, 10-shot learning task, use classes_per_set=5 and samples_per_class=10
'''

# Experiment Setup
batch_size = 10
fce = True
classes_per_set = 2
samples_per_class = 10
channels = 1
# Training setup
total_epochs = 20
total_train_batches = 1000
total_val_batches = 10
total_test_batches = 1000
# Parse other options
args = Options().parse()

patient_test = "BDP"
print("patient's name is :{}".format(patient_test))
TRAIN_PATH = "/home/cbd109-2/yh.link/python/dataset/zero_data/{}/train".format(patient_test)
TEST_PATH = "/home/cbd109-2/yh.link/python/dataset/zero_data/{}/test".format(patient_test)

# datas = Data_info(path_train=TRAIN_PATH, path_test=TEST_PATH)
# train_data = MyDataset(datas.data_train)  # 作为训练集
# test_data = MyDataset(datas.data_test)  # 作为测试集

# train_sampler = CategoriesSampler(train_data.labels, 100,
#                                   args.train_way, args.shot + args.query)
# train_loader = DataLoader(dataset=train_data, batch_sampler=train_sampler,
#                           num_workers=8, pin_memory=True)
#
# val_sampler = CategoriesSampler(test_data.labels, 400,
#                                 args.test_way, args.shot + args.query)
# val_loader = DataLoader(dataset=test_data, batch_sampler=val_sampler,
#                         num_workers=8, pin_memory=True)

LOG_DIR = args.log_dir + '/miniImageNetOneShot_run-batchSize_{}-fce_{}-classes_per_set{}-samples_per_class{}-channels{}' \
    .format(batch_size, fce, classes_per_set, samples_per_class, channels)

# create logger
# logger = Logger(LOG_DIR)

# args.dataroot = '/home/aberenguel/Dataset/miniImagenet'
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
dataTrain = miniImagenetOneShot.miniImagenetOneShotDataset(dataroot=args.dataroot,
                                                           type='train',
                                                           nEpisodes=total_train_batches * batch_size,
                                                           classes_per_set=classes_per_set,
                                                           samples_per_class=samples_per_class)

dataVal = miniImagenetOneShot.miniImagenetOneShotDataset(dataroot=args.dataroot,
                                                         type='val',
                                                         nEpisodes=total_val_batches * batch_size,
                                                         classes_per_set=classes_per_set,
                                                         samples_per_class=samples_per_class)

dataTest = miniImagenetOneShot.miniImagenetOneShotDataset(dataroot=args.dataroot,
                                                          type='test',
                                                          nEpisodes=total_test_batches * batch_size,
                                                          classes_per_set=classes_per_set,
                                                          samples_per_class=samples_per_class)

obj_oneShotBuilder = miniImageNetBuilder(dataTrain, dataVal, dataTest)
obj_oneShotBuilder.build_experiment(batch_size, classes_per_set, samples_per_class, channels, fce)

best_val = 0.
total_accuracy_val = []
total_recall_val = []
total_precision_val = []
total_f1_val = []
total_auc_val = []
with tqdm.tqdm(total=total_epochs + 10) as pbar_e:
    for e in range(0, total_epochs + 10):
        total_c_loss, total_accuracy = obj_oneShotBuilder.run_training_epoch()
        print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

        total_test_c_loss, total_test_accuracy = obj_oneShotBuilder.run_testing_epoch()
        print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))

        print('train_loss', total_c_loss)
        print('train_acc', total_accuracy)
        print('test_loss', total_test_c_loss)
        print('test_acc', total_test_accuracy)

        if e >= total_epochs:
            total_val_c_loss, total_val_accuracy, performances = obj_oneShotBuilder.run_validation_epoch()
            total_accuracy_val += performances["accuracy"]
            total_precision_val += performances["precision"]
            total_recall_val += performances["recall"]
            total_f1_val += performances["f1"]
            total_auc_val += performances["auc"]
        else:
            if total_test_accuracy >= best_val:  # if new best val accuracy -> produce test statistics
                best_val = total_test_accuracy
                total_val_c_loss, total_val_accuracy, performances = obj_oneShotBuilder.run_validation_epoch()
                print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))
                print('val_loss', total_val_c_loss)
                print('val_acc', total_val_accuracy)
            else:
                total_test_c_loss = -1
                total_test_accuracy = -1
        pbar_e.update(1)
    avg_accuracy, h_acc = mean_confidence_interval(total_accuracy_val)
    avg_precision, h_p = mean_confidence_interval(total_precision_val)
    avg_recall, h_r = mean_confidence_interval(total_recall_val)
    avg_f1, h_f = mean_confidence_interval(total_f1_val)
    avg_auc, h_a = mean_confidence_interval(total_auc_val)
    result = "\naccuracy:{}, h:{}\nprecision:{}, h:{}\nrecall:{}, h:{}\nf1:{}, h:{}\n auc:{}, h:{}".format(
        avg_accuracy, h_acc, avg_precision, h_p, avg_recall, h_r, avg_f1, h_f, avg_auc, h_a)
    logger(result)
    os.system("~/login")  # login in network
    Pyemail("experiments ending !", result)

    # logger.step()
