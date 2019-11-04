import argparse

from torch.utils.data import DataLoader

from torch.utils.data import Dataset

from samplers import CategoriesSampler
import logging
from convnet import Convnet

from utils import *


class Data_info():
    def __init__(self, path_val):  # verification dataset
        index_name_val = os.listdir(path_val)
        data_val = []
        for index, name in enumerate(index_name_val):
            path = os.path.join(path_val, name)
            dir_names = os.listdir(path)
            for n in dir_names:
                full_path = os.path.join(path, n)
                data_val.append((full_path, index))

        self.val = data_val
        self.val_length = len(data_val)


class MyDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas
        data, labels = zip(*datas)
        self.labels = list(labels)

    def __getitem__(self, item):
        d_p, label = self.datas[item]
        data = np.load(d_p)
        result = matrix_normalization(data, (130, 200))
        result = result.astype('float32')
        result = result[np.newaxis, :]
        # result = trans_data(vae_model, result)
        return result, label

    def __len__(self):
        return len(self.datas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--load', default='./save/proto-1/max-acc.pth')
    parser.add_argument('--batch', type=int, default=2000)
    parser.add_argument('--way', type=int, default=2)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=30)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    patient_test = "BDP"
    VAL_PATH = "/home/cbd109-2/yh.link/python/dataset/zero_data/{}/val".format(patient_test)

    # dataset = MiniImageNet('test')
    data_info = Data_info(VAL_PATH)
    val_data = MyDataset(data_info.val)  # 标准数据集的构造

    sampler = CategoriesSampler(val_data.labels,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(val_data, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()
    ave_acc = []
    ave_precision = []
    ave_recall = []
    ave_f1 = []
    ave_auc = []

    cal = IndicatorCalculation()

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]

        x = model(data_shot)
        x = x.reshape(args.shot, args.way, -1).mean(dim=0)
        p = x

        logits = euclidean_metric(model(data_query), p)

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        pre = torch.argmax(logits, dim=1)

        cal.set_values(pre, label)
        ave_acc.append(cal.get_accuracy())
        ave_precision.append(cal.get_precision())
        ave_recall.append(cal.get_recall())
        ave_f1.append(cal.get_f1score())
        ave_auc.append(cal.get_auc())
        # ave_acc.append(acc)
        print('batch {}: Accuracy:{:.2f}, Precision:{:.2f},Recall:{:.2f},F1:{:.2f},AUC:{:.2f}'.format(i, ave_acc[-1],
                                                                                                     ave_precision[-1],
                                                                                                     ave_recall[-1],
                                                                                                     ave_f1[-1],
                                                                                                     ave_auc[-1]))

        x = None;
        p = None;
        logits = None

    total_acc, h_acc = mean_confidence_interval(ave_acc)
    total_pre, h_pre = mean_confidence_interval(ave_precision)
    total_recall, h_r = mean_confidence_interval(ave_recall)
    total_f1, h_f1 = mean_confidence_interval(ave_f1)
    total_auc, h_auc = mean_confidence_interval(ave_auc)
    result = "accuracy:{:.5f},h:{:.5f}\nprecision:{:.5f},h:{:.5f}\nrecall:{:.5f},h:{:.5f}\nf1:{:.5f},h:{:.5f}\n" \
             "auc:{:.5f},h:{:.5f}".format(total_acc, h_acc, total_pre, h_pre, total_recall, h_r, total_f1, h_f1,
                                          total_auc, h_auc)
    logger(result, name="prototypical.txt")
