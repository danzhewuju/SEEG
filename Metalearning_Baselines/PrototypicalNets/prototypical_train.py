import argparse
import os.path as osp
from utils import MyDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from samplers import CategoriesSampler
from convnet import Convnet
from utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric, Data_info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--train-way', type=int, default=2)
    parser.add_argument('--test-way', type=int, default=2)
    parser.add_argument('--step-size', type=int, default=20)
    parser.add_argument('--save-path', default='./save/proto-1')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    ensure_path(args.save_path)
    patient_test = "SYF"
    print("patient's name is :{}".format(patient_test))
    TRAIN_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/data/seeg/zero_data/{}/train".format(patient_test)
    TEST_PATH = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/data/seeg/zero_data/{}/test".format(patient_test)

    # trainset = MiniImageNet('train')
    datas = Data_info(path_train=TRAIN_PATH, path_test=TEST_PATH)
    train_data = MyDataset(datas.data_train)  # 作为训练集
    test_data = MyDataset(datas.data_test)  # 作为测试集

    train_sampler = CategoriesSampler(train_data.labels, 100,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=train_data, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

    val_sampler = CategoriesSampler(test_data.labels, 400,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=test_data, batch_sampler=val_sampler,
                            num_workers=8, pin_memory=True)

    model = Convnet().cuda(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.5)


    def save_model(name):
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))


    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    for epoch in range(1, args.max_epoch + 1):
        lr_scheduler.step()

        model.train()

        tl = Averager()
        ta = Averager()

        for i, batch in enumerate(train_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.train_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                  .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            proto = None;
            logits = None;
            loss = None

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)

            proto = None;
            logits = None;
            loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > trlog['max_acc']:
            trlog['max_acc'] = va
            save_model('max-acc')

        trlog['train_loss'].append(tl)
        trlog['train_acc'].append(ta)
        trlog['val_loss'].append(vl)
        trlog['val_acc'].append(va)

        torch.save(trlog, osp.join(args.save_path, 'trlog'))

        save_model('epoch-last')

        if epoch % args.save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.max_epoch)))
