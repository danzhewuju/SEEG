import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
from torch.utils.data import DataLoader

sys.path.append('../')
from Mamlnet import Seegnet
from meta import Meta


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

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)

    # batchsz here means total episode number
    mini = Seegnet(args.dataset_dir, mode='train_vae', n_way=args.n_way, k_shot=args.k_spt,
                   k_query=args.k_qry,
                   batchsz=args.epoch)
    mini_test = Seegnet(args.dataset_dir, mode='test_vae', n_way=args.n_way, k_shot=args.k_spt,
                        k_query=args.k_qry,
                        batchsz=100)
    last_accuracy = 0.0
    plt_train_loss = []
    plt_train_acc = []

    plt_test_loss = []
    plt_test_acc = []
    for epoch in range(args.epoch // args.epoch):
        # fetch meta_batchsz num of episode each time
        db = DataLoader(mini, args.task_num, shuffle=True, num_workers=1, pin_memory=True)

        for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

            x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

            accs, loss_q = maml(x_spt, y_spt, x_qry, y_qry)

            if step % 20 == 0:
                d = loss_q.cpu()
                dd = d.detach().numpy()
                plt_train_loss.append(dd)
                plt_train_acc.append(accs[-1])
                print('step:', step, '\ttraining acc:', accs)

            if step % 50 == 0:  # evaluation
                db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
                accs_all_test = []
                loss_all_test = []

                for x_spt, y_spt, x_qry, y_qry in db_test:
                    x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device), \
                                                 x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                    accs, loss_test = maml.finetunning(x_spt, y_spt, x_qry, y_qry)

                    loss_all_test.append(loss_test)
                    accs_all_test.append(accs)

                # [b, update_step+1]
                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                plt_test_acc.append(accs[-1])
                avg_loss = np.mean(np.array(loss_all_test))
                plt_test_loss.append(avg_loss)

                print('Test acc:', accs)
                test_accuracy = accs[-1]
                if test_accuracy >= last_accuracy:
                    # save networks
                    torch.save(maml.state_dict(), str(
                        "./models/maml" + str(args.n_way) + "way_" + str(
                            args.k_spt) + "shot.pkl"))
                    last_accuracy = test_accuracy
    plt.figure()
    plt.title("testing info")
    plt.xlabel("episode")
    plt.ylabel("Acc/loss")
    plt.plot(plt_test_loss, label='Loss')
    plt.plot(plt_test_acc, label='Acc')
    plt.legend(loc='upper right')
    plt.savefig('./drawing/test.png')
    plt.show()

    plt.figure()
    plt.title("training info")
    plt.xlabel("episode")
    plt.ylabel("Acc/loss")
    plt.plot(plt_train_loss, label='Loss')
    plt.plot(plt_train_acc, label='Acc')
    plt.legend(loc='upper right')
    plt.savefig('./drawing/train.png')
    plt.show()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=10000)
    argparser.add_argument('--n_way', type=int, help='n way', default=2)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=100)
    argparser.add_argument('--imgc', type=int, help='imgc', default=5)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.01)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)
    argparser.add_argument('--dataset_dir', type=str, help="training data set", default="../data/seeg/zero_data")

    args = argparser.parse_args()

    main()
