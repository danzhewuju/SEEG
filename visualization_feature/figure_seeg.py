#!/usr/bin/Python
'''
author: Alex
function:
'''

import sys

from util import *

sys.path.append('..')

from RelationNet.Seegdata import seegdata


def seeg_visualization(path):
    data = np.load(path)
    plt.imshow(data)
    plt.show()


def seeg_plot(path):
    data_raw = read_raw(path)
    data_raw.plot(duration=30)


def get_all_visualization_feature(key='LK', status='preseizure', top_k=20):
    seeg = seegdata()
    path_dict = seeg.get_all_path_by_keyword(status)
    data = path_dict[key]
    print("data length: {}".format(len(data)))
    count = len(data)
    p = data.pop()
    sum = np.load(p)
    for p in data:
        d = np.load(p)
        d = np.abs(d)
        sum += d
    avg_p = sum / count
    print(avg_p)
    plt.imshow(avg_p)
    plt.show()
    t_axis = np.sum(avg_p, axis=1)
    print(t_axis)
    t_axis = np.abs(t_axis)
    t_dict = dict(zip(t_axis, range(len(t_axis))))
    t_dict = sorted(t_dict.items(), key=lambda x: x[0])  # 将能量从高到低排序
    print(t_dict)
    top = min(top_k, len(t_dict))
    top_data = []
    for i in range(top):
        top_data.append(t_dict[i][-1])
    print("top channels:{}".format(top_data))
    top_data = sorted(top_data)
    print("top channels:{}".format(top_data))
    return avg_p


# def test():
#     path_preseizure = '../data/seeg/train/pre_zeizure/9a0f3d18-7870-11e9-bc1a-5187a394df03-0.npy'
#     path_sleep_normal = '../data/seeg/train/sleep_normal/2b9ed7ac-7637-11e9-aa98-2919e0abbf44-2.npy'
#     seeg_visualization(path_preseizure)
#     seeg_visualization(path_sleep_normal)

# path_preseizure_raw = '../data/raw_data/LK_Pre_seizure/LK_SZ1_pre_seizure_raw.fif'
# path_sleep_raw = '../data/raw_data/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-0.fif'
# seeg_plot(path_preseizure_raw)
# seeg_plot(path_sleep_raw)


if __name__ == '__main__':
    pre_seizure = get_all_visualization_feature(status='preseizure')
    normal_sleep = get_all_visualization_feature(status='sleep')
    dd = pre_seizure - normal_sleep
    dd = np.abs(dd)
    plt.imshow(dd)
    plt.show()
    t_axis = np.sum(dd, axis=1)
    t_axis = np.abs(t_axis)
    t_dict = dict(zip(t_axis, range(len(t_axis))))
    t_dict = sorted(t_dict.items(), key=lambda x: x[0])  # 将能量从高到低排序
    print(t_dict)
    top = min(20, len(t_dict))
    top_data = []
    for i in range(top):
        top_data.append(t_dict[i][-1])
    print("top channels:{}".format(top_data))
    top_data = sorted(top_data)
    print("top channels:{}".format(top_data))
