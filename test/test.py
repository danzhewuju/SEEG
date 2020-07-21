#!/usr/bin/python
import sys
sys.path.append('../')
from RelationNet.Seegdata import *
from util.util_file import *
import re
from util.util_file import IndicatorCalculation, similarity_dtw, LogRecord, get_label_data
import logging
from tqdm import tqdm
import pandas as pd
import torch


# from torch.nn import functional as F


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_1():
    b = []
    for index in range(5):
        a = np.random.randint(0, 100, 10)
        a = a[np.newaxis, :]
        b.append(a)
    # a = np.random.randint(0, 10, 10)
    c = np.array(b)
    c = c.reshape((-1, 10))
    print(c)


def test_2():
    path = "../data/output_data/sleep_frame_eeg.npy"
    data = np.load(path)
    print(data.shape)
    print(data)


def test_4(path="../data/data_path.txt"):
    '''

    :param path: 文件的存储目录
    :return:
    '''

    f = open(path, 'r', encoding="UTF-8")
    paths = [line.rsplit()[0] for line in
             f.readlines()]  # ['E:\\数据集\\表型数据集\\SEEG_Data\\LK处理数据', 'E:\\数据集\\表型数据集\\SEEG_Data\\SGH处理数据']
    f.close()
    print("dir : {}".format(paths))
    LK_path = glob.glob(os.path.join(paths[0], '*.fif'))
    SGH_path = glob.glob(os.path.join(paths[1], '*.fif'))
    print("LK files: {}, count: {}".format(LK_path, len(LK_path)))
    print("SGH files: {}, count: {}".format(SGH_path, len(SGH_path)))
    file_map = {"LK": LK_path, "GHB": SGH_path}
    return file_map


def test_9():  # 其他功能的探索
    a = {"1": 12, "2": 23, "3": 321}
    b = list(a.keys())
    print("Test")
    print(b)


def test_11():
    path_dir = '../data/raw_data/Pre_seizure'
    print(os.listdir(path_dir))


def test_12():
    path_channel = '../data/data_slice/channels_info_back/ZK_seq.csv'
    path = '../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw.fif'
    raw_data = read_raw(path)
    raw_data.plot()
    print(raw_data.info['ch_names'])
    data = pd.read_csv(path_channel, sep=',')
    d_list = data['channels']
    common_channels = list(d_list)
    new_raw_data = select_channel_data_mne(raw_data, common_channels)
    new_raw_data.reorder_channels(common_channels)
    new_raw_data.plot()
    print(new_raw_data.info['ch_names'])

    # print(d_list)
    # print(data['channels'])


def test_13():
    a = np.random.randint(0, 100, 20)
    b = a.copy()
    c = a.copy()
    random.shuffle(b)
    random.shuffle(c)
    print(a, b, c)


def test_14():
    seeg = seegdata()
    path_dir_seizure = "../data/data_slice/split/preseizure"
    seeg.set_path_dir(path_dir_seizure)
    sleep_bwt = seeg.get_all_path_by_keyword('within_warning_time')
    sleep_bwt_label1 = sleep_bwt['LK']  # 发病前的一段时间,警戒线之外
    for p in sleep_bwt_label1:
        d = np.load(p)
        print(d.shape)


def test_15():
    path = './path.csv'
    paths = pd.read_csv(path, sep=',')
    print(paths)
    raw_data_path = paths['path']
    raw_name = paths['name']
    f = open('./channel.txt', 'w', encoding='UTF-8')
    f.writelines('name, channels\n')
    for n, p in zip(raw_name, raw_data_path):
        data = read_edf_raw(p)
        channels = get_channels_names(data)
        # print(channels)
        print("name:{}, channels: {}".format(n, len(channels)))
        dd = n + ',' + str(channels) + '\n'
        f.writelines(dd)
    f.close()


def test_17():
    a = np.random.randint(0, 100, 10).tolist()
    b = random.sample(a, 5)
    print(b)


def test_channels_matching():
    path = '/home/cbd109-3/Users/data/hmq/Huashan_Hospital_Preprocessed_Data/huashan_data_normal_3/SJ_Sleep.edf'
    data = read_edf_raw(path)
    channels_name = get_channels_names(data)
    print(len(channels_name))
    print(channels_name)


def test_18():
    path = '../data/seeg/val/pre_zeizure'
    paths = [os.path.join(path, x) for x in os.listdir(path)]
    # print(paths)
    path_a = []
    for p in paths:
        d = np.load(p)
        if d.shape != (130, 200):
            path_a.append(p)
    # print(path_a)
    dd = np.load(path_a[0])
    print(dd.shape)
    dd = matrix_normalization(dd, (130, 200))
    print(dd.shape)


def test_cvs():
    data = {'id': [1, 2, 3],
            'name': ['yuhao', 'alex', 'jj']
            }
    d = pd.DataFrame(data)
    print(d)
    c = d[d.id == 2].index
    print(c)
    # d.loc[c, 'name'] = 'lijian'
    # print(d)


def test_edf():
    path_channel = '../data/data_slice/channels_info_back/ZK_seq.csv'
    channel_info = pd.read_csv(path_channel)['channels']
    query_t = [x for x in channel_info if "5" in x]

    path = '../data/raw_data/ZK/ZK_SLEEP/ZK_Sleep_raw.fif'
    data = read_raw(path)
    time = data.times
    channels_name = get_channels_names(data)
    query = [x for x in channels_name if "5" in x]
    print(time)
    print(query)
    print(query_t)
    return True


def test_def_eeg_information():
    path = "../data/raw_data/WSH/WSH_Pre_seizure/WSH_SZ1_pre_seizure_raw.fif"
    data = read_raw(path)
    time = get_recorder_time(data)
    print(time)


def test19():
    path = "../data/seeg/zero_data/val/pre_zeizure/0db9c69c-a842-11e9-b4a4-331cd3a6dda7-0.npy"
    data = np.load(path)
    img = trans_numpy_cv2(data)
    cv2.imwrite("./test.png", img)
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    plt.imshow(img)
    plt.show()
    plt.imshow(data)
    plt.show()


def test20():
    data = np.random.randn(200, 130)
    print(data)
    d = np.argsort(data)[:-5]
    print(d)
    d_f = data.flatten()
    index_id = d_f.argsort()[-2:]
    x_index, y_index = np.unravel_index(index_id, data.shape)
    location = list(zip(x_index, y_index))
    print(location)


def test_21():
    path = "./heatmap/19df46c0-a894-11e9-bc3a-338334ea1429-0-loc-0-7-6-7-0.jpg"
    channels_str = re.findall('-loc-(.+).jpg', path)[0]
    channels_number = map(int, channels_str.split('-'))
    channels_number = list(set(channels_number))
    channels_number.sort()
    print(channels_number)
    d = re.sub('-loc(.+).jpg', '', path)
    name = d + ".npy"
    name = re.findall('heatmap/(.+)', name)[0]
    print(name)

    path_1 = "'../data/data_slice/split/preseizure/LKKKKKKK/1d61665c-a894-11e9-bc3a-338334ea1429-0.npy'"
    name = re.findall("/.+/(.+)", path_1)[0]
    print(name)


def test_22():
    uuids = []
    for i in range(100):
        str = uuid.uuid1()
        print(str)
        uuids.append(str)
    uuids.reverse()
    print(uuids)
    uuids.sort()
    print(uuids)


def test_23():
    clean_dir("../data/seeg/")


def test_24():
    a = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]

    b = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    cal = IndicatorCalculation(b, a)
    print(cal.get_accuracy(), cal.get_precision(), cal.get_recall(), cal.get_f1score(), cal.calculate_auc())


def recall(p, r):
    f = 2 * p * r / (p + r)
    print("recall:{}".format(f))


def log(f):
    def wrapper(*args, **kwargs):
        logging.basicConfig(level=logging.INFO, filename='test.log', format="%(levelname)s:%(asctime)s:%(message)s")
        print("This is log!")
        return f(*args)

    return wrapper


def test_dir():
    path = "./yuhao/ni/hao"
    dir_create_check(path)
    print(path)


def dtw_test():
    path = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/visualization_feature/raw_data_time_sequentially/preseizure/BDP/27a4193a-f62c-11e9-a8f2-e0d55e6ff654-0.npy"
    data = np.load(path)
    scores = []
    for p in tqdm(data[:20]):
        score = 0
        for m in data:
            score += similarity_dtw(p, m)
        score /= data.shape[0]
        scores.append(score)
    print(scores)


def uuid_test():
    n = 10
    uuids = []
    for i in range(10):
        s = uuid.uuid1()
        uuids.append(str(s))
    u = uuids.copy()
    print(u)
    random.shuffle(uuids)
    uu = uuids.copy()
    print(uu)
    uuids.sort()
    print(uuids)
    if u == uu:
        print("True")
    else:
        print("False")


def a():
    a = np.random.rand(10)
    print(a)
    b = np.pad(a, (0, 10), 'constant')
    print(b)


def b():
    a = range(0, 10)
    b = range(10, 20)
    dict_a_b = {"a": a, "b": b}
    a_b = pd.DataFrame(dict_a_b)
    a_b.to_csv("./test.csv", index=None)
    print(a_b)


def test_auc():
    a = [1, 1, 0, 0, 1]
    b = [1, 1, 0, 0, 0]
    a = np.asarray(a)
    b = np.asarray(b)
    a = torch.from_numpy(a)
    b = torch.from_numpy(b)
    cal = IndicatorCalculation()
    cal.set_values(b, a)
    print("ACC:{}, Precision:{}, Recall:{}, F1:{}, AUC:{}".format(cal.get_accuracy(), cal.get_precision(),
                                                                  cal.get_recall(), cal.get_f1score(), cal.get_auc()))


import multiprocessing


def test_log_record():
    result = "this is testing!"
    LogRecord.write_log(result)


def test_get_label():
    path = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/data/seeg/zero_data/BDP/train"
    data = get_label_data(path)
    print(data)


def test_npy():
    path = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/NoteBookJupyter/data/pre_1/heatmap_data_storage/e076f2d4-2552-11ea-9699-e0d55e6ff654-0-loc-137-140-141-136-135.npy"
    data = np.load(path)
    print(data)


if __name__ == '__main__':
    path = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/data/raw_data/BDP/BDP_Pre_seizure/BDP_SZ2_pre_seizure_raw.fif"
    data = read_raw(path)
    time_length = get_recorder_time(data) / 2
    print(time_length)

# print(__file__)
# test_log_record()
