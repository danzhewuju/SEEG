import re
import sys

sys.path.append("../")
from util.seeg_utils import *
from util import *
from PIL import Image
from grad_cam import *


def test_1():
    path = "./heatmap/0f24adba-a894-11e9-bc3a-338334ea1429-0-loc-0-0-1-1-0.jpg"
    raw_path = "../data/data_slice/split/preseizure/LK/0f24adba-a894-11e9-bc3a-338334ea1429-0.npy"
    raw_data = np.load(raw_path)
    save_path = "./"
    path_tmp = path
    channels_str = re.findall('-loc-(.+).jpg', path_tmp)[0]
    channels_number = map(int, channels_str.split('-'))
    channels_number = list(set(channels_number))
    if len(channels_number) > 3:
        channels_number = channels_number[:3]
    channels_number.sort()

    seeg_npy_plot(raw_data, channels_number, save_path)


def test_2():
    # 图像拼接
    path_dir = "./log/heatmap.csv"
    data = pd.read_csv(path_dir, sep=',')
    print(data.loc[1]["grant truth"])


def test_3():
    path = "../data/raw_data/LK/LK_Pre_seizure/LK_SZ1_pre_seizure_raw.fif"
    data = read_raw(path)
    data.filter(0, 30)
    print("Hello")
    # data_1 = filter_hz(data, 0, 30)


if __name__ == '__main__':
    test_3()
