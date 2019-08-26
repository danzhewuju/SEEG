import re
from util.seeg_utils import *
from util import *
from PIL import Image


def test_1():
    path = "./examples/0f24adba-a894-11e9-bc3a-338334ea1429-0-loc-0-0-1-1-0.jpg"
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
    path_dir = "./examples"
    path_dir = get_first_dir_path(path_dir)
    path_1 = path_dir[0]
    path_2 = path_dir[1]
    imag_test = Image.open(path_1)
    dst_1 = imag_test.transpose(Image.ROTATE_90)
    width, height = dst_1.size
    imag_test = Image.open(path_2)
    dst_2 = imag_test.transpose(Image.ROTATE_90)
    result = Image.new(dst_1.mode, (width*2, height))

    result.paste(dst_1, box=(0,0))
    result.paste(dst_2, box=(200, 0))
    plt.imshow(result)
    plt.show()


