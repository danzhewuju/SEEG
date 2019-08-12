import re
from util.seeg_utils import *

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
