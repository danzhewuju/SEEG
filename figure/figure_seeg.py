#!/usr/bin/Python
'''
author: Alex
function:
'''
from util import *


def seeg_visualization(path):
    data = np.load(path)
    plt.imshow(data)
    plt.show()


def seeg_plot(path):
    data_raw = read_raw(path)
    data_raw.plot(duration=30)


path_preseizure = '../data/seeg/train/pre_zeizure/9a0f3d18-7870-11e9-bc1a-5187a394df03-0.npy'
path_sleep_normal = '../data/seeg/train/sleep_normal/2b9ed7ac-7637-11e9-aa98-2919e0abbf44-2.npy'
seeg_visualization(path_preseizure)
seeg_visualization(path_sleep_normal)

# path_preseizure_raw = '../data/raw_data/LK_Pre_seizure/LK_SZ1_pre_seizure_raw.fif'
# path_sleep_raw = '../data/raw_data/LK_SLEEP/LK_Sleep_Aug_4th_2am_seeg_raw-0.fif'
# seeg_plot(path_preseizure_raw)
# seeg_plot(path_sleep_raw)