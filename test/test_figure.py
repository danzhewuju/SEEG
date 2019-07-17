#!/usr/bin/python
import matplotlib.pyplot as plt
import mne
import numpy as np
from util import *

'''
主要是用于图像的展示

'''


def test_1():
    path_lk0 = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data_slice/LK_label0_raw.fif'
    path_sgh0 = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data_slice/SGH_label0_raw.fif"

    path_lk1 = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data_slice/LK_label1_raw.fif'
    path_sgh1 = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data_slice/SGH_label1_raw.fif"
    save_normal = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data_slice/split/normal'
    save_cases = '/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data_slice/split/cases'

    raw_lk_normal = read_raw(path_lk0)
    raw_lk_normal.plot()
    raw_lk_cases = read_raw(path_lk1)
    raw_lk_cases.plot()
