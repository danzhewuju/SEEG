#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/12/23 21:00
# @Author  : Alex
# @Site    : 
# @File    : work_station.py
# @Software: PyCharm

from util.seeg_utils import *
from util.util_file import *
import numpy as np
if __name__ == '__main__':
    path = "/home/cbd109-3/Users/data/yh/Program/Python/SEEG/data/raw_data/BDP/BDP_Pre_seizure/BDP_SZ2_pre_seizure_raw.fif"
    data = read_raw(path)
    time_length = get_recorder_time(data) / 2
    print(time_length)

