#!/usr/bin/python3

import mne
import numpy as np
import matplotlib.pyplot as plt


def read_raw(path):
    raw = mne.io.read_raw_fif(path, preload=True)
    return raw


def get_channels_names(raw):
    channel_names = raw.info['ch_names']
    return channel_names


def save_numpy_info(data, path):  # 存储numpy的数据
    np.save(path, data)
    print("Successfully save!")
    return True
