#!/usr/bin/python
import numpy as np
from main.pre_Processing import *


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


def test_3():
    path_data = "../data/output_data/sleep_frame_eeg.npy"
    sleep_time_frame = read_sleep_date(path_data)
    length = len(sleep_time_frame)
    ratio = duration_time / length
    x = [ratio * x for x in range(length)]
    draw_plot(x, sleep_time_frame)


# 模板
def test_n():
    print("test_n finished!")
