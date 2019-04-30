#!/usr/bin/python
import pandas as pd


def get_common_channels_by_file(path="../data/seizure/common_channels.csv"): # 通过文件来返回公共的信道
    data = pd.read_csv(path, sep=',')
    d_list = data['channels']
    return d_list
