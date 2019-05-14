# /usr/bin/Python

from util import *


def get_duration_data(raw_path, name, save_dir, start, end_times, gap_time=30):
    '''

    :param raw_path: 原始数据的路径
    :param name: 保存的名称
    :param save_dir:
    :param start:
    :param end_times:
    :param gap_time:
    :return:
    '''
    if end_time - gap_time > start:
        raw_data = read_raw(raw_path)
        channel_names = get_channels_names(raw_data)
        duration_data = get_duration_raw_data(raw_data, start, end_times - gap_time)
        if os.path.exists(save_dir) is not True:
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, name)
        save_path += '_raw.fif'

        rewrite(duration_data, channel_names, save_path)
        return duration_data
    else:
        print("时间区间不合理！！！")
        return None


if __name__ == '__main__':
    start = 0
    save_dir = "../data/raw_data/Pre_seizure"

    # 对应相关数据的目录
    # raw_path = "../data/raw_data/LK_SZ/LK_SZ1_seeg_raw.fif"
    # name = "LK_SZ1_pre_seizure"
    # end_time = 546

    # raw_path = "../data/raw_data/LK_SZ/LK_SZ2_seeg_raw.fif"
    # name = "LK_SZ2_pre_seizure"
    # end_time = 564

    # raw_path = "../data/raw_data/LK_SZ/LK_SZ3_seeg_raw.fif"
    # name = "LK_SZ3_pre_seizure"
    # end_time = 733

    # raw_path = "../data/raw_data/LK_SZ/LK_SZ4_seeg_raw.fif"
    # name = "LK_SZ4_pre_seizure"
    # end_time = 995

    raw_path = "../data/raw_data/LK_SZ/LK_SZ5_seeg_raw.fif"
    name = "LK_SZ5_pre_seizure"
    end_time = 1535

    # raw_path = "../data/raw_data/LK_SZ/LK_SZ6_seeg_raw.fif"
    # name = "LK_SZ6_pre_seizure"
    # end_time = 702

    get_duration_data(raw_path, name, save_dir, start, end_time, gap_time=30)
