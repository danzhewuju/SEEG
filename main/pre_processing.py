# /usr/bin/Python
'''
主要是数据的划分问题，将数据划分为癫痫发作前和正常睡眠

'''

from util import *

warning_time = 300  # 设计的预警时间为300


def get_duration_data(raw_path, name, save_dir, start, end_times, gap_time=30):  # 用来获取癫痫发作前的睡眠数据
    '''

    :param raw_path: 原始数据的路径
    :param name: 保存的名称
    :param save_dir:
    :param start:
    :param end_times:
    :param gap_time:
    :return:
    '''
    if end_times - gap_time > start:
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


def os_mkdir(save_dir, dir):  # mkdir some dir
    new_path = os.path.join(save_dir, dir)
    if os.path.exists(new_path) is not True:
        os.makedirs(new_path)
        print("new dir has been created! {}".format(new_path))
    else:
        print("{} dir is existed!".format(new_path))


def multi_class_handle():
    '''

        将癫痫的发作前的睡眠状态进行了细分
        对于多分类的数据进行预处理， 主要是将癫痫发作前的睡眠状态进行细分
    '''
    start = 0
    save_dir = "../data/raw_data/Pre_seizure"

    within_warning_time_dir = "within_warning_time"
    before_warning_time_dir = "before_warning_time"
    os_mkdir(save_dir, within_warning_time_dir)
    os_mkdir(save_dir, before_warning_time_dir)
    save_dir_wwt = os.path.join(save_dir, within_warning_time_dir)
    save_dir_bwt = os.path.join(save_dir, before_warning_time_dir)

    # 对应相关数据的目录

    raw_path = "../data/raw_data/LK_SZ/LK_SZ1_seeg_raw.fif"
    within_warning_name = "LK_SZ1_pre_seizure_within_warning_time"
    end_time = 546
    before_warning_name = "LK_SZ1_pre_seizure_before_warning_time"
    before_warning_end_time = end_time - warning_time

    get_duration_data(raw_path, before_warning_name, save_dir_bwt, start, before_warning_end_time, gap_time=60)
    get_duration_data(raw_path, within_warning_name, save_dir_wwt, before_warning_end_time, end_time, gap_time=30)

    raw_path = "../data/raw_data/LK_SZ/LK_SZ2_seeg_raw.fif"
    within_warning_name = "LK_SZ2_pre_seizure_within_warning_time"
    end_time = 564
    before_warning_name = "LK_SZ2_pre_seizure_before_warning_time"
    before_warning_end_time = end_time - warning_time

    get_duration_data(raw_path, before_warning_name, save_dir_bwt, start, before_warning_end_time, gap_time=60)
    get_duration_data(raw_path, within_warning_name, save_dir_wwt, before_warning_end_time, end_time, gap_time=30)

    raw_path = "../data/raw_data/LK_SZ/LK_SZ3_seeg_raw.fif"
    within_warning_name = "LK_SZ3_pre_seizure_within_warning_time"
    end_time = 733
    before_warning_name = "LK_SZ3_pre_seizure_before_warning_time"
    before_warning_end_time = end_time - warning_time

    get_duration_data(raw_path, before_warning_name, save_dir_bwt, start, before_warning_end_time, gap_time=60)
    get_duration_data(raw_path, within_warning_name, save_dir_wwt, before_warning_end_time, end_time, gap_time=30)

    raw_path = "../data/raw_data/LK_SZ/LK_SZ4_seeg_raw.fif"
    within_warning_name = "LK_SZ4_pre_seizure_within_warning_time"
    end_time = 995
    before_warning_name = "LK_SZ4_pre_seizure_before_warning_time"
    before_warning_end_time = end_time - warning_time

    get_duration_data(raw_path, before_warning_name, save_dir_bwt, start, before_warning_end_time, gap_time=60)
    get_duration_data(raw_path, within_warning_name, save_dir_wwt, before_warning_end_time, end_time, gap_time=30)

    raw_path = "../data/raw_data/LK_SZ/LK_SZ5_seeg_raw.fif"
    within_warning_name = "LK_SZ5_pre_seizure_within_warning_time"
    end_time = 1535
    before_warning_name = "LK_SZ5_pre_seizure_before_warning_time"
    before_warning_end_time = end_time - warning_time

    get_duration_data(raw_path, before_warning_name, save_dir_bwt, start, before_warning_end_time, gap_time=60)
    get_duration_data(raw_path, within_warning_name, save_dir_wwt, before_warning_end_time, end_time, gap_time=30)

    raw_path = "../data/raw_data/LK_SZ/LK_SZ6_seeg_raw.fif"
    within_warning_name = "LK_SZ6_pre_seizure_within_warning_time"
    end_time = 702
    before_warning_name = "LK_SZ6_pre_seizure_before_warning_time"
    before_warning_end_time = end_time - warning_time

    get_duration_data(raw_path, before_warning_name, save_dir_bwt, start, before_warning_end_time, gap_time=30)
    get_duration_data(raw_path, within_warning_name, save_dir_wwt, before_warning_end_time, end_time, gap_time=30)

    return True


def bi_class_handle():
    '''
        对于二分类的数据进行预处理
    '''
    start = 0
    save_dir = "../data/raw_data/LK_Pre_seizure"

    # 对应相关数据的目录

    raw_path = "../data/raw_data/LK_SZ/LK_SZ1_seeg_raw.fif"
    end_time = 546
    name = "LK_SZ1_pre_seizure"
    get_duration_data(raw_path, name, save_dir, start, end_time, gap_time=30)

    raw_path = "../data/raw_data/LK_SZ/LK_SZ2_seeg_raw.fif"
    end_time = 564
    name = "LK_SZ2_pre_seizure"
    get_duration_data(raw_path, name, save_dir, start, end_time, gap_time=30)

    raw_path = "../data/raw_data/LK_SZ/LK_SZ3_seeg_raw.fif"
    end_time = 733
    name = "LK_SZ3_pre_seizure"
    get_duration_data(raw_path, name, save_dir, start, end_time, gap_time=30)

    raw_path = "../data/raw_data/LK_SZ/LK_SZ4_seeg_raw.fif"
    end_time = 995
    name = "LK_SZ4_pre_seizure"
    get_duration_data(raw_path, name, save_dir, start, end_time, gap_time=30)

    raw_path = "../data/raw_data/LK_SZ/LK_SZ5_seeg_raw.fif"
    end_time = 1535
    name = "LK_SZ5_pre_seizure"
    get_duration_data(raw_path, name, save_dir, start, end_time, gap_time=30)

    raw_path = "../data/raw_data/LK_SZ/LK_SZ6_seeg_raw.fif"
    end_time = 702
    name = "LK_SZ6_pre_seizure"
    get_duration_data(raw_path, name, save_dir, start, end_time, gap_time=30)


if __name__ == '__main__':
    bi_class_handle()
