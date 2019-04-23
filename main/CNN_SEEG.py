#!/usr/bin/python
from Util import *

file_map = get_all_file_path()
# print(file_map['SGH'])
path_LK_Seizure = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data processed/LK/LK_Seizure01_9min6Sec_seeg_raw.fif"
path_SGH_Seizure_1 = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data processed/SGH/SGH_Seizure01_19min9Sec_seeg_raw-1.fif"
path_SGH_Seizure = "/home/cbd109-2/Users/yh/Program/Python/tmp/SEEG/data/data processed/SGH/SGH_Seizure01_19min9Sec_seeg_raw.fif"

raw_LK = read_raw(path_LK_Seizure)
raw_SGH_1 = read_raw(path_SGH_Seizure_1)
raw_SGH = read_raw(path_SGH_Seizure)
plt.figure()
raw_LK.plot()
plt.show()
LK_channels_names = get_channels_names(raw_LK)
SGH_channels_names = get_channels_names(raw_SGH)
print(get_channels_names(raw_LK))
print(get_channels_names(raw_SGH_1))
common_channels = get_common_channels(LK_channels_names, SGH_channels_names)
print(common_channels)
print(max(raw_LK.times)/60)
raw_SGH = data_connection(raw_SGH, raw_SGH_1)
print(max(raw_SGH.times)/60)
