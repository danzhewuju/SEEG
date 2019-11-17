import re
import sys

sys.path.append("../")
import json
import os
import re

path = "./log/BDP/preseizure/"
data_list = os.listdir(path)
print(data_list)
for p in data_list:
    if "preseizure" in p:
        print("True")
    else:
        print("false")
