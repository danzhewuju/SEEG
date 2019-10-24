import re
import sys

sys.path.append("../")
import json


def test_3():
    name = "BDP"
    data = json.load(open("./json_path/config.json"))
    path = data["handel.sequentially__path_BDP"].format(name)
    print(path)


if __name__ == '__main__':
    test_3()
