#!/usr/bin/python
import numpy as np


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

# if __name__ == '__main__':
#     test_1()
