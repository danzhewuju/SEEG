#!/usr/bin/python3
import matplotlib.pyplot as plt


def draw_plot(x, y):
    plt.figure()
    plt.xlabel('s')
    plt.ylabel('uV')
    plt.plot(x, y)
    plt.show()
    return True
