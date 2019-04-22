#!/usr/bin/python3
import matplotlib.pyplot as plt


def draw_plot(x, y):
    plt.figure()
    plt.title('EEG channel EEG C#-Ref-1')
    plt.xlabel('s')
    plt.ylabel('uV')
    plt.plot(x, y)
    plt.show()
    return True
