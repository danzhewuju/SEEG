#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/8/21 13:54
# @Author  : Alex
# @Site    : 
# @File    : Draw_SEEG.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
import numpy as np
from util.util_file import matrix_normalization

data_1 = np.load('./e076f2ea-2552-11ea-9699-e0d55e6ff654-0.npy')
data_1 = matrix_normalization(data_1)
data_2 = np.load('./e076f2ea-2552-11ea-9699-e0d55e6ff654-vae-0.npy')
htmp1 = np.load('./3037c2d1-1ccf-11ea-9699-e0d55e6ff654-0-loc-34-34-33-33-34.npy')
cord = np.load('./3037c2d1-1ccf-11ea-9699-e0d55e6ff654-0.npy')
# plt.imshow(data_1)
# plt.show()
htmp2 = []
for x in htmp1:
    yy = []
    for y in x:
        yy.append([y[2], y[1], y[0]])
    htmp2.append(yy)
htmp2 = np.array(htmp2)

fig = plt.figure()
gs = grd.GridSpec(2, 3, width_ratios=[10,1,10])  # 把fig划分成2x2网格
# plt.subplots_adjust(wspace =0.2, hspace =0)#调整子图间距

ax = plt.subplot(gs[0])  # 往左上网格填元素
ax.set_yticks([])
# ax.get_yaxis().set_visible(False)
# x_major_locator=MultipleLocator(2/200)
# ax.xaxis.set_major_locator(x_major_locator)
p = ax.imshow(data_1, interpolation='nearest')  # htmp2是热力图变量
ax.text(90, 165,'(a)')
xlabel = ['$0.0$', '$0.5$', '$1.0$', '$1.5$', '$2.0$']
plt.xticks(np.arange(0, 201, 50), xlabel)
plt.ylabel("Channel", labelpad=10)

ax1 = plt.subplot(gs[2])
ax1.text(90, 165, '(b)')
ax1.set_yticks([])
p = ax1.imshow(htmp2, interpolation='nearest')  # htmp2是热力图变量
xlabel = ['$0.0$', '$0.5$', '$1.0$', '$1.5$', '$2.0$']
plt.xticks(np.arange(0, 201, 50), xlabel)
plt.ylabel("Channel", labelpad=10)

ax2 = plt.subplot(gs[3])  # 设置左下网格画波形
ax2.text(90, 180, '(c)')
ax2.set_yticks([])
ax2 = ax2.imshow(data_2, interpolation='nearest')  # 设置左下网格画波形
xlabel = ['$0.0$', '$0.5$', '$1.0$', '$1.5$', '$2.0$']
plt.xticks(np.arange(0, 201, 50), xlabel)
plt.ylabel("Channel", labelpad=10)
plt.xlabel("Time [sec]")

# colorAx = plt.subplot(gs[3])
# cb = plt.colorbar(p, cax = colorAx)
# cb.set_label('RWU')
ax4 = plt.subplot(gs[5])  # 设置左下网格画波形

# plt.figure(figsize=(2, 2))
# ax4.set_yticks([])
ax4.plot(np.linspace(0, 2, 200), cord *1000000, 'k', lw=0.5)  # 波形数据
# xlabel = ['$0.0$', '$0.5$', '$1.0$', '$1.5$', '$2.0$']
# plt.xticks(np.arange(0, 201, 50), xlabel)
plt.ylabel("Voltage [uV]")
plt.xlabel("Time [sec]")
ax4.text(90,180, '(d)')

plt.savefig('heatmap.pdf')
plt.show()
