# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/13
# User      : WuY
# File      : utils_f.py
# 一些工具

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


class spikevent:
    """
        神经元模型的峰值收集器
        N: 收集 N 个神经的尖峰事件
    """
    def __init__(self, N):
        self.N = N
        self.Tspike_list = [[] for i in range(N)]

    def __call__(self, t, flaglaunch):
        """
        t: 模拟实时时间
        flaglaunch: 是否尖峰的标志 (放电开启标志)
        """
        for i in range(self.N):
            if flaglaunch[i]>0.9:
                self.Tspike_list[i].append(t)

    def pltspikes(self):
        plt.eventplot(self.Tspike_list)


