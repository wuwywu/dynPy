# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/15
# User      : WuY
# File      : statis_sync.py
# 用于同步研究的统计量


import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


# ================================= 同步因子 =================================
class synFactor:
    """
        计算同步因子  
        N: 需要计算变量的数量
        Tn: 计算次数(int), Time/dt

        描述：一个用于量化同步的归一化统计量（范围在0-1之间）。
                越趋近与1，同步越好；
                越趋近与0，同步越差。
    """
    def __init__(self, N, Tn):
        self.N = N      # 矩阵大小
        self.Tn = Tn    # 计算次数
        self.count = 0  # 统计计算次数

        # 初始化累加器
        self.up1 = 0.
        self.up2 = 0.
        self.down1 = np.zeros(N)
        self.down2 = np.zeros(N)

    def __call__(self, x):
        self.up1, self.up2, self.count = cal_synFactor(x, self.up1, self.up2, self.down1, self.down2, self.count, self.Tn)

    def return_syn(self):
        return return_synFactor(self.up1, self.up2, self.down1, self.down2, self.count, self.Tn)
    
@njit
def cal_synFactor(x, up1, up2, down1, down2, count, Tn):
    """
        计算同步因子
    """
    F = np.mean(x)
    up1 += F * F / Tn
    up2 += F / Tn
    down1 += x * x / Tn
    down2 += x / Tn
    count += 1  # 计算次数叠加W

    return up1, up2, count

@njit
def return_synFactor(up1, up2, down1, down2, count, Tn):
    """
        返回同步因子
    """
    if count != Tn:
        print(f"输入计算次数{Tn},实际计算次数{count}")

    down = np.mean(down1 - down2 ** 2)
    if down > -0.000001 and down < 0.000001:
        return 1.
    
    up = up1 - up2 ** 2

    return up / down



