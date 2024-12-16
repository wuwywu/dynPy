# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/12/16
# User      : WuY
# File      : DLS.py
# refs  : Y. Wu, Q. Ding, W. Huang, T. Li, D. Yu, Y. Jia, Dynamic learning of synchronization in coupled nonlinear systems, Nonlinear Dyn. 112(24), 21945-21967 (2024).
# Y. Wu, Q. Ding, W. Huang, X. Hu, Z. Ye, Y. Jia, Dynamic modulation of external excitation enhance synchronization in complex neuronal network, Chaos Soliton. Fract. 183, 114896 (2024).
# Z. Ye, Y. Wu, Q. Ding, Y. Xie, Y. Jia, Finding synchronization state of higher-order motif networks by dynamic learning, Phys. Rev. Res. 6, 033071 (2024).


import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random

# np.random.seed(2024)
# random.seed(2024)


class DLS:
    """
    dynamic learning of synchronization (DLS) algorithm

    args:
        N :     总的节点数
        local : 需要调整的状态变量的位置
        alpha : 使用 DLS 的学习率参数
    """
    def __init__(self, N=1, local=[1, 2], alpha=0.1):
        self.N = N        # 需要学习的参量数
        self.local = local  # 需要调整的状态变量的位置
        self.alpha = alpha  # 使用 DLS 的学习率参数
        # 存储每个local元素对应的单位矩阵对角线乘以alpha
        self.P = np.full((len(local), N), alpha)

    def train(self, w, factor, input, self_y=None, dt=0.01):
        """
        dx/dt = w*factor + f(x)
        input = x_(t+1)

        用来训练你想要修正的值,使得状态变量同步
        args:
            w : 需要更新的参数：如权重或设定的需要修正的值 (N_节点, N_节点/N_自定义)
            factor : 与 w 相乘的量    (N_节点, N_节点/N_自定义)
            input : 时刻 t+1 的状态变量, 在给出其他量后   (N_节点,)
            self_y : 自定义输入的值，与这个值同步, float
            dt  :   积分步长
        """
        # 外部因素的输入值  (N_节点, N_节点/N_自定义)
        factor_dt = factor * dt  # (N_节点, N_节点/N_自定义)

        if self_y is not None:
            yMean = self_y  # 监督学习
        else:
            yMean = input[self.local].mean()

        # 最小二乘法差值(N_节点,)
        error_input = input - yMean

        DLS_jit(w, factor_dt, error_input, self.local, self.P)

    def reset(self):
        self.P = np.full((len(self.local), self.N), self.alpha)



# ==================  并行版 动态学习同步算法 ==================
@njit
def DLS_jit(w, factor_dt, error_input, local, P):
    """
    dx/dt = w*factor + f(x)
    input = x_(t+1)
    factor_dt : factor * dt

    用来训练你想要修正的值,使得状态变量同步
    args:
        w               : 权重或设定的需要修正的值 (N_节点, N_节点/N_自定义)
        factor_dt       : 外部因素的输入值(N_节点, N_外部输入)
        error_input     : 最小二乘法差值(N_节点,)
        local           : 需要调整的状态变量的位置
        P               : 存储每个local元素对应的单位矩阵对角线乘以alpha
    """
    local_factor_dt = factor_dt[local]     # 形状是 (len(self.local), N)
    local_error_input = error_input[local]     # 形状是 (len(self.local),)

    # 计算 Prs（仅需要对角线与输入相乘）
    Prs = P * local_factor_dt  # 直接逐元素相乘

    # 计算 a 的向量化版本
    as_ = 1. / (1. + np.sum(local_factor_dt * Prs, axis=1))

    # 更新 Ps，只更新对角线部分
    P_updates = as_[:, np.newaxis] * (Prs ** 2)
    P -= P_updates

    # 更新权重 w，使用高级索引和广播去除for循环
    delta_w = (as_ * local_error_input)[:, np.newaxis] * Prs

    # 使用广播直接更新 w
    w[local] -= delta_w


