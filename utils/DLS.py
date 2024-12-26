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
        self.N = N          # 需要学习的参量数
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
        train_DLS(w, factor, input, self.local, self.P, dt=dt, self_y=self_y)

    def reset(self):
        self.P = np.full((len(self.local), self.N), self.alpha)

# w = w.clip(w_min, w_max) # 对权重进行裁剪

class DLS_ADMM:
    """
    dynamic learning of synchronization (DLS) algorithm
    加入了交替方向乘子法, 限定调节范围 (alternating direction method of multipliers (ADMM))

    args:
        N :     总的节点数
        local : 需要调整的状态变量的位置
        alpha : 使用 DLS 的学习率参数
        rho   : ADMM的惩罚参数
        use_admm : 使用ADMM的开关
        w_min : 权重的最小值约束
        w_max : 权重的最大值约束
    """
    def __init__(self, N=1, local=[1, 2], alpha=0.1, rho=0.1, use_admm=True, w_min=None, w_max=None):
        self.N = N          # 需要学习的参量数
        self.local = local  # 需要调整的状态变量的位置
        self.alpha = alpha  # 使用 DLS 的学习率参数
        self.rho = rho      # ADMM的惩罚参数
        self.w_min = w_min  # 权重的最小值约束
        self.w_max = w_max  # 权重的最大值约束
        self.use_admm = use_admm  # 是否使用ADMM

        # 存储每个local元素对应的单位矩阵对角线乘以alpha
        self.P = np.full((len(local), N), alpha)

        # 初始化ADMM的z和mu
        self.z = np.zeros((len(local), N))
        self.mu = np.zeros((len(local), N))

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
        train_DLS(w, factor, input, self.local, self.P, dt=dt, self_y=self_y)

        # 进行ADMM更新（如果使用ADMM）
        if self.use_admm:
            update_admm(w, self.z, self.mu, self.rho, self.local, self.w_min, self.w_max)

    def reset(self):
        self.P = np.full((len(self.local), self.N), self.alpha)


class DLS_ADMM_multiranges:
    """
    dynamic learning of synchronization (DLS) algorithm
    加入了交替方向乘子法, 限定调节范围 (alternating direction method of multipliers (ADMM))
    逐渐将权重逼近到给定的多个范围内，包括多个不连续的范围。

    args:
        N :     总的节点数
        local : 需要调整的状态变量的位置
        alpha : 使用 DLS 的学习率参数
        rho   : ADMM的惩罚参数
        use_admm : 使用ADMM的开关
        ranges : 给定的多个范围的集合，[(min1, max1), (min2, max2), ...]
    """
    def __init__(self, N=1, local=[1, 2], alpha=0.1, rho=0.1, use_admm=True, ranges=None):
        self.N = N          # 需要学习的参量数
        self.local = local  # 需要调整的状态变量的位置
        self.alpha = alpha  # 使用 DLS 的学习率参数
        self.rho = rho      # ADMM的惩罚参数
        self.use_admm = use_admm  # 是否使用ADMM

        # 定义范围限制
        self.ranges = np.asarray(ranges)

        # 存储每个local元素对应的单位矩阵对角线乘以alpha
        self.P = np.full((len(local), N), alpha)

        # 初始化ADMM的z和mu
        self.z = np.zeros((len(local), N))
        self.mu = np.zeros((len(local), N))

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
        train_DLS(w, factor, input, self.local, self.P, dt=dt, self_y=self_y)

        # 进行ADMM更新（如果使用ADMM）
        if self.use_admm:
            update_admm_multiranges(w, self.z, self.mu, self.rho, self.local, self.ranges)

    def reset(self):
        self.P = np.full((len(self.local), self.N), self.alpha)


# ==================  并行版 调用训练方法 ==================
@njit
def train_DLS(w, factor, input, local, P, dt=0.01, self_y=None):
    """
    dx/dt = w*factor + f(x)
    input = x_(t+1)

    用来训练你想要修正的值,使得状态变量同步
    args:
        w : 需要更新的参数：如权重或设定的需要修正的值 (N_节点, N_节点/N_自定义)
        factor : 与 w 相乘的量    (N_节点, N_节点/N_自定义)
        input : 时刻 t+1 的状态变量, 在给出其他量后   (N_节点,)
        local : 需要调整的状态变量的位置
        P : 存储每个local元素对应的单位矩阵对角线乘以alpha
        self_y : 自定义输入的值，与这个值同步, float
        dt  :   积分步长
    """
    # 外部因素的输入值  (N_节点, N_节点/N_自定义)
    factor_dt = factor * dt  # (N_节点, N_节点/N_自定义)

    if self_y is not None:
        yMean = self_y  # 监督学习
    else:
        yMean = input[local].mean()

    # 最小二乘法差值(N_节点,)
    error_input = input - yMean

    DLS_jit(w, factor_dt, error_input, local, P)


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

# 使用admm 限制单区域的权重
@njit
def update_admm(w, z, mu, rho, local, w_min=None, w_max=None):
    """
        更新 ADMM 中的辅助变量 z 和拉格朗日乘子 mu
        args:
            w   : 需要更新的参数：如权重或设定的需要修正的值 (N_节点, N_节点/N_自定义)
            z   : ADMM的辅助变量
            mu  : 拉格朗日乘子
            rho : ADMM的惩罚参数
            local : 需要调整的状态变量的位置
            w_min : 权重的最小值约束
            w_max : 权重的最大值约束
    """

    # 计算delta_w_admm
    delta_w_admm = rho * (z - w[local]) + mu / rho

    # 添加轻微的L2正则化项，防止权重过度调整
    # delta_w_admm += 1e-5 * w[self.local]

    # 更新权重w
    w[local] -= delta_w_admm

    # 更新辅助变量z
    z_new = w[local] + mu / rho

    # 应用权重约束
    if w_min is not None:
        z_new = np.maximum(z_new, w_min)
    if w_max is not None:
        z_new = np.minimum(z_new, w_max)

    # 更新 z 和 mu
    z[:] = z_new  # 更新z

    # 更新拉格朗日乘子mu
    mu += rho * (w[local] - z)


# 使用admm 限制多个区域的权重
@njit
def update_admm_multiranges(w, z, mu, rho, local, ranges):
        """
        更新 ADMM 中的辅助变量 z 和拉格朗日乘子 mu
        args:
            w   : 需要更新的参数：如权重或设定的需要修正的值 (N_节点, N_节点/N_自定义)
            z   : ADMM的辅助变量
            mu  : 拉格朗日乘子
            rho : ADMM的惩罚参数
            local : 需要调整的状态变量的位置
            ranges : 给定的多个范围的集合，np.array([(min1, max1), (min2, max2),...])
    """

        # 计算delta_w_admm
        delta_w_admm = rho * (z - w[local]) + mu / rho

        # 添加轻微的L2正则化项，防止权重过度调整
        # delta_w_admm += 1e-5 * w[self.local]

        # 更新权重w
        w[local] -= delta_w_admm

        # 更新辅助变量z
        z_new = w[local] + mu / rho

        # 对 z_new 的所有元素应用多个范围的约束
        z_new = apply_range_constraints(z_new, ranges)

        # 更新 z 和 mu
        z[:] = z_new  # 更新z

        # 更新拉格朗日乘子mu
        mu += rho * (w[local] - z)

@njit
def apply_range_constraints(z_new, ranges):
    """
    根据多个范围，对 z_new 进行调整，将其约束到最近的范围边界。

    ## 饱和边界约束，也可以直接对权重 w 使用
    """

    # 定义范围限制
    ranges = np.asarray(ranges)

    # 提取范围的最小值和最大值
    w_min = ranges[:, 0]  # 形状为 (N_ranges,)
    w_max = ranges[:, 1]  # 形状为 (N_ranges,)

    N_local, N_num = z_new.shape  # 获取 z_new 的形状
    N_ranges = ranges.shape[0]  # 获取范围的数量

    # 扩展 z_new 以便与所有范围的最小值和最大值进行广播比较
    z_expanded = z_new[:, :, np.newaxis]  # 形状为 (N_local, N_num, 1)

    # 判断哪些值在范围内
    in_range = np.logical_and(z_expanded >= w_min, z_expanded <= w_max)  # 判断 z_new 是否在范围内

    # 对 z_new 超出所有范围的情况，判断值是否不在任何范围内
    out_of_range = np.zeros((N_local, N_num), dtype=np.bool_)
    for i in range(N_local):
        for j in range(N_num):
            if np.all(~in_range[i, j, :]):
                out_of_range[i, j] = True

    # 计算距离最小值和最大值的差
    dist_to_min = np.abs(z_expanded - w_min)  # 形状为 (N_local, N_num, N_ranges)
    dist_to_max = np.abs(z_expanded - w_max)  # 形状为 (N_local, N_num, N_ranges)

    # 找到每个点最近的最小值和最大值
    nearest_min_idx = np.argmin(dist_to_min, axis=2)  # 形状为 (N_local, N_num)
    nearest_max_idx = np.argmin(dist_to_max, axis=2)  # 形状为 (N_local, N_num)

    # 选择最近的边界值
    z_adjusted = np.copy(z_new)

    for i in range(N_local):
        for j in range(N_num):
            if out_of_range[i, j]:
                if dist_to_min[i, j, nearest_min_idx[i, j]] < dist_to_max[i, j, nearest_max_idx[i, j]]:
                    z_adjusted[i, j] = w_min[nearest_min_idx[i, j]]
                else:
                    z_adjusted[i, j] = w_max[nearest_max_idx[i, j]]

    return z_adjusted

