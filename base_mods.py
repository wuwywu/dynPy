# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/13
# User      : WuY
# File      : base_mods.py
# 文件中包含：
# 1、节点动力学基础模块
# 3、突触动力学基础模块
# 2、数值模拟算法


import os
import sys
import copy
import numpy as np
from numba import njit, prange


# ================================= 神经元模型的基类 =================================
class Neurons:
    def __init__(self, N, method="euler", dt=0.01):
        self.num = N  # 神经元数量
        self.dt = dt
        self.method = method
        method_options = ["euler", "rk4"]
        if method not in method_options:
            raise ValueError(f"无效选择，method在{method_options}选择")
        if method == "euler":   self.method = Euler
        if method == "rk4":   self.method = RK4 

    def __call__(self, Io=0, axis=[0]):
        """
        args:
            Io: 输入到神经元模型的外部激励，
                shape:
                    (len(axis), self.num)
                    (self.num, )
                    float
            axis: 需要加上外部激励的维度
                list
        """
        # I = np.zeros((self.N_vars, self.num))
        # I[0, :] = self.Iex  # 恒定的外部激励
        # I[axis, :] += Io
        # params_list = list(self.params_nodes.values())
        # self.method(model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        self.t += self.dt  # 时间前进




# ================================= 演算法 =================================
@njit
def Euler(fun, x0, t, dt, *args):
    """
    使用 euler 方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        t: 运行时间
        x0: 上一个时间单位的状态变量
        dt: 时间步长
    :return: 
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    # 计算下一个时间单位的状态变量
    x0 += dt * fun(x0, t, *args)
    return x0

@njit
def RK4(fun, x0, t, dt, *args):
    """
    使用 Runge-Kutta 方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        t: 运行时间
        x0: 上一个时间单位的状态变量
        dt: 时间步长
    :return:
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    k1 = fun(x0, t, *args)
    k2 = fun(x0 + (dt / 2.) * k1, t + (dt / 2.), *args)
    k3 = fun(x0 + (dt / 2.) * k2, t + (dt / 2.), *args)
    k4 = fun(x0 + dt * k3, t + dt, *args)

    x0 += (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x0

