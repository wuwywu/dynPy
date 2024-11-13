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
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


# ================================= 神经元模型的基类 =================================
class Neurons:
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，（"euler", "rk4"）
        dt : 计算步长
        spiking : 是否计算神经元的放电（True, False）

        params_f (dict): 节点模型参数
        
        t (float): 模拟的理论时间
    """
    def __init__(self, N, method="euler", dt=0.01, spiking=True):
        self.N = N  # 神经元数量
        self.dt = dt
        self.method = method
        method_options = ["euler", "rk4"]
        if method not in method_options:
            raise ValueError(f"无效选择，method在{method_options}选择")
        if method == "euler":   self.method = Euler
        if method == "rk4":   self.method = RK4
        if spiking:  
            self.spiking = spiking
            self._spikes_eval = spikes_eval
        self._params_f()
        self._vars_f()

    def _params_f(self):
        self.th_up = 0      # 放电阈上值
        self.th_down = -10  # 放电阈下值

    def _vars_f(self):
        self.t = 0  # 运行时间
        # 模型放电变量
        self.flag = np.zeros(self.N, dtype=int)           # 模型放电标志(>0, 放电)
        self.flaglaunch = np.zeros(self.N, dtype=int)     # 模型开始放电标志(==1, 放电刚刚开始)
        self.firingTime = np.zeros(self.N)                # 记录放电时间(上次放电)

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
        # I = np.zeros((self.N_vars, self.N))
        # I[0, :] = self.Iex  # 恒定的外部激励
        # I[axis, :] += Io
        # params_list = list(self.params_nodes.values())
        # self.method(model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        # if self.spiking: 
        #     self._spikes_eval(self.vars_nodes[0], self.params_f, self.flag, self.flaglaunch, self.firingTime)  # 放电测算

        self.t += self.dt  # 时间前进

@njit
def spikes_eval(mem, t, th_up, th_down, flag, flaglaunch, firingTime):
    """
        在非人工神经元中，计算神经元的 spiking
    """
    # -------------------- 放电开始 --------------------
    flaglaunch[:] = 0                                           # 重置放电开启标志
    firing_StartPlace = np.where((mem > th_up) & (flag == 0))   # 放电开始的位置
    flag[firing_StartPlace] = 1                                 # 放电标志改为放电
    flaglaunch[firing_StartPlace] = 1                           # 放电开启标志
    firingTime[firing_StartPlace] = t                           # 记录放电时间

    #  -------------------- 放电结束 -------------------
    firing_endPlace = np.where((mem < th_down) & (flag == 1))   # 放电结束的位置
    flag[firing_endPlace] = 0                                   # 放电标志改为放电


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

