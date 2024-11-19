# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/17
# User      : WuY
# File      : Rulkov.py
# Rulkov map 模型
# refernce : N.F. Rulkov, Regularization of Synchronized Chaotic Bursts. Phys. Rev. Lett. 86(1), 183-186 (2001).
# description : 一种混沌簇放电离散的神经元模型
# 注意：初始化时间要设置足够

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random
from base_mods import Neurons

# np.random.seed(2024)
# random.seed(2024)


@njit
def Rulkov_model(vars, t, I, params):
    res = np.zeros_like(vars)
    # 常数参数
    alpha, beta, sigma, I_ext = params
    # 状态变量
    x, y = vars

    x_new = alpha/(1+x**2) + y + I[0]
    y_new = y - sigma*x - beta + I[1]

    res[0] = x_new
    res[1] = y_new

    return res


class Rulkov(Neurons):
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，(""discrete")
        dt : 计算步长(dt=1)
        spiking : 是否计算神经元的放电（True, False）

        params_nodes (dict): 节点模型参数
        vars_nodes (numpy.ndarray): 节点模型状态变量
        t (float): 模拟的理论时间
    """
    def __init__(self, N, method="discrete", dt=1):
        super().__init__(N, method, dt)
        # self.N = N  # 神经元数量
        # self.dt = dt
        # self.method = method
        self._params()
        self._vars()

    def _params(self):
        # 常数参数
        self.params_nodes = {
            "alpha": 4.3,
            "beta": .001,
            "sigma": .001,
            "Iex": 0.,
        }
        self.th_up = .5         # 放电阈上值
        self.th_down = .5       # 放电阈下值

    def _vars(self):
        self.t = 0.  # 运行时间
        # 模型变量的初始值
        self.x0 = np.random.rand(self.N)
        self.y0 = np.random.rand(self.N)
        self.vars_nodes = np.vstack((self.x0, self.y0))

        self.N_vars = 2 # 变量的数量

    def __call__(self, Io=0, axis=[0]):
        """
        args:
            Io: 输入到神经元模型的外部激励，
                shape:
                    (len(axis), self.N)
                    (self.N, )
                    float
            axis: 需要加上外部激励的维度
                list
        """
        I = np.zeros((self.N_vars, self.N))    
        I[axis, :] += Io

        params_list = list(self.params_nodes.values())
        self.method(Rulkov_model, self.vars_nodes, self.t, self.dt, I, params_list)  #
        
        if self.spiking: 
            self._spikes_eval(self.vars_nodes[0], self.t, self.th_up, self.th_down, self.flag, self.flaglaunch, self.firingTime)  # 放电测算

            if self.record_spike_times:
                # 调用单独的记录峰值时间的函数
                self._record_spike_times(self.flaglaunch, self.t, self.spike_times, self.spike_counts, self.max_spikes)

        self.t += 1  # 时间前进


if __name__ == "__main__": 
    N = 2
    nodes = Rulkov(N=N)  # , temperature=6.3
    nodes.params_nodes["alpha"] = 4.1
    # nodes.set_vars_vals([0])
    # print(nodes.vars_nodes)

    time = []
    mem = []

    for i in range(3000):
        nodes()

    nodes.record_spike_times = True
    for i in range(2000):
        nodes()
        time.append(nodes.t)
        mem.append(nodes.vars_nodes[0].copy())
    
    valid_spike_times = nodes.return_spike_times()
    # print(valid_spike_times)
    # print(nodes.cal_isi())
    print(nodes.cal_cv())

    ax1 = plt.subplot(211)
    plt.plot(time, mem)
    plt.subplot(212, sharex=ax1)
    plt.eventplot(valid_spike_times)

    plt.show()
