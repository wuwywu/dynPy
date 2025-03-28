# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/03/28
# User      : WuY
# File      : FHN_mag.py
# FitzHugh-Nagumo(FHN)+电磁  模型
# ref : Y. Xie, A novel memristive neuron model and its energy characteristics, Cogn Neurodyn 18 (2024) 1989–2001. https://doi.org/10.1007/s11571-024-10065-5.

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random
from base_mods import Neurons

# np.random.seed(2024)
# random.seed(2024)


@njit
def FHN_mag_model(vars, t, I, params):
    """
        FitzHugh-Nagumo(FHN)+电磁 模型
    """
    res = np.zeros_like(vars)
    # 状态变量
    v, w, phi = vars
    # 常数参数
    a, b, c, k_1, k_2, alpha, beta, u_0, A, omega = params

    # 模型方程
    rho = alpha + beta * phi**2
    u_s = u_0 + A * np.cos(omega * t)
    dv_dt = -rho * (v - u_s) - w + v - v*v*v/3 + I[0]
    dw_dt = c * (v + a - b * w) + I[1]
    dphi_dt = -k_1 * (v - u_s) - k_2 * phi + I[2]

    # 输出结果
    res[0] = dv_dt
    res[1] = dw_dt
    res[2] = dphi_dt

    return res


class FHN_mag(Neurons):
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，（"euler", "rk4"）
        dt : 计算步长

        params_nodes (dict): 节点模型参数
        vars_nodes (numpy.ndarray): 节点模型状态变量
        t (float): 模拟的理论时间
    """
    def __init__(self, N=1, method="euler", dt=0.01):
        super().__init__(N, method=method, dt=dt)
        # self.N = N  # 神经元数量
        # self.dt = dt
        # self._method = method
        self.model = FHN_mag_model  # 模型的微分方程
        self._params()
        self._vars()

    def _params(self):
        # 模型参数
        self.params_nodes = {
            "a": .3,
            "b": 1.,
            "c": .1,
            "k_1": 1.,
            "k_2": 1.8,
            "alpha": .1,
            "beta": .3,
            "u_0": .2,
            "A": .8,
            "omega": .9
        }
        self.th_up = 1    # 放电阈上值
        self.th_down = 1  # 放电阈下值

    def _vars(self):
        self.t = 0.  # 运行时间
        # 模型变量的初始值
        self.v0 = np.random.rand(self.N)
        self.w0 = np.random.rand(self.N)
        self.phi0 = np.random.rand(self.N)
        self.vars_nodes = np.vstack((self.v0, self.w0, self.phi0))

        self.N_vars = 3  # 变量的数量

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
        params_list = np.asarray(list(self.params_nodes.values()))
        self.method(self.model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        if self.spiking:
            self._spikes_eval(self.vars_nodes[0], self.t, self.th_up, self.th_down, self.flag, self.flaglaunch, self.firingTime)  # 放电测算

            if self.record_spike_times:
                # 调用单独的记录峰值时间的函数
                self._record_spike_times(self.flaglaunch, self.t, self.spike_times, self.spike_counts, self.max_spikes)

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 2
    method = "euler"               # "rk4", "euler"
    nodes = FHN_mag(N=N, method=method)  #
    nodes.params_nodes["omega"] = 0.9
    nodes.params_nodes["A"] = 0.8
    # nodes.N = 3
    # nodes.set_vars_vals([0, 0])
    # print(nodes.vars_nodes)

    time = []
    mem = []
    for i in range(100_00):
        nodes()

    nodes.record_spike_times = True
    for i in range(2000_00):
        nodes()
        time.append(nodes.t)
        mem.append(nodes.vars_nodes[0].copy())

    valid_spike_times = nodes.return_spike_times()
    print(valid_spike_times)
    # print(nodes.cal_isi())
    print(nodes.cal_cv())
    
    kop_list = nodes.cal_kop(min_spikes=3)
    first_last_spk, last_first_spk = kop_list[-1]  
    print(kop_list[0])

    ax1 = plt.subplot(311)
    plt.plot(time, mem)
    plt.subplot(312, sharex=ax1)
    plt.eventplot(valid_spike_times)
    plt.subplot(313, sharex=ax1)
    plt.plot(np.linspace(first_last_spk, last_first_spk, kop_list[2].shape[1]), kop_list[2].T)

    plt.show()
