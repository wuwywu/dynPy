# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/24
# User      : WuY
# File      : ML.py
# Morris–Lecar(ML) 模型

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
def ML_model(vars, t, I, params):
    """
        Morris-Lecar(ML) 模型
    """
    res = np.zeros_like(vars)
    # 状态变量
    v, w = vars
    # 常数参数
    gK, gL, gCa, EK, EL, ECa, Cm, V1, V2, V3, V4, phi, I_ext = params

    # 模型方程
    m_inf = 0.5 * (1. + np.tanh((v - V1) / V2))
    w_inf = 0.5 * (1. + np.tanh((v - V3) / V4))
    tau_w = 1. / np.cosh((v - V3) / (2 * V4))
    # 变量的导数
    dv_dt = (I_ext - gL * (v - EL) - gK * w * (v - EK)  - gCa * m_inf * (v - ECa) + I[0]) / Cm
    dw_dt = phi * (w_inf - w) / tau_w + I[1]

    # 输出结果
    res[0] = dv_dt
    res[1] = dw_dt

    return res

class ML(Neurons):
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
        # self.method = method
        self._params()
        self._vars()

    def _params(self):
        self.params_nodes = {
            "gK": 8.,     # K+通道的最大电导 (mS/cm^2)
            "gL": 2.,     # L通道的最大电导 (mS/cm^2)
            "gCa": 4.4,   # Ca++通道的最大电导 (mS/cm^2)
            "EK": -84.,   # K+通道的反向电位 (mV)
            "EL": -60.,    # L通道的反向电位 (mV)
            "ECa": 120.,  # Ca++通道的反向电位 (mV)
            "Cm": 20.,    # 膜电容 (uF/cm^2)
            "V1": -1.2,   # Ca++通道电位阈值 (mV)
            "V2": 18.,    # Ca++通道电位阈值 (mV)
            "V3": 2.,     # K+通道电位阈值 (mV)
            "V4": 30.,    # K+通道电位阈值 (mV)
            "phi": 0.04,   # 门控变量的参数
            "Iex": 100.,  # 恒定的外部激励电流 (uA/cm^2)
        }
        self.th_up = 10.        # 放电阈上值
        self.th_down = 10.      # 放电阈下值

    def _vars(self):
        self.t = 0.  # 运行时间
        # 模型变量的初始值
        self.v0 = np.random.rand(self.N) * 100 - 50
        self.w0 = np.random.rand(self.N)
        self.vars_nodes = np.vstack((self.v0, self.w0))

        self.N_vars = 2  # 变量的数量

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
        self.method(ML_model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        if self.spiking:
            self._spikes_eval(self.vars_nodes[0], self.t, self.th_up, self.th_down, self.flag, self.flaglaunch, self.firingTime)  # 放电测算
            
            if self.record_spike_times:
                # 调用单独的记录峰值时间的函数
                self._record_spike_times(self.flaglaunch, self.t, self.spike_times, self.spike_counts, self.max_spikes)

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 2
    method = "euler"               # "rk4", "euler"
    nodes = ML(N=N, method=method)  #, temperature=6.3
    nodes.params_nodes["Iex"] = 100.
    # nodes.N = 3
    # nodes.set_vars_vals([0, 0])
    # print(nodes.vars_nodes)

    time = []
    mem = []
    
    for i in range(500_00):
        nodes()

    nodes.record_spike_times = True
    for i in range(1000_00):
        nodes()
        time.append(nodes.t)
        mem.append(nodes.vars_nodes[0].copy())

    valid_spike_times = nodes.return_spike_times()
    print(valid_spike_times)
    # print(nodes.cal_isi())
    print(nodes.cal_cv())

    kop_list = nodes.cal_kop(min_spikes=10)
    first_last_spk, last_first_spk = kop_list[-1]  
    print(kop_list[0])

    ax1 = plt.subplot(311)
    plt.plot(time, mem)
    plt.subplot(312, sharex=ax1)
    plt.eventplot(valid_spike_times)
    plt.subplot(313, sharex=ax1)
    plt.plot(np.linspace(first_last_spk, last_first_spk, kop_list[2].shape[1]), kop_list[2].T)

    plt.show()
