# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/13
# User      : WuY
# File      : HH.py
# Hodgkin-Huxley(HH) 模型

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
def HH_model(vars, t, I, params):
    res = np.zeros_like(vars)
    # 常数参数
    g_Na, g_K, g_L, E_Na, E_K, E_L, C_m, I_ext, temperature = params

    # 状态变量
    V, m, h, n = vars

   # 计算 α 和 β 函数
    alpha_n = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    beta_n = 0.125 * np.exp(-(V + 65) / 80)

    alpha_m = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta_m = 4 * np.exp(-(V + 65) / 18)

    alpha_h = 0.07 * np.exp(-(V + 65) / 20)
    beta_h = 1 / (1 + np.exp(-(V + 35) / 10))

    # 门控变量的导数
    dm_dt = alpha_m * (1 - m) - beta_m * m + I[1]
    dh_dt = alpha_h * (1 - h) - beta_h * h + I[2]
    dn_dt = alpha_n * (1 - n) - beta_n * n + I[3]

    phi = 3.0 ** ((temperature - 6.3) / 10)  # 温度系数
    dm_dt *= phi
    dn_dt *= phi
    dh_dt *= phi

    # 离子电流
    I_Na = g_Na * m**3 * h * (V - E_Na)
    I_K = g_K * n**4 * (V - E_K)
    I_L = g_L * (V - E_L)

    # 膜电位的导数
    dV_dt = (I_ext - I_Na - I_K - I_L + I[0]) / C_m

    # 输出结果
    res[0] = dV_dt
    res[1] = dm_dt
    res[2] = dh_dt
    res[3] = dn_dt

    return res


class HH(Neurons):
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，（"euler", "rk4"）
        dt : 计算步长
        temperature: 温度(℃)

        params_nodes (dict): 节点模型参数
        vars_nodes (numpy.ndarray): 节点模型状态变量
        t (float): 模拟的理论时间
    """
    def __init__(self, N=1, method="euler", dt=0.01, temperature=6.3):
        super().__init__(N, method=method, dt=dt)
        # self.N = N  # 神经元数量
        # self.dt = dt
        # self.method = method
        self.temperature = temperature
        self._params()
        self._vars()

    def _params(self):
        self.params_nodes = {
            'g_Na': 120.,     # 钠离子通道的最大电导(mS/cm2)
            'g_K': 36.,       # 钾离子通道的最大电导(mS/cm2)
            'g_L': 0.3,      # 漏离子电导(mS/cm2)
            'E_Na': 50.,      # 钠离子的平衡电位(mV)
            'E_K': -77.,      # 钾离子的平衡电位(mV)
            'E_L': -54.4,    # 漏离子的平衡电位(mV)
            'Cm': 1.,       # 比膜电容(uF/cm2)
            'Iex': 10.,       # 恒定的外部激励电流(uA/cm2)
            "temperature": self.temperature,     # 标准温度(℃) 实验温度为6.3
        }   

    def _vars(self):
        self.t = 0.  # 运行时间
        # 模型变量的初始值
        self.v0 = np.random.uniform(-.3, .3, self.N)
        self.m0 = 1 * np.random.rand(self.N)
        self.h0 = 1 * np.random.rand(self.N)
        self.n0 = 1 * np.random.rand(self.N)
        self.vars_nodes = np.vstack((self.v0, self.m0, self.h0, self.n0))

        self.N_vars = 4  # 变量的数量

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
        self.method(HH_model, self.vars_nodes, self.t, self.dt, I, params_list)  #
        
        if self.spiking: 
            self._spikes_eval(self.vars_nodes[0], self.t, self.th_up, self.th_down, self.flag, self.flaglaunch, self.firingTime)  # 放电测算

            if self.record_spike_times:
                # 调用单独的记录峰值时间的函数
                self._record_spike_times(self.flaglaunch, self.t, self.spike_times, self.spike_counts, self.max_spikes)

        self.t += self.dt  # 时间前进


if __name__ == "__main__": 
    N = 2
    method = "euler"               # "rk4", "euler"
    nodes = HH(N=N, method=method)  # , temperature=6.3
    nodes.params_nodes["Iex"] = 10.
    # nodes.set_vars_vals([0])
    # print(nodes.vars_nodes)

    time = []
    mem = []
    for i in range(100_00):
            nodes()

    nodes.record_spike_times = True
    for i in range(200_00):
        nodes()
        time.append(nodes.t)
        mem.append(nodes.vars_nodes[0].copy())

    valid_spike_times = nodes.return_spike_times()
    # print(valid_spike_times)
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
