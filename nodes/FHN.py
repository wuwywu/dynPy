# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/24
# User      : WuY
# File      : FHN.py
# FitzHugh-Nagumo(FHN) 模型

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
def FHN_model(vars, t, I, params):
    """
        FitzHugh-Nagumo(FHN) 模型
    """
    res = np.zeros_like(vars)
    # 状态变量
    v, w = vars
    # 常数参数
    a, b, c, I_ext = params

    # 模型方程
    dv_dt = v - v**3 / 3 - w + I_ext + I[0]
    dw_dt = a * (v + c - b * w) + I[1]

    # 输出结果
    res[0] = dv_dt
    res[1] = dw_dt

    return res

@njit
def FHN_mag_model(vars, t, I, params):
    """
        FitzHugh-Nagumo(FHN)+电磁 模型
    """
    res = np.zeros_like(vars)
    # 状态变量
    v, w, phi = vars
    # 常数参数
    a, b, c, lamda, beta, k, k1, k2, phi_ext, Iex = params

    # 模型方程
    rho = lamda + 3. * beta * phi**2
    dv_dt = v * (v - a) * (1 - v) - w + k * v * rho + Iex + I[0]
    dw_dt = b * (v - c * w) + I[1]
    dphi_dt = k1 * v - k2 * phi + phi_ext + I[2]

    # 输出结果
    res[0] = dv_dt
    res[1] = dw_dt
    res[2] = dphi_dt

    return res


class FHN(Neurons):
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
        self.model = FHN_model  # 模型的微分方程
        self._params()
        self._vars()

    def _params(self):
        # 模型参数
        self.params_nodes = {
            "a": 0.08,
            "b": 0.8,
            "c": 0.7,
            "Iex": 1.,
        }
        self.th_up = 1    # 放电阈上值
        self.th_down = 1  # 放电阈下值

    def _vars(self):
        self.t = 0.  # 运行时间
        # 模型变量的初始值
        self.v0 = np.random.rand(self.N)
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
        self.method(FHN_model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        if self.spiking:
            self._spikes_eval(self.vars_nodes[0], self.t, self.th_up, self.th_down, self.flag, self.flaglaunch, self.firingTime)  # 放电测算

            if self.record_spike_times:
                # 调用单独的记录峰值时间的函数
                self._record_spike_times(self.flaglaunch, self.t, self.spike_times, self.spike_counts, self.max_spikes)

        self.t += self.dt  # 时间前进


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
            "a": 0.5,
            "b": 0.025,
            "c": 1.,
            "lamda": 0.1,
            "beta": 0.02,
            "k": 1.,
            "k1": 0.5,
            "k2": 0.9,
            "phi_ext": 2.4,
            "Iex": 0.,
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
        params_list = list(self.params_nodes.values())
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
    nodes = FHN(N=N, method=method)  #
    # nodes.params_nodes["Iex"] = 1.
    nodes = FHN_mag(N=N, method=method)  #
    nodes.params_nodes["Iex"] = 0.
    # nodes.N = 3
    # nodes.set_vars_vals([0, 0])
    # print(nodes.vars_nodes)

    time = []
    mem = []
    for i in range(100_00):
        nodes()

    nodes.record_spike_times = True
    for i in range(500_00):
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
