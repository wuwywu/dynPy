# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/16
# User      : WuY
# File      : Iz.py
# Izhikevich(Iz) 模型

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
def Iz_model(vars, t, I, params):
    res = np.zeros_like(vars)
    # 常数参数
    a, b, c, d, thresh, I_ext = params
    # 状态变量
    V, u = vars

    dV_dt = 0.04 * V * V + 5 * V - u + 140 + I_ext + I[0]
    du_dt = a * (b * V - u) + I[1]

    # 输出结果
    res[0] = dV_dt
    res[1] = du_dt

    return res

@njit
def spikes_eval_Iz(var, t, thresh, flaglaunch, firingTime, c, d):
    """
        计算神经元是否发放脉冲
    """
    flaglaunch[:] = 0                                   # 重置放电开启标志
    firing_StartPlace = np.where(var[0] > thresh)
    flaglaunch[firing_StartPlace] = 1                   # 放电开启标志
    firingTime[firing_StartPlace] = t                   # 记录放电时间

    var[0] = var[0] * (1 - flaglaunch) + flaglaunch * c
    var[1] += flaglaunch * d


class Iz(Neurons):
    """
        Izhikevich 脉冲神经元
        reference： E.M. Izhikevich, Simple model of spiking neurons, IEEE Transactions on neural networks, 14(6), 1569-1572 (2003).
        v' = 0.04v^2 + 5v + 140 - u + I
        u' = a(bv-u)
        下面是将Izh离散化的写法
        if v>= thresh:
            v = c
            u = u + d

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
        """
            excitatory neurons: a=0.02, b=0.2, c=−65, d=8
            inhibitory neurons: a=0.02, b=0.25, c=−65, d=2.
        """
        self.params_nodes = {
            "a": 0.02,
            "b": 0.2,
            "c": -65.,
            "d": 8.,
            "threshold":30.,
            "Iex": 10.
        }

    def _vars(self):
        self.t = 0  # 运行时间
        # 模型
        self.mem = np.random.uniform(-.1, .1, self.N)
        self.u = np.random.rand(self.N)
        self.vars_nodes = np.vstack((self.mem, self.u))

        self.N_vars = 2  # 变量的数量

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
        I = np.zeros((self.N_vars, self.N))
        I[axis, :] += Io
        params_list = list(self.params_nodes.values())
        self.method(Iz_model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        thresh, c, d = self.params_nodes["threshold"], self.params_nodes["c"], self.params_nodes["d"]
        spikes_eval_Iz(self.vars_nodes, self.t, thresh, self.flaglaunch, self.firingTime, c, d)
        if self.record_spike_times:
            # 调用单独的记录峰值时间的函数
            self._record_spike_times(self.flaglaunch, self.t, self.spike_times, self.spike_counts, self.max_spikes)

        self.t += self.dt  # 时间前进
        

if __name__ == "__main__":
    N = 2
    method = "euler"               # "rk4", "euler"
    nodes = Iz(N=N, method=method)  # , temperature=6.3
    nodes.params_nodes["Iex"] = 20.
    # nodes.set_vars_vals([0])
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
