# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/21
# User      : WuY
# File      : HR.py
# Hindmarsh-Rose(HR) 模型
# reference : J.L. Hindmarsh, R.M. Rose, A model of neuronal bursting using three coupled first order differential equations, Proc.R. Soc. Lond. Ser. B 221(1222), 87-102 (1984).

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random
from base_mods import Neurons

# seed = 0
# np.random.seed(seed)                # 给numpy设置随机种子

@njit
def HR_model(vars, t, I, params):
    res = np.zeros_like(vars)
    # 常数参数
    a, b, c, d, r, s, xR, I_ext = params
    # 状态变量
    x, y, z = vars

    # 变量的导数
    dx_dt = y - a * x ** 3 + b * x ** 2 - z + I_ext + I[0]
    dy_dt = c - d * x ** 2 - y + I[1]
    dz_dt = r*(s * (x - xR) - z) + I[2]

    # 输出结果
    res[0] = dx_dt
    res[1] = dy_dt
    res[2] = dz_dt

    return res

class HR(Neurons):
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，("euler", "heun", "rk4")
        dt : 计算步长
        temperature: 温度(℃)

        params_nodes (dict): 节点模型参数
        vars_nodes (numpy.ndarray): 节点模型状态变量
        t (float): 模拟的理论时间
    """
    def __init__(self, N=1, method="euler", dt=0.01):
        super().__init__(N, method, dt)
        # self.N = N  # 神经元数量
        # self.dt = dt
        # self.method = method
        self.model = HR_model  # 模型的微分方程
        self._params()
        self._vars()

    def _params(self):
        # Parameters for the Hindmarsh-Rose model
        # 混沌簇放电 (a =1.;b =3.;c =1.;d =5.;s =4;r =0.006;xR =-1.6;Iex =3.)
        self.params_nodes = {
            "a" : 1.,
            "b" : 3.,
            "c" : 1.,
            "d" : 5.,
            "r" : 0.006,
            "s" : 4.,
            "xR" : -1.6,
            "Iex": 3.2,         # 恒定的外部激励
            } 
        self.th_up = 1.         # 放电阈上值
        self.th_down = 1.       # 放电阈下值

    def _vars(self):
        self.t = 0.  # 运行时间
        # 模型变量
        self.x0 = np.random.rand(self.N) - 1.5  # Membrane potential variable, initialized randomly
        self.y0 = np.random.rand(self.N) - 10.  # Recovery variable
        self.z0 = np.random.rand(self.N) - 0.5  # Adaptation variable
        self.vars_nodes = np.vstack((self.x0, self.y0, self.z0))

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
        self.method(HR_model, self.vars_nodes, self.t, self.dt, I, params_list)

        if self.spiking: 
            self._spikes_eval(self.vars_nodes[0], self.t, self.th_up, self.th_down, self.flag, self.flaglaunch, self.firingTime)  # 放电测算

            if self.record_spike_times:
                # 调用单独的记录峰值时间的函数
                self._record_spike_times(self.flaglaunch, self.t, self.spike_times, self.spike_counts, self.max_spikes)

        self.t += self.dt  # Time step forward


if __name__ == "__main__":

    N = 2
    dt = 0.01
    method = "rk4"  # "rk4", "euler", "heun"
    nodes = HR(N=N, method=method, dt=dt)
    nodes.set_vars_vals([0.1, 0.1, 0.1])
    # print(nodes.vars_nodes)
    nodes.params_nodes["Iex"] = 3.2

    for i in range(200_00):
        nodes()

    time = []
    mem = []

    for i in range(200_00):
        nodes()
        time.append(nodes.t)
        mem.append(nodes.vars_nodes[0].copy())

    nodes.vars_nodes[0, 0] += 1e-2

    nodes.record_spike_times = True
    for i in range(1000_00):
        nodes()
        time.append(nodes.t)
        mem.append(nodes.vars_nodes[0].copy())

    valid_spike_times = nodes.return_spike_times()
    # print(valid_spike_times)
    # print(nodes.cal_isi())
    # print(nodes.cal_cv())

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

