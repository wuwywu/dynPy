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
from base_mods import Neurons

# seed = 0
# np.random.seed(seed)                # 给numpy设置随机种子

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
        vars_nodes (numpy.ndarray): 节点模型变量
        t (float): 模拟的理论时间
    """
    def __init__(self, N=1, method="euler", dt=0.01, temperature=None):
        super().__init__(N, method=method, dt=dt)
        # self.num = N  # 神经元数量
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
        self.t = 0  # 运行时间
        # 模型变量的初始值
        self.v0 = np.random.uniform(-.3, .3, self.num)
        self.m0 = 1 * np.random.rand(self.num)
        self.h0 = 1 * np.random.rand(self.num)
        self.n0 = 1 * np.random.rand(self.num)
        self.vars_nodes = np.array([self.v0, self.m0, self.h0, self.n0])

        self.N_vars = 4  # 变量的数量

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
        Iex = self.params_nodes["Iex"]      # 恒定的外部激励
        I = np.zeros((self.N_vars, self.num))
        I[0, :] = Iex      
        I[axis, :] += Io
        params_list = list(self.params_nodes.values())
        self.method(HH_model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        self.t += self.dt  # 时间前进



if __name__ == "__main__": 
    N = 2
    method = "euler" # "rk4", "euler"
    nodes = HH(N=N, method=method, temperature=6.3)
    nodes.params_nodes["Iex"] = 6.3

    time = []
    mem = []

    for i in range(10000):
        nodes()
        time.append(nodes.t)
        mem.append(nodes.vars_nodes[0].copy())

    ax1 = plt.subplot()
    plt.plot(time, mem)

    plt.show()
