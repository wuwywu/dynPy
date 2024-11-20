# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/20
# User      : WuY
# File      : Rossler.py
# Rössler system 模型
# refernce : O.E. Rössler, An equation for continuous chaos, Phys. Lett. A 57(5), 397-398 (1976).

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random
from base_mods import Nodes

# np.random.seed(2024)
# random.seed(2024)


@njit
def Rossler_model(vars, t, I, params):
    res = np.zeros_like(vars)
    # 常数参数
    alpha, beta, gamma = params
    # 状态变量
    x, y, z = vars

    # 变量的导数
    dx_dt = -y - z + I[0]
    dy_dt = x + alpha * y + I[1]
    dz_dt = beta + z * (x - gamma) + I[2]

    # 输出结果
    res[0] = dx_dt
    res[1] = dy_dt
    res[2] = dz_dt

    return res


class Rossler(Nodes):
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，("euler", "rk4")
        dt : 计算步长(dt=.01)

        params_nodes (dict): 节点模型参数
        vars_nodes (numpy.ndarray): 节点模型状态变量
        t (float): 模拟的理论时间
    """
    def __init__(self, N=1, method="euler", dt=.01):
        super().__init__(N, method, dt)
        # self.N = N  # 神经元数量
        # self.dt = dt
        # self.method = method
        self._params()
        self._vars()

    def _params(self):
        # 常数参数
        self.params_nodes = {
            "alpha": 0.2,
            "beta": 0.2,
            "gamma": 9.,
        }

    def _vars(self):
        self.t = 0.  # 运行时间
        # 模型变量的初始值
        self.x0 = np.random.rand(self.N)
        self.y0 = np.random.rand(self.N)
        self.z0 = np.random.rand(self.N)
        self.vars_nodes = np.vstack((self.x0, self.y0, self.z0))

        self.N_vars = 3  # 变量的数量

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
        self.method(Rossler_model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 2
    dt = 0.01
    method = "rk4"  # "rk4", "euler"
    nodes = Rossler(N=N, method=method, dt=dt)
    nodes.set_vars_vals([0.1, 0.1, 0.1])
    # print(nodes.vars_nodes)

    for i in range(200_00):
        nodes()

    nodes.vars_nodes[0, 0] += 1e-3

    time = []
    mem = []

    for i in range(200_00):
        nodes()
        time.append(nodes.t)
        mem.append(nodes.vars_nodes[0].copy())

    plt.plot(time, mem)

    plt.show()
