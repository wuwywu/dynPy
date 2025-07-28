# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/07/22
# User      : WuY
# File      : Hastings_Powell.py
# Hastings-Powell 模型
# description : 三物种混沌食物链模式，每个斑块中都有一个猎物、一个捕食者和一个超级捕食者物种。
# ref : A. Hastings, T. Powell, Chaos in a Three-Species Food Chain, Ecology 72(3), 896-903 (1991).


import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random
from base_mods import Nodes

# np.random.seed(2025)
# random.seed(2025)


@njit
def HP_model(vars, t, I, params):
    """
        HastingsPowell (HP) 模型
    """
    res = np.zeros_like(vars)
    # 状态变量
    x, y, z = vars
    # 常数参数
    a1, b1, a2, b2, d1, d2 = params

    # 模型方程
    f_1 = a1 * x / (1 + b1 * x)
    f_2 = a2 * y / (1 + b2 * y)
    dx_dt = x * (1 - x) - f_1 * y + I[0]
    dy_dt = f_1 * y - f_2 * z - d1 * y + I[1]
    dz_dt = f_2 * z - d2 * z + I[2]

    # 输出结果
    res[0] = dx_dt
    res[1] = dy_dt
    res[2] = dz_dt

    return res


class HP(Nodes):
    """
        N : 建立节点的数量
        method : 计算非线性微分方程的方法，("euler", "heun", "rk4", "rkf45")
        dt : 计算步长

        params_nodes (dict): 节点模型参数
        vars_nodes (numpy.ndarray): 节点模型状态变量
        t (float): 模拟的理论时间
    """
    def __init__(self, N=1, method="rk4", dt=0.01):
        super().__init__(N, method=method, dt=dt)
        # self.N = N  # 神经元数量
        # self.dt = dt
        # self._method = method
        self.model = HP_model  # 模型的微分方程
        self._params()
        self._vars()

    def _params(self):
        # 模型参数
        self.params_nodes = {
            "a1": 5.,
            "b1": 3.,
            "a2": 0.1,
            "b2": 2.,
            "d1": 0.4,
            "d2": 0.01,
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
            Io: 输入到节点模型的外部激励，
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
        self.method(HP_model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 2
    dt = 0.01
    method = "rk4"  # "rk4", "euler", "heun", "rkf45"
    nodes = HP(N=N, method=method, dt=dt)
    nodes.set_vars_vals([0.1, 0.1, 0.1])
    # print(nodes.vars_nodes)

    nodes.extend_params()  # 扩展参数

    print(nodes.params_nodes)

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
