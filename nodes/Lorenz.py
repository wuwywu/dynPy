# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/16
# User      : WuY
# File      : Lorenz.py
# Lorenz system 模型
# refernce : E.N. Lorenz, Deterministic nonperiodic fow, J. Atmos. Sci. 20, 130-141 (1963).

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
def Lorenz_model(vars, t, I, params):
    res = np.zeros_like(vars)
    # 常数参数
    sigma, rho, beta = params

    # 状态变量
    x, y, z = vars

    # 变量的导数
    dx_dt = sigma * (y - x) + I[0] 
    dy_dt = x * (rho - z) - y + I[1]
    dz_dt = x * y - beta * z + I[2]

    # 输出结果
    res[0] = dx_dt
    res[1] = dy_dt
    res[2] = dz_dt

    return res

class Lorenz(Nodes):
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，("euler", "heun", "rk4")
        dt : 计算步长(dt=.01)

        params_nodes (dict): 节点模型参数
        vars_nodes (numpy.ndarray): 节点模型状态变量
        t (float): 模拟的理论时间
    """
    def __init__(self, N=1, method="euler", dt=0.01):
        super().__init__(N, method, dt)
        # self.N = N  # 神经元数量
        # self.dt = dt
        # self.method = method
        self.model = Lorenz_model  # 模型的微分方程
        self._params()
        self._vars()

    def _params(self):
        self.params_nodes = {
            "sigma": 10.,
            "rho": 28.,
            "beta": 8 / 3,
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
        self.method(Lorenz_model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        self.t += self.dt  # 时间前进


if __name__ == "__main__":
    N = 2
    dt = 0.01
    method = "rk4"  # "rk4", "euler", "heun"
    nodes = Lorenz(N=N, method=method, dt=dt)
    # nodes.set_vars_vals([0])
    # print(nodes.vars_nodes)

    time = []
    mem = []

    for i in range(100_00):
        nodes()
        time.append(nodes.t)
        mem.append(nodes.vars_nodes[0].copy())

    plt.plot(time, mem)

    plt.show()