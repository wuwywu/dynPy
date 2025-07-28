# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/07/28
# User      : WuY
# File      : non_D_ZBKE.py
# non-dimensionalized Zhabotinsky-Buchholtz-Kiyatkin-Epstein (ZBKE) model
# description : 典型化学振荡反应，在时空维度展现出周期性、混沌等复杂动力学行为，为非线性化学动力学、耦合振荡器同步等研究提供了理想实验体系
# ref : A. M. Zhabotinsky, F. Buchholtz, A. B. Kiyatkin, I. R. Epstein, Oscillations and waves in metal-ion-catalyzed bromate oscillating reactions in highly oxidized states. J. Phys. Chem. 97, 7578–7584 (1993).


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
def ZBKE_model(vars, t, I, params):
    """
    ZBKE模型的数值计算实现
    
    参数:
        vars: 状态变量 [u, v]
        t: 时间变量（未直接使用，但为了保持接口一致性保留）
        I: 输入光强 [I_i]
        params: 模型参数，包括 [epsilon1, epsilon2, epsilon3, alpha, beta, gamma, mu, q, Iext]
    
    返回:
        res: 状态变量的导数 [du/dt, dv/dt]
    """
    res = np.zeros_like(vars)
    
    # 解析参数
    epsilon1, epsilon2, epsilon3, alpha, beta, gamma, mu, q, Iext = params
    
    # 解析状态变量
    u, v = vars
    
    # 计算w的稳态值 (w_{i,ss})
    sqrt_term = np.sqrt(16 * gamma * epsilon2 * u + v**2 - 2 * v + 1)
    w_ss = (sqrt_term + v - 1) / (4 * gamma * epsilon2)
    
    # 计算u的导数
    term2_numerator = alpha * q * v
    term2_denominator = epsilon3 + 1 - v
    term2 = (term2_numerator / term2_denominator + beta) * (mu - u) / (mu + u)
    term3 = gamma * epsilon2 * w_ss**2 + (1 - v) * w_ss
    term4 = -u**2 - u
    
    # u的导数 (考虑epsilon1的缩放)
    du_dt = (I[0] + Iext + term2 + term3 + term4) / epsilon1
    
    # 计算v的导数
    dv_term1 = 2 * (I[1] + Iext)
    dv_term2 = (1 - v) * w_ss
    dv_term3 = - term2_numerator / term2_denominator
    
    dv_dt = dv_term1 + dv_term2 + dv_term3
    
    # 输出结果
    res[0] = du_dt
    res[1] = dv_dt
    
    return res


class ZBKE(Nodes):
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
        self.model = ZBKE_model  # 模型的微分方程
        self._params()
        self._vars()

    def _params(self):
        # 模型参数
        self.params_nodes = {
            "epsilon1"  : 0.11,         # time scale parameters
            "epsilon2"  : 1.7e-5,       # time scale parameters
            "epsilon3"  : 1.6e-3,       # time scale parameters
            "alpha"     : 0.1,          # kinetic parameters
            "beta"      : 1.7e-5,       # kinetic parameters    
            "gamma"     : 1.2,          # kinetic parameters
            "mu"        : 2.4e-4,       # kinetic parameters
            "q"         : 0.9,          # stoichiometric parameter(0.5-0.995)
            "Iext"      : 1.1e-4        # background light intensity
        }

    def _vars(self):
        self.t = 0.  # 运行时间
        # 模型变量的初始值
        self.u0 = np.random.uniform(0., 1., self.N)
        self.v0 = np.random.uniform(0., 1., self.N)
        self.vars_nodes = np.vstack((self.u0, self.v0))

        self.N_vars = 2  # 变量的数量

    def __call__(self, Io=0, axis=np.array([0, 1])):
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
        self.method(self.model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        self.t += self.dt  # 时间前进

    
if __name__ == "__main__":
    N = 10
    dt = 2e-4
    method = "euler"  # "rk4", "euler", "heun", "rkf45"
    nodes = ZBKE(N=N, method=method, dt=dt)
    # nodes.set_vars_vals([0.1, 0.1])
    nodes.params_nodes["q"] = 0.5
    # print(nodes.vars_nodes)

    for i in range(200_0000):
        nodes()

    # nodes.vars_nodes[0, 0] += 1e-3

    time = []
    mem = []

    for i in range(200_0000):
        nodes()
        time.append(nodes.t)
        mem.append(nodes.vars_nodes[0].copy())

    plt.plot(time, mem)

    plt.show()
