# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/18
# User      : WuY
# File      : analys_prc.py
# 相位响应曲线 (Phase Response Curves, PRCs)
# refs : C. Börgers, An Introduction to Modeling Neuronal Dynamics,
# Springer International Publishing, Cham, 2017.
# https://doi.org/10.1007/978-3-319-51171-9.
# 描述 : 这个代码用于测量神经元的 "相位漂移" 和 "相位响应曲线"
# 包含了两个工具：
#           1、phase_shift : 相位漂移
#           2、Phase_Response_Curves : 相位响应曲线
# 使用方法，1、代码下面；2、测试文件中。

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random
from base_mods import Euler, RK4, discrete

# np.random.seed(2024)
# random.seed(2024)


# =========================== 定义 PRC 所使用的突触形式 ==========================
class syn_prc:
    """
        AMPA-like synaptic input pulse
        这是参考文献中给的一种化学突触形式，这个化学突触作为 PRCs 的输入脉冲使用
        args:
            tau_peak : 突触门控变量到达峰值的时间，
                通过这个值控制 与突触前相关的 q 变量的时间参数 tau_d_q
                (使用的方法是二分法)
            dt : 算法的时间步长
            method : 计算非线性微分方程的方法，("euler", "rk4", "discrete")
    """
    def __init__(self, N, method="euler", dt=.01, tau_peak=0.5):
        self.N = N  # 神经元数量
        self.dt = dt
        method_options = ["euler", "rk4", "discrete"]
        if method not in method_options:
            raise ValueError(f"无效选择，method在{method_options}选择")
        if method == "euler":   self.method = Euler
        elif method == "rk4":   self.method = RK4
        elif method == "discrete":   self.method = discrete

        self._params()
        self._vars()
        tau_r, tau_d = self.params_syn["tau_r"], self.params_syn["tau_d"]
        self.tau_d_q = tau_d_q_function(tau_r, tau_d, tau_peak, dt)

    def _params(self):
        # 突触参数
        self.params_syn = {
            "e": 0.,            # 化学突触的平衡电位(mv)
            "tau_r": 0.5,       # 突触上升时间常数
            "tau_d": 2.0,       # 突触衰减时间常数
            "g_syn": 0.1,       # 突触的最大电导
        }

    def _vars(self):
        # 突触状态变量
        self.q0 = np.zeros(self.N)
        self.s0 = np.zeros(self.N)
        self.vars_syn = np.vstack((self.q0, self.s0))

        self.N_vars = 2  # 变量的数量

    def __call__(self, t, ts):
        """
        args:
            ts : list/numpy 所有峰放电的时间
        """
        # 计算 tau_d_q
        ts = np.array(ts)
        self.vars_syn[0][np.abs(t - ts) < (0.5 * self.dt)] = 1
        params_list = list(self.params_syn.values())
        self.method(syn_model, self.vars_syn, t, self.dt, params_list, self.tau_d_q)


@njit
def syn_model(vars, t, params, tau_d_q):
    res = np.zeros_like(vars)
    # 状态变量
    q, s = vars
    # 突触参数
    e, tau_r, tau_d, g_syn = params

    dq_dt = - q / tau_d_q
    ds_dt = q * (1 - s) / tau_r - s / tau_d

    res[0] = dq_dt
    res[1] = ds_dt

    return res

def tau_d_q_function(tau_r, tau_d, tau_peak, dt):
    """
        利用 tau_r 和 tau_d 计算出 tau_peak 相应的 tau_d_q
    """
    # 给 tau_d_q 设置一个区间
    tau_d_q_left = 1.0
    while tau_peak_function(tau_r, tau_d, tau_d_q_left, dt) > tau_peak:
        tau_d_q_left *= 0.5

    tau_d_q_right = tau_r
    while tau_peak_function(tau_r, tau_d, tau_d_q_right, dt) < tau_peak:
        tau_d_q_right *= 2.0

    # 使用二分法 (bisection method) 求出与 tau_peak 对应的 tau_d_q
    while tau_d_q_right - tau_d_q_left > 1e-12:
        tau_d_q_mid = 0.5 * (tau_d_q_left + tau_d_q_right)
        if (tau_peak_function(tau_r, tau_d, tau_d_q_mid, dt) <= tau_peak):
            tau_d_q_left = tau_d_q_mid
        else:
            tau_d_q_right = tau_d_q_mid

    tau_d_q = 0.5 * (tau_d_q_left + tau_d_q_right)

    return tau_d_q

def tau_peak_function(tau_r, tau_d, tau_d_q, dt):
    """
        通过 tau_d_q 给出 s 的峰值时间
    """
    # 参数
    dt05 = 0.5 * dt

    s = 0
    t = 0
    ds_dt = np.exp(-t / tau_d_q) * (1.0 - s) / tau_r - s / tau_d
    while ds_dt > 0:
        t_old = t
        ds_dt_old = ds_dt
        s_tmp = s + dt05 * ds_dt
        ds_dt_tmp = np.exp(-(t + dt05) / tau_d_q) * \
                    (1.0 - s_tmp) / tau_r - s_tmp / tau_d
        s = s + dt * ds_dt_tmp
        t = t + dt
        ds_dt = np.exp(-t / tau_d_q) * (1.0 - s) / tau_r - s / tau_d

    tau_peak_new = (t_old * (-ds_dt) + t * ds_dt_old) / (ds_dt_old - ds_dt)  # 线性插值法

    return tau_peak_new




