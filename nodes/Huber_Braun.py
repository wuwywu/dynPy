# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/07/02
# User      : WuY
# File      : Huber_Braun.py
# Huber-Braun 模型
# description : 模型通过四个描述物理和生理量的微分方程，细致刻画了外周冷受体细胞膜上离子电流（如快速动力学电流、慢速亚阈值电流等 ）
# 随温度变化的情况。借助这些方程，能精准模拟出在不同温度条件下，冷受体如何产生相应的脉冲活动，
# 例如从单峰放电模式转变为爆发式放电模式，进而揭示冷受体感知温度并将其转化为电信号的具体生理机制。
# refs : [1] H.A. Braun, M.T. Huber, M. Dewald, K. Schäfer, K. Voigt, Computer simulations of neuronal signal transduction: The role of nonlinear dynamics and noise, Int. J. Bifurcation Chaos 8 (1998) 881–889.
# [2] A. Farrera-Megchun, P. Padilla-Longoria, G.J.E. Santos, J. Espinal-Enríquez, R. Bernal-Jaquez, Explosive synchronization driven by repulsive higher-order interactions in coupled neurons, Chaos, Solitons  Fractals 196 (2025) 116368.


import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random
from base_mods import Neurons

# np.random.seed(2025)
# random.seed(2025)


@njit
def Huber_Braun_model(vars, t, I, params):
    """
    Huber-Braun神经元模型的微分方程系统
    
    参数:
        vars: 状态变量数组 [V, a_r, a_sd, a_sr]
        t: 时间 (ms)
        I: 外部电流数组
        params: 模型参数数组 [g_d, g_r, g_sd, g_sr, g_L, E_d, E_r, E_sd, E_sr, E_L, 
                        s_d, s_r, s_sd, V_0d, V_0r, V_0sd, 
                        tau_r, tau_sd, tau_sr, eta, theta, C_m, temperature]
    
    返回:
        res: 导数数组 [dV/dt, da_r/dt, da_sd/dt, da_sr/dt]
    """
    res = np.zeros_like(vars)
    
    # 提取参数
    (g_d, g_r, g_sd, g_sr, g_L, E_d, E_r, E_sd, E_sr, E_L,
     s_d, s_r, s_sd, V_0d, V_0r, V_0sd,
     tau_r, tau_sd, tau_sr, eta, theta, C_m, temperature) = params
    
    # 提取状态变量
    V, a_r, a_sd, a_sr = vars
    
    # 计算温度因子
    rho_T = 1.3 ** ((temperature - 25) / 10)  # 电流缩放因子
    phi_T = 3.0 ** ((temperature - 25) / 10)  # 动力学因子
    
    # 1. 快速去极化电流 (I_d) 的激活变量 (稳态值)
    a_d = 1.0 / (1.0 + np.exp(-s_d * (V - V_0d)))
    
    # 2. 快速复极化电流 (I_r) 的激活变量
    a_r_inf = 1.0 / (1.0 + np.exp(-s_r * (V - V_0r)))
    da_r_dt = phi_T * (a_r_inf - a_r) / tau_r + I[1]
    
    # 3. 慢速亚阈值去极化电流 (I_sd) 的激活变量
    a_sd_inf = 1.0 / (1.0 + np.exp(-s_sd * (V - V_0sd)))
    da_sd_dt = phi_T * (a_sd_inf - a_sd) / tau_sd + I[2]
    
    # 4. 计算I_sd电流值
    I_sd = rho_T * g_sd * a_sd * (V - E_sd)
    
    # 5. 慢速亚阈值复极化电流 (I_sr) 的激活变量
    da_sr_dt = phi_T * (-eta * I_sd - theta * a_sr) / tau_sr + I[3]
    
    # 计算所有离子电流
    I_d = rho_T * g_d * a_d * (V - E_d)
    I_r = rho_T * g_r * a_r * (V - E_r)
    I_sr = rho_T * g_sr * a_sr * (V - E_sr)
    I_L = g_L * (V - E_L)  # 漏电流不包含温度因子
    
    # 膜电位的导数
    dV_dt = (I[0] - I_d - I_r - I_sd - I_sr - I_L) / C_m
    
    # 输出结果
    res[0] = dV_dt
    res[1] = da_r_dt
    res[2] = da_sd_dt
    res[3] = da_sr_dt
    
    return res


class Huber_Braun(Neurons):
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，("euler", "heun", "rk4", "rkf45")
        dt : 计算步长
        temperature: 温度(℃)

        params_nodes (dict): 节点模型参数
        vars_nodes (numpy.ndarray): 节点模型状态变量
        t (float): 模拟的理论时间
    """
    def __init__(self, N=1, method="euler", dt=0.01, temperature=25.):
        super().__init__(N, method=method, dt=dt)
        # self.N = N  # 神经元数量
        # self.dt = dt
        # self.method = method
        self.temperature = temperature
        self.model = Huber_Braun_model  # 模型的微分方程
        self._params()
        self._vars()

    def _params(self):
        self.params_nodes = {
            'g_d': 1.5,  # 去极化通道的最大电导(mS/cm2)
            'g_r': 2.0,  # 复极化通道的最大电导(mS/cm2)
            'g_sd': 0.25,  # 慢速亚阈值去极化通道的最大电导(mS/cm2)
            'g_sr': 0.4,  # 慢速亚阈值复极化通道的最大电导(mS/cm2)
            'g_L': 0.1,  # 漏离子电导(mS/cm2)

            'E_d': 50.,  # 去极化通道的平衡电位(mV)
            'E_r': -90.,  # 复极化通道的平衡电位(mV)
            'E_sd': 50.,  # 慢速亚阈值去极化通道的平衡电位(mV)
            'E_sr': -90.,  # 慢速亚阈值复极化通道的平衡电位(mV)
            'E_L': -60.,  # 漏离子的平衡电位(mV)

            's_d': 0.25,  # 去极化通道的激活函数参数
            's_r': 0.25,  # 复极化通道的激活函数参数
            's_sd': 0.09,  # 慢速亚阈值去极化通道的激活函数参数

            'V_0d': -25.,  # 去极化通道的激活函数参数
            'V_0r': -25.,  # 复极化通道的激活函数参数
            'V_0sd': -40.,  # 慢速亚阈值去极化通道的激活函数参数

            'tau_r': 2.,  # 复极化通道的时间常数(ms)
            'tau_sd': 10.,  # 慢速亚阈值去极化通道的时间常数(ms)
            'tau_sr': 20.,  # 慢速亚阈值复极化通道的时间常数(ms)

            'eta': 0.012,  # 慢速亚阈值复极化通道的激活函数参数
            'theta': 0.17,  # 慢速亚阈值复极化通道的激活函数参数
            'Cm': 1.,  # 比膜电容(uF/cm2)
            "temperature": self.temperature,  # 温度(℃)
        } 

        self.th_up = -20      # 放电阈上值
        self.th_down = -20  # 放电阈下值
    
    def _vars(self):
        self.t = 0.  # 运行时间
        # 模型变量的初始值
        self.V0 = np.random.uniform(-.3, .3, self.N)
        self.a_r0 = 1 * np.random.rand(self.N)
        self.a_sd0 = 1 * np.random.rand(self.N)
        self.a_sr0 = 1 * np.random.rand(self.N)
        self.vars_nodes = np.vstack((self.V0, self.a_r0, self.a_sd0, self.a_sr0))

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
        self.method(self.model, self.vars_nodes, self.t, self.dt, I, params_list)  #
        
        if self.spiking: 
            self._spikes_eval(self.vars_nodes[0], self.t, self.th_up, self.th_down, self.flag, self.flaglaunch, self.firingTime)  # 放电测算

            if self.record_spike_times:
                # 调用单独的记录峰值时间的函数
                self._record_spike_times(self.flaglaunch, self.t, self.spike_times, self.spike_counts, self.max_spikes)

        self.t += self.dt  # 时间前进


if __name__ == "__main__": 
    N = 2
    method = "euler"              # "rk4", "euler", "heun", "rkf45"
    nodes = Huber_Braun(N=N, method=method, temperature=25.)  # , temperature=25.
    # nodes.N = 3
    # nodes.set_vars_vals([0, 0, 0, 0])
    # nodes._vars_f()
    # print(nodes.vars_nodes)

    time = []
    mem = []
    for i in range(200_00):
        nodes()

    nodes.record_spike_times = True
    for i in range(2500_00):
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
    
