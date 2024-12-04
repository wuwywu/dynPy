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
        self._method = method
        method_map = {"euler": Euler, "rk4": RK4, "discrete": discrete}
        if method not in method_map:
            raise ValueError(f"无效选择，method 必须是 {list(method_map.keys())}")
        self.method = method_map[method]

        self._params()
        self._vars()
        tau_r, tau_d = self.params_syn["tau_r"], self.params_syn["tau_d"]
        self.tau_d_q = tau_d_q_function(tau_r, tau_d, tau_peak, dt)

    def _params(self):
        # 突触参数
        self.params_syn = {
            "tau_r": 0.5,       # 突触上升时间常数
            "tau_d": 2.0,       # 突触衰减时间常数
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

        return self.vars_syn[0], self.vars_syn[1]

@njit
def syn_model(vars, t, params, tau_d_q):
    res = np.zeros_like(vars)
    # 状态变量
    q, s = vars
    # 突触参数
    tau_r, tau_d= params

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


# =========================== PRC 所使用的脉冲形式 ==========================
@njit
def generate_pulses(current_time, ts, pulse_widths=1.):
    """
        根据当前时间和给定的多个脉冲起始时间及宽度输出脉冲值（1 或 0）
        current_time: 当前时间
        ts: 脉冲开始的时间列表
        pulse_widths: 脉冲的宽度列表，可以是单个值或多个值
    """
    ts = np.asarray(ts)

    # 如果 pulse_widths 是一个单一值，将其扩展为与 t_starts 长度相同的列表
    if isinstance(pulse_widths, (int, float)):
        pulse_widths = np.full_like(ts, pulse_widths)
    else:
        pulse_widths = np.asarray(pulse_widths)  # 将 pulse_widths 转换为 NumPy 数组
    
    # 使用 NumPy 向量化计算
    pulse_ends = ts + pulse_widths  # 计算每个脉冲的结束时间
    pulse_values = (ts <= current_time) & (current_time < pulse_ends)  # 判断是否在脉冲区间内
    return pulse_values.astype(np.int32)  # 将布尔值转换为 0 或 1

# =========================== 神经元的相位漂移 ==========================
class phase_shift:
    """
        这个代码，给出神经元的相位漂移
        args:
            node : （类）节点类
            phase : list 刺激相位 range(0, 1)

        输入脉冲被设置在第5个到第6个峰之间，代码定位到：_node_init
            self.ts_list = self.in_phase*self.T + self.T_spike_list[4]
        里面的重要参数：
            T_spike_list        : 没有输入脉冲时，spikes的时间
            T_spike_act_list    : 有输入脉冲时，spikes的时间
            ts_list             : 给输入脉冲的时间
            in_phase            : 给输入脉冲的的相位
            mem_no_in           : 没有输入脉冲时，膜电位的变化
            mem_in              : 有输入脉冲时，膜电位的变化
    """
    def __init__(self, nodes, phase=[0.5], pulses=False, syn_tau_peak=0.5, syn_tau_r=0.5, syn_tau_d=2.0):
        self.nodes = nodes                            # 输入节点
        self.N = len(phase)
        self.dt = self.nodes.dt
        self._method = self.nodes._method

        self.pulses = pulses
        if self.pulses:
            self.syn_in = generate_pulses
        else:
            self.syn_in = syn_prc(self.N, dt=self.dt, method=self._method, tau_peak=syn_tau_peak)       # 实例化输入突触
            self.syn_in.params_syn["tau_r"] = syn_tau_r
            self.syn_in.params_syn["tau_d"] = syn_tau_d

        self.in_phase = np.array(phase)
        self._params()
        self._vars()
        self._node_init()

    def _params(self):
        self.T_init =10000      # 初始化节点时间
        self.e = 0.             # 化学突触的平衡电位
        self.g_syn = 0.1        # 突触的最大电导

        self.pulses_width = 1.  # 脉冲宽度

        self.th_up = self.nodes.th_up       # 放电阈值
        self.th_down = self.nodes.th_down   # 放电阈下值

    def _vars(self):
        # 放电变量
        self.max_init = -np.inf                     # 初始最大值
        self.max = -np.inf + np.zeros(self.N)     # 初始变化最大值
        self.nn = np.zeros(self.N, dtype=int)            # 记录每个节点的ISI的个数
        self.flag = np.zeros(self.N, dtype=int)          # 放电标志
        self.T_pre = np.zeros(self.N)             # 前峰时间
        self.T_post = np.zeros(self.N)            # 后峰时间

    def __call__(self):
        mem = self.nodes.vars_nodes[0]
        self.T_spike_act_list = np.zeros((self.N, 10))  # 记录刺激后的峰

        t_final = 5000  # 最大初始时间
        self.mem_in = []
        self.I_in = []
        while self.nn.min() < 9 and self.nodes.t < t_final / self.dt:
            t = self.nodes.t
            if self.pulses:
                s = self.syn_in(t, self.ts_list, self.pulses_width)
            else:
                q, s = self.syn_in(t, self.ts_list)

            I = self.g_syn * s * (self.e - mem)
            self.I_in.append(I.copy())
            self.nodes(I)
            self._spikes_eval(mem)  # 放电测算

            self.mem_in.append(mem.copy())

        self.mem_in= np.array(self.mem_in)

    def _node_init(self):
        """
            这个函数的作用是：
                1、重置节点初始值（所有节点初始值都设定为一样）
                2、重置节点数量
                3、重置节点运行时间
                4、给出节点的振荡周期
                5、通过相位给出输入脉冲刺激的时间
        """
        # 初始化节点
        for i in range(self.T_init):
            self.nodes()

        # 记录初始值，重置时间
        self.nodes.t = 0.                               # 初始化运行时间
        vars_init = self.nodes.vars_nodes[:, 0].copy()  # 设定变量初始值

        # ================================== 记录没有输入脉冲时峰值时间 ==================================
        mem = self.nodes.vars_nodes[0]
        th_up = self.th_up        # 放电阈值
        th_down = self.th_down    # 放电阈下值
        max_init = -np.inf              # 初始最大值（负无穷大）
        max = -np.inf + np.zeros(1)     # 初始变化最大值（负无穷大）
        nn = np.zeros(1)                # 记录每个节点的ISI的个数
        flag = np.zeros(1)              # 放电标志
        T_pre = np.zeros(1)             # 前峰时间
        T_post = np.zeros(1)            # 后峰时间

        self.T_spike_list = []               # 记录峰的时间

        t_final = 10000                  # 最大初始时间
        ISI_list = []
        self.mem_no_in = []
        while nn[0]<10 and self.nodes.t<t_final/self.dt:
            # 运行节点
            self.nodes()
            self.mem_no_in.append(self.nodes.vars_nodes[0, 0].copy())

            t = self.nodes.t
            # -------------------- 放电开始 --------------------
            firing_StartPlace = np.where((mem > th_up) & (flag == 0))  # 放电开始的位置
            flag[firing_StartPlace] = 1  # 放电标志改为放电
            # -------------------- 放电期间 --------------------
            firing_Place = np.where((mem > max) & (flag == 1))         # 放电期间并且还没有到达峰值
            max[firing_Place] = mem[firing_Place]
            T_post[firing_Place] = t                                   # 存储前面峰的时间
            # -------------------- 放电结束 --------------------
            firing_endPlace = np.where((mem < th_down) & (flag == 1))  # 放电结束的位置
            firing_endPlace2 = np.where((mem < th_down) & (flag == 1) & (nn > 2))  # 放电结束的位置2
            flag[firing_endPlace] = 0  # 放电标志改为放电
            nn[firing_endPlace] += 1  # 结束放电ISI数量+1

            ISI = T_post[firing_endPlace2] - T_pre[firing_endPlace2]        # ISI（峰峰间隔，周期）
            ISI_list.extend(ISI)

            T_pre[firing_endPlace] = T_post[firing_endPlace]
            self.T_spike_list.extend(T_post[firing_endPlace])

            max[firing_endPlace] = max_init

        # 初始化 `节点初始值`，`初始时间` 和 `节点数量`；给出振荡周期
        self.nodes.N = self.N
        self.nodes.set_vars_vals(vars_vals=vars_init)
        self.nodes.t = 0.  # 初始化运行时间

        self.T = ISI_list[-1]

        # 通过相位给给出 `输入脉冲` 时间(第5个峰后添加，可以修改)
        self.ts_list = self.in_phase*self.T + self.T_spike_list[4]

    def _spikes_eval(self, mem):
        """
            测试放电
        """
        # -------------------- 放电开始 --------------------
        firing_StartPlace = np.where((mem > self.th_up) & (self.flag == 0))  # 放电开始的位置
        self.flag[firing_StartPlace] = 1  # 放电标志改为放电
        # -------------------- 放电期间 --------------------
        firing_Place = np.where((mem > self.max) & (self.flag == 1))  # 放电期间并且还没有到达峰值
        self.max[firing_Place] = mem[firing_Place]
        self.T_post[firing_Place] = self.nodes.t
        #  -------------------- 放电结束 -------------------
        firing_endPlace = np.where((mem < self.th_down) & (self.flag == 1))  # 放电结束的位置
        firing_endPlace2 = np.where((mem < self.th_down) & (self.flag == 1) & (self.nn > 2))  # 放电结束的位置2
        self.flag[firing_endPlace] = 0  # 放电标志改为放电
        self.nn[firing_endPlace] += 1  # 结束放电ISI数量+1

        self.T_pre[firing_endPlace] = self.T_post[firing_endPlace]
        if firing_endPlace[0].size != 0:
            # 给出放电的坐标
            coordinates = np.stack((firing_endPlace[0], self.nn[firing_endPlace]-1), axis=-1)
            # print(firing_endPlace[0])
            self.T_spike_act_list[coordinates[:, 0], coordinates[:, 1]] = self.T_post[firing_endPlace]

        self.max[firing_endPlace] = self.max_init

    def plot_phase_shift(self):
        x_l = self.mem_in.shape[0]
        x_ = np.arange(x_l)*self.dt
        fig, axs = plt.subplots(self.N, sharex="all", layout='constrained')
        if self.N == 1:
            axs.plot(x_, self.mem_in[:, 0])
            axs.plot(x_, self.mem_no_in[:x_l], color="r")
            axs.axvline(self.ts_list[0], color='k', linestyle='--', lw=2)
        else:
            for i in range(self.N):
                axs[i].plot(x_, self.mem_in[:, i])
                axs[i].plot(x_, self.mem_no_in[:x_l], color="r")
                axs[i].axvline(self.ts_list[i], color='k', linestyle='--', lw=2)

        plt.xlim(self.T_spike_list[3], self.T_spike_list[6])


# =========================== 神经元的相位响应曲线 ==========================
class Phase_Response_Curves:
    """
        这个代码，给出神经元的相位响应曲线
        args:
            node : （类）节点类
            N_phase : int 给出相位列表的数量

        输入脉冲被设置在第5个到第6个峰之间，代码定位到：_node_init
            self.ts_list = self.in_phase*self.T + self.T_spike_list[4]
        里面的重要参数：
            T_spike_list  : 没有输入脉冲时，spikes的时间
            T_spike_act_list : 有输入脉冲时，spikes的时间
            ts_list :   给输入脉冲的时间
            in_phase :  给输入脉冲的的相位
    """
    def __init__(self, nodes, N_phase=500, pulses=False, syn_tau_peak=0.5, syn_tau_r=0.5, syn_tau_d=2.0):
        self.in_phase = np.linspace(0, 1, N_phase)
        self.N = N_phase
        self.nodes = nodes                            # 输入节点
        self.dt = self.nodes.dt
        self._method = self.nodes._method
        self.pulses = pulses
        if self.pulses:
            self.syn_in = generate_pulses
        else:
            self.syn_in = syn_prc(self.N, dt=self.dt, method=self._method, tau_peak=syn_tau_peak)       # 实例化输入突触
            self.syn_in.params_syn["tau_r"] = syn_tau_r
            self.syn_in.params_syn["tau_d"] = syn_tau_d

        self._params()
        self._vars()
        self._node_init()

    def _params(self):
        self.T_init =10000      # 初始化节点时间
        self.e = 0.             # 化学突触的平衡电位
        self.g_syn = 0.1        # 突触的最大电导

        self.pulses_width = 1.  # 脉冲宽度

        self.th_up = self.nodes.th_up       # 放电阈值
        self.th_down = self.nodes.th_down   # 放电阈下值

    def _vars(self):
        # 放电变量
        self.max_init = -np.inf                     # 初始最大值
        self.max = -np.inf + np.zeros(self.N)     # 初始变化最大值
        self.nn = np.zeros(self.N, dtype=int)            # 记录每个节点的ISI的个数
        self.flag = np.zeros(self.N, dtype=int)          # 放电标志
        self.T_pre = np.zeros(self.N)             # 前峰时间
        self.T_post = np.zeros(self.N)            # 后峰时间

    def __call__(self):
        mem = self.nodes.vars_nodes[0]
        self.T_spike_act_list = np.zeros((self.N, 10))  # 记录刺激后的峰

        t_final = 5000  # 最大初始时间
        while self.nn.min() < 9 and self.nodes.t < t_final / self.dt:
            t = self.nodes.t
            if self.pulses:
                s = self.syn_in(t, self.ts_list, self.pulses_width)
            else:
                q, s = self.syn_in(t, self.ts_list)
            I = self.g_syn * s * (self.e - mem)
            self.nodes(I)
            self._spikes_eval(mem)  # 放电测算

        return self.return_PRC()

    def return_PRC(self):
        T = self.T                  # 周期
        in_phase = self.in_phase    # 刺激的相位（在第5和6峰之间）
        T_spike_act_list = self.T_spike_act_list[:, 5]  # 第6个峰的时间
        ts_list = self.ts_list      # 输入脉冲的时间
        self.PRC = 1 - in_phase - (T_spike_act_list - ts_list) / T

        return self.PRC

    def _node_init(self):
        """
            这个函数的作用是：
                1、重置节点初始值（所有节点初始值都设定为一样）
                2、重置节点数量
                3、重置节点运行时间
                4、给出节点的振荡周期
                5、通过相位给出输入脉冲刺激的时间
        """
        # 初始化节点
        for i in range(self.T_init):
            self.nodes()

        # 记录初始值，重置时间
        self.nodes.t = 0.                               # 初始化运行时间
        vars_init = self.nodes.vars_nodes[:, 0].copy()  # 设定变量初始值

        # ================================== 记录没有输入脉冲时峰值时间 ==================================
        mem = self.nodes.vars_nodes[0]
        th_up = self.th_up        # 放电阈值
        th_down = self.th_down    # 放电阈下值
        max_init = -np.inf              # 初始最大值（负无穷大）
        max = -np.inf + np.zeros(1)     # 初始变化最大值（负无穷大）
        nn = np.zeros(1)                # 记录每个节点的ISI的个数
        flag = np.zeros(1)              # 放电标志
        T_pre = np.zeros(1)             # 前峰时间
        T_post = np.zeros(1)            # 后峰时间

        self.T_spike_list = []               # 记录峰的时间

        t_final = 10000                  # 最大初始时间
        ISI_list = []
        self.mem_no_in = []
        while nn[0]<10 and self.nodes.t<t_final/self.dt:
            # 运行节点
            self.nodes()
            self.mem_no_in.append(self.nodes.vars_nodes[0, 0].copy())

            t = self.nodes.t
            # -------------------- 放电开始 --------------------
            firing_StartPlace = np.where((mem > th_up) & (flag == 0))  # 放电开始的位置
            flag[firing_StartPlace] = 1  # 放电标志改为放电
            # -------------------- 放电期间 --------------------
            firing_Place = np.where((mem > max) & (flag == 1))         # 放电期间并且还没有到达峰值
            max[firing_Place] = mem[firing_Place]
            T_post[firing_Place] = t                                   # 存储前面峰的时间
            # -------------------- 放电结束 --------------------
            firing_endPlace = np.where((mem < th_down) & (flag == 1))  # 放电结束的位置
            firing_endPlace2 = np.where((mem < th_down) & (flag == 1) & (nn > 2))  # 放电结束的位置2
            flag[firing_endPlace] = 0  # 放电标志改为放电
            nn[firing_endPlace] += 1  # 结束放电ISI数量+1

            ISI = T_post[firing_endPlace2] - T_pre[firing_endPlace2]        # ISI（峰峰间隔，周期）
            ISI_list.extend(ISI)

            T_pre[firing_endPlace] = T_post[firing_endPlace]
            self.T_spike_list.extend(T_post[firing_endPlace])

            max[firing_endPlace] = max_init

        # 初始化 `节点初始值`，`初始时间` 和 `节点数量`；给出振荡周期
        self.nodes.N = self.N
        self.nodes.set_vars_vals(vars_vals=vars_init)
        self.nodes.t = 0.  # 初始化运行时间

        self.T = ISI_list[-1]

        # 通过相位给给出 `输入脉冲` 时间(第5个峰后添加，可以修改)
        self.ts_list = self.in_phase*self.T + self.T_spike_list[4]

    def _spikes_eval(self, mem):
        """
            测试放电
        """
        # -------------------- 放电开始 --------------------
        firing_StartPlace = np.where((mem > self.th_up) & (self.flag == 0))  # 放电开始的位置
        self.flag[firing_StartPlace] = 1  # 放电标志改为放电
        # -------------------- 放电期间 --------------------
        firing_Place = np.where((mem > self.max) & (self.flag == 1))  # 放电期间并且还没有到达峰值
        self.max[firing_Place] = mem[firing_Place]
        self.T_post[firing_Place] = self.nodes.t
        #  -------------------- 放电结束 -------------------
        firing_endPlace = np.where((mem < self.th_down) & (self.flag == 1))  # 放电结束的位置
        firing_endPlace2 = np.where((mem < self.th_down) & (self.flag == 1) & (self.nn > 2))  # 放电结束的位置2
        self.flag[firing_endPlace] = 0  # 放电标志改为放电
        self.nn[firing_endPlace] += 1  # 结束放电ISI数量+1

        self.T_pre[firing_endPlace] = self.T_post[firing_endPlace]
        if firing_endPlace[0].size != 0:
            # 给出放电的坐标
            coordinates = np.stack((firing_endPlace[0], self.nn[firing_endPlace]-1), axis=-1)
            # print(firing_endPlace[0])
            self.T_spike_act_list[coordinates[:, 0], coordinates[:, 1]] = self.T_post[firing_endPlace]

        self.max[firing_endPlace] = self.max_init

    def plot_phase_shift(self):
        fig, axs = plt.subplots(1, layout='constrained')
        axs.plot(self.in_phase, self.PRC)
        axs.axhline(y=0, color='r', linestyle='--', lw=2)

