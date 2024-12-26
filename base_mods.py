# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/13
# User      : WuY
# File      : base_mods.py
# 文件中包含：
"""
本模块包含：
    1、节点动力学基础模块(神经元，一般节点)
    2、突触动力学基础模块
    3、数值模拟算法(欧拉，龙格库塔，离散)
    4、常用的工具函数
        1). 延迟存储器, delayer
        2). 连接矩阵to拉普拉斯矩阵, to_laplacian
        3). 噪声产生器, noiser
        4). 极大极小值, find_extrema
        5). 二\三维平面流速场, flow_field2D/flow_field3D
        6). 零斜线 nullclines, find_nullclines
"""

"""
    神经元基础模块包含的功能：
        1、计算ISI (cal_isi)
        2、计算CV (cal_cv)
        3、计算Kuramoto Order Parameter (cal_kop)
        4、计算峰值时间 (return_spike_times)
        注：这些功能使用前需要打开峰值时间记录器(record_spike_times = True)

        5、cal_flow_field2D/flow_field3D (计算2/3维的速度场)
        6、cal_nullclines (计算零斜线)   
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
from scipy.optimize import root
import random


# os.environ['NUMBA_NUM_THREADS'] = '4' # 限制使用的线程数量
# np.random.seed(2024)
# random.seed(2024)

# ================================= 神经元模型的基类 =================================
"""
    注：
        1、模拟的理论时间 : t
        2、模拟的时间步长 : dt
        3、神经元基团数量 : N
    关于放电：
        4、放电开始的阈值 : th_up
        5、放电结束的阈值 : th_down
        6、放电的标志 flag (只要大于0就处于放电过程中)
        7、放电开启的标志 flaglaunch (只有在放电的时刻为1， 其他时刻为0)
        8、放电的时间 firingTime (与放电的标志时刻一致)
        9、峰值时间记录器 spike_times (size: [N, max_spikes])
        10、峰值时间计数器 spike_counts (size: [N])
"""
class Neurons:
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，("euler", "rk4", "discrete")
        dt : 计算步长

        params_f (dict): 节点模型参数
        
        t (float): 模拟的理论时间
    """
    def __init__(self, N, method="euler", dt=0.01):
        self.N = N  # 神经元数量
        self.dt = dt
        # 选择数值计算方法
        self._method = method
        method_map = {"euler": Euler, "rk4": RK4, "discrete": discrete}
        if method not in method_map:
            raise ValueError(f"无效选择，method 必须是 {list(method_map.keys())}")
        self.method = method_map[method]

        self.model = model  # 模型的微分方程

        self.fun_switch()
        self.fun_sets()
        self._params_f()
        self._vars_f()

    def _params_f(self):
        self.th_up = 0      # 放电阈上值
        self.th_down = -10  # 放电阈下值

    def _vars_f(self):
        self.t = 0  # 运行时间
        # 模型放电变量
        self.flag = np.zeros(self.N, dtype=np.int32)           # 模型放电标志(>0, 放电)
        self.flaglaunch = np.zeros(self.N, dtype=np.int32)     # 模型开始放电标志(==1, 放电刚刚开始)
        self.firingTime = np.zeros(self.N)                     # 记录放电时间(上次放电)
        # 初始化峰值时间记录相关变量
        self.max_spikes = 1000                                 # 假设每个神经元最多记录 1000 次放电
        self.spike_times = np.full((self.N, self.max_spikes), np.nan)
        self.spike_counts = np.zeros(self.N, dtype=np.int32)   # 放电次数计数

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
        axis = np.asarray(axis)
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

    def set_vars_vals(self, vars_vals=[0, 0, ...]):
        """
            用于自定义所有状态变量的值，可以用于更改网络大小(节点数)
            vars_vals:
                1、长度为 N_vars 的列表
                2、维度为 (N_vars, N) 的数组      
        """
        self._vars()
        self._vars_f()
        for i, val in enumerate(vars_vals):
            self.vars_nodes[i] = val * np.ones(self.N)

    def fun_switch(self):
        """
            功能开关
        """
        self.spiking = True                 # spiking 计算
        self.record_spike_times = False     # 记录spiking时间
    
    def fun_sets(self):
        """
            功能集合
        """
        self._spikes_eval = spikes_eval                 # spiking 计算函数
        self._record_spike_times = record_spike_times   # 记录spiking时间函数

    def return_spike_times(self):
        """
            返回有效的spiking时间
        """
        if not self.record_spike_times:
            raise ValueError("未启用峰值时间记录功能，无法返回spiking时间。")
        
        valid_spike_times = [self.spike_times[i, ~np.isnan(self.spike_times[i])] for i in range(self.N)]

        return valid_spike_times
    
    def cal_isi(self):
        """
            计算每个神经元的有效 ISI（脉冲间隔），去除 NaN 值，保留相同长度的 ISI 数组。

            返回：
                isi_array (ndarray): 二维数组，形状为 (N, max_valid_isi_count)，包含每个神经元的 ISI，未使用的元素填充为 0。
        """
        if not self.record_spike_times:
            raise ValueError("未启用峰值时间记录功能，无法计算 ISI。")

        isi_array = calculate_isi(self.spike_times, self.spike_counts, self.N)
        
        return isi_array
    
    def cal_cv(self):
        """
            计算每个神经元的有效 CV
            变异系数 The coefficient of variation (CV)
                CV=1，会得到泊松尖峰序列（稀疏且不连贯的尖峰）。
                CV<1，尖峰序列变得更加规则，并且对于周期性确定性尖峰，CV 趋近与0。
                CV>1，对应于比泊松过程变化更大的尖峰点过程。

            返回：
                cv_array (ndarray): 一维数组，长度为 N，包含每个神经元的 CV，未使用的元素填充为 0。
        """
        if not self.record_spike_times:
            raise ValueError("未启用峰值时间记录功能，无法计算 CV。")   

        cv_array = calculate_cv(self.spike_times, self.spike_counts, self.N)

        return cv_array
    
    def cal_kop(self, min_spikes=10):
        """
            调用 Kuramoto Order Parameter 计算。
            返回：
                mean_kop (float): 平均 Kuramoto Order Parameter。
                kuramoto (ndarray): 每个时间点的 Kuramoto Order Parameter。
                phase (ndarray): 每个神经元的相位矩阵。
                peak_id (ndarray): 每个神经元的峰值编号（计算完整相位变化）。
                valid_interval (tuple): (first_last_spk, last_first_spk)，有效计算的时间区间。
        """
        if not self.record_spike_times:
            raise ValueError("未启用峰值时间记录功能，无法计算 Kuramoto Order Parameter。")   
        
        return calculate_kuramoto(self.spike_times, self.dt, min_spikes=min_spikes)
    
    def cal_flow_field2D(self, select_dim=(0, 1), vars_lim=(-1., 1., -1., 1.), N=100, initial_vars=None, plt_flag=False):
        """
            二维平面流速场
            select_dim: 选择的维度 (x, y)
            vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max)
            initial_vars: 指定所有变量的值，形状为 (N_vars,)
            N    : 网格点数
            plt_flag : 是否绘制图像
        """
        params_list = list(self.params_nodes.values())
        dX_dt, dY_dt, X, Y = flow_field2D(self.model, params_list, self.N_vars, select_dim=select_dim, vars_lim=vars_lim, N=N, initial_vars=initial_vars)

        if plt_flag:
            plt.streamplot(X, Y, dX_dt, dY_dt, density=1.5, linewidth=1., arrowsize=1.2, arrowstyle='->', color='C1', zorder=0)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Vector Field')

        return dX_dt, dY_dt, X, Y
    
    def cal_flow_field3D(self, select_dim=(0, 1, 2), vars_lim=(-1., 1., -1., 1., -1., 1.), N=100, initial_vars=None):
        """
            三维流速场
            select_dim: 选择的维度 (x, y, z)
            vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max, z_min, z_max)
            N    : 网格点数
            initial_vars: 指定所有变量的值，形状为 (N_vars,)
        """
        params_list = list(self.params_nodes.values())
        dX_dt, dY_dt, dZ_dt, X, Y, Z = flow_field3D(self.model, params_list, self.N_vars, select_dim=select_dim, vars_lim=vars_lim, N=N, initial_vars=initial_vars)

        return dX_dt, dY_dt, dZ_dt, X, Y, Z

    def cal_nullclines(self, x_dim=0, y_dim=1, x_range=(0., 1.), dvar_dt=(0, 1), N=100, initial_guesse=None, initial_vars=None, plt_flag=False):
        """
            通用零斜线求解函数
            x_dim       : 指定自变量的维度 int  (自变量的维度)
            y_dim       : 指定求解的目标维度 int  (应变量的维度)
            x_range     : 零斜线自变量范围 (x_min, x_max)
            dvar_dt     : 求解函数的目标维度 
            N           : 零斜线的点数量
            initial_guesse: 指定初始值
            initial_vars: 指定所有变量的值，形状为 (N_vars,)
            plt_flag    : 是否绘制零斜线
        """
        nullclines_list = np.full((2, N), np.nan)
        params_list = list(self.params_nodes.values())
        for i, dv_dt_dim in enumerate(dvar_dt):
            nullclines = find_nullclines(self.model, params_list, self.N_vars, x_dim=x_dim, y_dim=y_dim, dv_dt_dim=dv_dt_dim, x_range=x_range, N=N, initial_guesse=initial_guesse, initial_vars=initial_vars)

            nullclines_list[i] = nullclines

        if plt_flag:
            v_range = np.linspace(x_range[0], x_range[1], N)
            plt.plot(v_range, nullclines_list[0], label=f'var{dvar_dt[0]} nullcline(dvar{dvar_dt[0]}_dt=0)')
            plt.plot(v_range, nullclines_list[1], label=f'var{dvar_dt[1]} nullcline (dvar{dvar_dt[1]}_dt=0)')
            plt.xlabel(f'var{dvar_dt[0]}')
            plt.ylabel(f'var{dvar_dt[1]}')
            plt.title('Nullclines')
            plt.legend()

        return nullclines_list


@njit
def model(vars, t, I, params):
    res = np.zeros_like(vars)
    # 常数参数
    # params
    # 状态变量
    # vars

    return res

@njit
def spikes_eval(mem, t, th_up, th_down, flag, flaglaunch, firingTime):
    """
        在非人工神经元中，计算神经元的 spiking
    """
    # -------------------- 放电开始 --------------------
    flaglaunch[:] = 0                                           # 重置放电开启标志
    firing_StartPlace = np.where((mem > th_up) & (flag == 0))   # 放电开始的位置
    flag[firing_StartPlace] = 1                                 # 放电标志改为放电
    flaglaunch[firing_StartPlace] = 1                           # 放电开启标志
    firingTime[firing_StartPlace] = t                           # 记录放电时间

    #  -------------------- 放电结束 -------------------
    firing_endPlace = np.where((mem < th_down) & (flag == 1))   # 放电结束的位置
    flag[firing_endPlace] = 0  
    
@njit
def record_spike_times(flaglaunch, t, spike_times, spike_counts, max_spikes):
    """
        记录峰值时间的函数，使用 njit 加速。

        参数：
            flaglaunch (ndarray): 刚开始放电的神经元标志数组。
            t (float): 当前时间。
            spike_times (ndarray): 存储峰值时间的二维数组。
            spike_counts (ndarray): 每个神经元已记录的峰值次数。
            max_spikes (int): 每个神经元最多记录的峰值次数。
    """
    N = flaglaunch.shape[0]
    for i in range(N):
        if flaglaunch[i] > 0.9 and spike_counts[i] < max_spikes:
            spike_times[i, spike_counts[i]] = t
            spike_counts[i] += 1

@njit
def calculate_isi(spike_times, spike_counts, N):
    """
    通过峰值时间计算有效的 ISI。

    参数：
        spike_times (ndarray): 二维数组，形状为 (N, max_spikes)，存储峰值时间。
        spike_counts (ndarray): 一维数组，长度为 N，记录每个神经元已记录的峰值次数。
        N (int): 神经元数量。

    返回：
        isi_array (ndarray): 二维数组，形状为 (N, max_spikes - 1)，包含每个神经元的 ISI，未使用的元素填充为 np.nan。
    """
    # 计算每个神经元的有效 ISI 数量
    isi_counts = np.maximum(spike_counts - 1, 0)

    # 找到最大有效 ISI 数量
    max_valid_isi_count = np.max(isi_counts)
    
    # 初始化 isi_array，使用 0 填充
    isi_array = np.full((N, max_valid_isi_count), np.nan)

    for i in range(N):
        count = spike_counts[i]
        if count > 1:
            # 提取有效的峰值时间
            valid_spike_times = spike_times[i, :count]
            # 计算 ISI
            isi = np.diff(valid_spike_times)
            # 将 ISI 左对齐放入 isi_array 中
            isi_array[i, :isi.size] = isi
        # 如果 count <= 1，isi_array[i, :] 保持为 0
    
    return isi_array

@njit
def calculate_cv(spike_times, spike_counts, N):
    """
        计算每个神经元的 CV（变异系数）。

        参数：
            spike_times (ndarray): 二维数组，形状为 (N, max_spikes)，存储峰值时间。
            spike_counts (ndarray): 一维数组，长度为 N，记录每个神经元已记录的峰值次数。
            N (int): 神经元数量。

        返回：
            cv_array (ndarray): 一维数组，长度为 N，每个元素是对应神经元的 CV 值。
    """
    cv_array = np.full(N, np.nan)

    for i in range(N):
        count = spike_counts[i]
        if count > 1:
            sum_isi = 0.0
            sum_isi_sq = 0.0
            for j in range(count - 1):
                isi = spike_times[i, j + 1] - spike_times[i, j]
                sum_isi += isi
                sum_isi_sq += isi * isi
            mean_isi = sum_isi / (count - 1)
            var_isi = sum_isi_sq / (count - 1) - mean_isi * mean_isi
            var_isi = np.abs(var_isi)
            std_isi = np.sqrt(var_isi)
                
            if mean_isi != 0:
                cv = std_isi / mean_isi
                cv_array[i] = cv
            else:
                cv_array[i] = 0

    return cv_array

@njit
def calculate_kuramoto(spike_times, dt, min_spikes=0):
    """
        使用 spike_times 计算 Kuramoto Order Parameter (KOP)，并输出附加信息。
        
        参数：
            spike_times (ndarray): 形状为 (N, max_spikes) 的二维数组，包含所有神经元的放电时间，NaN 表示无效数据。
            dt (float): 时间步长。
            min_spikes (int): 最小的峰值数量，用于计算 KOP。
            
        返回：
            mean_kop (float): 平均 Kuramoto Order Parameter。
            kuramoto (ndarray): 每个时间点的 Kuramoto Order Parameter。
            phase (ndarray): 每个神经元的相位矩阵。
            peak_id (ndarray): 每个神经元的峰值编号（计算完整相位变化）。
            valid_interval (tuple): (first_last_spk, last_first_spk)，有效计算的时间区间。
    """
    N = spike_times.shape[0]  # 神经元数量

    # 1. 找到每个神经元的第一个和最后一个放电时间
    first_spikes = []
    last_spikes = []
    for neuron_idx in range(N):
        neuron_spkt = spike_times[neuron_idx][~np.isnan(spike_times[neuron_idx])]
        if len(neuron_spkt) > min_spikes:  # 确保神经元有有效放电记录
            first_spikes.append(neuron_spkt[0])
            last_spikes.append(neuron_spkt[-1])
 
    # 检查是否存在有效神经元
    if len(first_spikes) == 0 or len(last_spikes) == 0:
        raise ValueError("没有满足条件的神经元放电的最小值，请检查输入数据或降低 min_spikes 的值！")
    
    first_spikes = np.asarray(first_spikes, dtype=np.float64)
    last_spikes = np.asarray(last_spikes, dtype=np.float64)

    # 定义有效时间区间
    first_last_spk = np.max(first_spikes)  # 最早的最后一个首峰时间
    last_first_spk = np.min(last_spikes)   # 最晚的第一个尾峰时间

    # 限制时间范围
    if first_last_spk >= last_first_spk:
        raise ValueError("有效时间区间无效，请检查 spike_times 数据！")

    # 生成时间向量
    time_start = np.min(first_spikes)  # 最早的第一个峰时间
    time_end = np.max(last_spikes)    # 最晚的最后一个峰时间
    time_vec = np.arange(time_start, time_end, dt)

    # 2. 初始化相位矩阵和峰值编号矩阵
    phase = np.ones((N, len(time_vec))) * -1    # 初始化为无效值
    peak_id = np.ones((N, len(time_vec))) * -1  # 初始化为无效值

    # 3. 计算每个神经元的相位
    for neuron_idx in range(N):
        neuron_spkt = spike_times[neuron_idx][~np.isnan(spike_times[neuron_idx])]
        for i in range(len(neuron_spkt) - 1):
            # 找到对应的时间索引，确保在有效区间内
            ti = max(0, np.searchsorted(time_vec, neuron_spkt[i]))
            tf = min(len(time_vec), np.searchsorted(time_vec, neuron_spkt[i + 1]))

            if tf > ti:  # 确保索引范围有效
                # 插值和峰值编号
                phase[neuron_idx, ti:tf] = np.linspace(0, 2 * np.pi, tf - ti)
                peak_id[neuron_idx, ti:tf] = i

    # 4. 计算完整相位
    # full_phase = 2 * np.pi * peak_id + phase  # 计算完整相位（包含峰值编号）

    # 5. 剔除无效相位区域，并计算 Kuramoto Order Parameter
    idxs = np.where((time_vec > first_last_spk) & (time_vec < last_first_spk))[0]
    phase = phase[:, idxs]
    peak_id = peak_id[:, idxs]  # 剪切出定义的区间
    peak_id -= peak_id[:, :1]
    
    # 计算 Kuramoto Order parameter
    N, T = phase.shape  # 神经元数量和时间点数量

    exp_phase = np.exp(1j * phase) # 复数e指数

    # 手动计算每个时间点的平均值
    mean_complex = np.zeros(T, dtype=np.complex128)
    for t in range(T):
        for n in range(N):
            mean_complex[t] += exp_phase[n, t]
        mean_complex[t] /= N  # 求平均值

    # 计算 Kuramoto Order Parameter
    kuramoto = np.zeros(T, dtype=np.float64)
    for t in range(T):
        kuramoto[t] = np.sqrt(mean_complex[t].real**2 + mean_complex[t].imag**2)

    mean_kop = np.mean(kuramoto)  # 平均 KOP

    return mean_kop, kuramoto, phase, peak_id, (first_last_spk, last_first_spk)


# ================================= 一般nodes的基类 =================================
"""
注：
    1、模拟的理论时间 : t
    2、模拟的时间步长 : dt
    3、节点基团数量 : N
"""
class Nodes:
    """
        N: 创建节点的数量
        method : 计算非线性微分方程的方法，("euler", "rk4", "discrete")
        dt : 计算步长

        t (float): 模拟的理论时间
    """
    def __init__(self, N, method="euler", dt=0.01):
        self.N = N  # 神经元数量
        self.dt = dt
        # 选择数值计算方法
        self._method = method
        method_map = {"euler": Euler, "rk4": RK4, "discrete": discrete}
        if method not in method_map:
            raise ValueError(f"无效选择，method 必须是 {list(method_map.keys())}")
        self.method = method_map[method]

        self.model = model  # 模型的微分方程

        self._params_f()
        self._vars_f()

    def _params_f(self):
        pass

    def _vars_f(self):
        self.t = 0.  # 运行时间

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
        # I = np.zeros((self.N_vars, self.N))
        # I[axis, :] += Io

        # params_list = list(self.params_nodes.values())
        # self.method(model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        self.t += self.dt  # 时间前进

    def set_vars_vals(self, vars_vals=[0, 0, ...]):
        """
            用于自定义所有状态变量的值
        """
        for i, val in enumerate(vars_vals):
            self.vars_nodes[i] = val

    def cal_flow_field2D(self, select_dim=(0, 1), vars_lim=(-1., 1., -1., 1.), N=100, plt_flag=False):
        """
            二维平面流速场
            select_dim: 选择的维度 (x, y)
            vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max)
            N    : 网格点数
        """
        params_list = list(self.params_nodes.values())
        dX_dt, dY_dt, X, Y = flow_field2D(self.model, params_list, self.N_vars, select_dim=select_dim, vars_lim=vars_lim, N=N)

        if plt_flag:
            plt.streamplot(X, Y, dX_dt, dY_dt, density=1.5, linewidth=1., arrowsize=1.2, arrowstyle='->', color='C1', zorder=0)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Vector Field')

        return dX_dt, dY_dt, X, Y
    
    def cal_flow_field3D(self, select_dim=(0, 1, 2), vars_lim=(-1., 1., -1., 1., -1., 1.), N=100):
        """
            三维流速场
            select_dim: 选择的维度 (x, y, z)
            vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max, z_min, z_max)
            N    : 网格点数
        """
        params_list = list(self.params_nodes.values())
        dX_dt, dY_dt, dZ_dt, X, Y, Z = flow_field3D(self.model, params_list, self.N_vars, select_dim=select_dim, vars_lim=vars_lim, N=N)

        return dX_dt, dY_dt, dZ_dt, X, Y, Z

    def cal_nullclines(self, x_dim=0, y_dim=1, x_range=(0., 1.), dvar_dt=(0, 1), N=100, initial_guesse=None, initial_vars=None, plt_flag=False):
        """
            通用零斜线求解函数
            x_dim       : 指定自变量的维度 int  (自变量的维度)
            y_dim       : 指定求解的目标维度 int  (应变量的维度)
            x_range     : 零斜线自变量范围 (x_min, x_max)
            dvar_dt     : 求解函数的目标维度 
            N           : 零斜线的点数量
            initial_guesse: 指定初始值
            initial_vars: 指定所有变量的值，形状为 (N_vars,)
            plt_flag    : 是否绘制零斜线
        """
        nullclines_list = np.full((2, N), np.nan)
        params_list = list(self.params_nodes.values())
        for i, dv_dt_dim in enumerate(dvar_dt):
            nullclines = find_nullclines(self.model, params_list, self.N_vars, x_dim=x_dim, y_dim=y_dim, dv_dt_dim=dv_dt_dim, x_range=x_range, N=N, initial_guesse=initial_guesse, initial_vars=initial_vars)

            nullclines_list[i] = nullclines

        if plt_flag:
            v_range = np.linspace(x_range[0], x_range[1], N)
            plt.plot(v_range, nullclines_list[0], label=f'var{dvar_dt[0]} nullcline(dvar{dvar_dt[0]}_dt=0)')
            plt.plot(v_range, nullclines_list[1], label=f'var{dvar_dt[1]} nullcline (dvar{dvar_dt[1]}_dt=0)')
            plt.xlabel(f'var{dvar_dt[0]}')
            plt.ylabel(f'var{dvar_dt[1]}')
            plt.title('Nullclines')
            plt.legend()

        return nullclines_list
    
    
@njit
def model(vars, t, I, params):
    res = np.zeros_like(vars)
    # 常数参数
    # params
    # 状态变量
    # vars

    return res


# ================================= 突触模型的基类 =================================
"""
注：
    1、突触权重 w           (size  : [post_N, pre_N])
    2、模拟的理论时间 t     (与突触后的运行时间一致)
    3、模拟的时间步长 dt    (与突触后的运行时间一致)
    4、连接矩阵 conn        (size  : [post_N, pre_N])
    5、突触前和突触后 pre, post
"""
class Synapse:
    """
        pre     :    网络前节点
        post    :    网络后节点
        conn    :    连接矩阵   (size  : [post_N, pre_N])
        synType :    突触类型   ("electr", "chem")
        method  :    计算非线性微分方程的方法，("euler", "rk4", "discrete")
    """
    def __init__(self, pre, post, conn=None, synType="electr", method="euler"):
        # 选择数值计算方法
        self._method = method
        method_map = {"euler": Euler, "rk4": RK4, "discrete": discrete}
        if method not in method_map:
            raise ValueError(f"无效选择，method 必须是 {list(method_map.keys())}")
        self.method = method_map[method]

        # 选择突触类型
        self.synType = synType
        if self.synType == "electr":
            self.syn = syn_electr       # 电突触
        elif self.synType == "chem":
            self.syn = syn_chem         # 化学突触

        self.pre = pre                  # 网络前节点
        self.post = post                # 网络后节点
        self.conn = conn                # 连接矩阵
        self.dt = post.dt               # 计算步长
        self._params_f()
        self._vars_f()

    def _params_f(self):
        # 0维度--post，1维度--pre
        self.w = .1 * np.ones((self.post.N, self.pre.N))  # 设定连接权重

    def _vars_f(self):
        self.t = self.post.t = 0.  # 运行时间

    def __call__(self):
        """
            开头和结尾更新时间(重要)
            self.t = self.post.t
            self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t  # 这个是非常重要的

        # 触前和突触后的状态
        pre_state = [self.pre.vars_nodes[0], self.pre.firingTime, self.pre.flaglaunch.astype(np.float64)]
        post_state = [self.post.vars_nodes[0], self.post.firingTime, self.post.flaglaunch.astype(np.float64)]
        # params_list = list(self.params_syn.values())

        I_post = self.syn(pre_state, post_state, self.w, self.conn)  # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post

@njit
def syn_electr(pre_state, post_state, w, conn, *args):
    """
        电突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        w: 突触权重
        conn: 连接矩阵
    """
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state

    # 计算突触前和突触后的膜电位差值
    vj_vi = pre_mem-post_mem[:, None]     # pre减post
    # 计算突触电流
    Isyn = (w*conn*vj_vi).sum(axis=1)   # 0维度--post，1维度--pre

    return Isyn

@njit
def syn_chem(pre_state, post_state, w, conn, *args):
    """
        化学突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        w: 突触权重
        conn: 连接矩阵
    """
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state

    # return I_syn


# ================================= 演算法 =================================
@njit
def Euler(fun, x0, t, dt, *args):
    """
    使用 euler 方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        x0: 上一个时间单位的状态变量
        t: 运行时间
        dt: 时间步长
    :return: 
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    # 计算下一个时间单位的状态变量
    x0 += dt * fun(x0, t, *args)
    return x0

@njit
def RK4(fun, x0, t, dt, *args):
    """
    使用 Runge-Kutta 方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        x0: 上一个时间单位的状态变量
        t: 运行时间
        dt: 时间步长
    :return:
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    k1 = fun(x0, t, *args)
    k2 = fun(x0 + (dt / 2.) * k1, t + (dt / 2.), *args)
    k3 = fun(x0 + (dt / 2.) * k2, t + (dt / 2.), *args)
    k4 = fun(x0 + dt * k3, t + dt, *args)

    x0 += (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x0

@njit
def discrete(fun, x0, t, dt, *args):
    """
    使用离散方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        x0: 上一个时间单位的状态变量
        t: 运行时间
        dt: 时间步长(设定为1)
    :return:
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    x0[:] = fun(x0, t, *args)
    return x0


# ================================= 常用工具 =================================
# ========= 延迟存储器 =========
class delayer:
    """
        N: 延迟变量数量
        Tn: 延迟时长
    """
    def __init__(self, N, Tn):
        self.N = N                            # 延迟变量数量
        self.delayLong = Tn                   # 延迟时长
        self.k = 0                            # 指针位置    
        self.delay = np.zeros((N, Tn+1))      # 延迟存储矩阵

    def __call__(self, x):
        """
            x: 输入的延迟变量
        """
        delay_o, self.k = delay(x, self.k, self.delayLong, self.delay)

        return delay_o

@njit
def delay(x, k, delayLong, delay):
    """
        x: 输入的延迟变量
        k: 指针位置
        delayLong: 延迟时长
        delay: 延迟存储矩阵
    """
    # 计算延迟位置索引
    delayed_k = (k - delayLong) % (delayLong+1)

    # 输出延迟值
    delay_o = delay[:, delayed_k].copy() 
    k = (k + 1) % (delayLong+1)             # 前进指针
    delay[:, k] = x

    return delay_o, k


# ========= 连接矩阵to拉普拉斯矩阵 =========
@njit
def to_laplacian(adjacency_matrix):
    """
        计算拉普拉斯矩阵
        adjacency_matrix: 邻接矩阵
    """
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    return laplacian_matrix


# ========= 噪声 =========
class noiser:
    """
        噪声产生
        D_noise:    噪声强度
        dt:         计算步步长
        N:          要产生噪声的尺度，可以是整数或元组
        type:       噪声类型，可选：["white"、"color"]
        lam_color:  色噪声的相关率
    """
    def __init__(self, D_noise, dt, N, type="white", lam_color=0.1):
        self.D_noise = D_noise
        self.dt = dt
        self.N = N
        self.lam_color = lam_color

        self.type = type
        type_map = {"white": Gaussian_white_noise, "color": Gaussian_colour_noise}
        if type not in type_map:
            raise ValueError(f"无效选择，type 必须是 {list(type_map.keys())}")
        self.noisee = type_map[type]

        # 初始化噪声
        self.noise = dt*np.random.normal(loc=0., scale=np.sqrt(D_noise*lam_color), size=N)

    def __call__(self):
        """
            产生噪声
        """
        if self.type == "white":
            self.noise = self.noisee(self.D_noise, self.noise, self.dt, self.N)
        elif self.type == "color":
            self.noise = self.noisee(self.D_noise, self.noise, self.dt, self.N, self.lam_color)

@njit
def Gaussian_white_noise(D_noise, noise, dt, size):
    """
        高斯白噪声
        D_noise:    噪声强度
        dt:         计算步步长
        size:       要产生噪声的尺度，可以是整数或元组
    """
    # a = np.random.rand(size)
    # b = np.random.rand(size)
    # noise = np.sqrt(-4*D_noise*dt*np.log(a)) * np.cos(2*np.pi*b)
    noise = np.random.normal(loc=0., scale=np.sqrt(2*D_noise*dt), size=size)
    return noise

@njit
def Gaussian_colour_noise(D_noise, noise, dt, size, lam_color):
    """
        高斯色噪声
        D_noise:    噪声强度
        noise:      上一步的噪声
        dt:         计算步步长
        size:       要产生噪声的尺度，可以是整数或元组
        lam_color:  色噪声的相关率
    """
    # a = np.random.rand(size)
    # b = np.random.rand(size)
    # g_w = np.sqrt(-4*D_noise*dt*np.log(a)) * np.cos(2*np.pi*b)
    g_w = np.random.normal(loc=0., scale=np.sqrt(2*D_noise*dt), size=size) 
    noise = noise - dt * lam_color * noise + lam_color*g_w
    noise = dt * noise

    return noise


# ========= 极大极小值 =========
@njit
def find_extrema(time_series_matrix):
    """
        对多个时间序列矩阵中的每个时间序列，逐个求取极大值和极小值，并返回填充为统一维度的结果。
        
        参数:
            time_series_matrix: 二维数组，形状为(N, T)，包含N个时间序列，每个时间序列有T个时间步。
        
        返回:
            maxima_values (N, max_maxima_len): 极大值
            maxima_indices (N, max_maxima_len): 极大值索引
            minima_values (N, max_minima_len): 极小值
            minima_indices (N, max_minima_len): 极小值索引
    """
    N, T = time_series_matrix.shape
    
    # 初始化存储极大值和极小值的列表
    maxima_values_list = []
    maxima_indices_list = []
    minima_values_list = []
    minima_indices_list = []
    
    for i in range(N):
        time_series = time_series_matrix[i]
        
        # 查找极大值和极小值的索引
        maxima_indices = np.where((time_series[1:-1] > time_series[:-2]) & (time_series[1:-1] > time_series[2:]))[0] + 1
        minima_indices = np.where((time_series[1:-1] < time_series[:-2]) & (time_series[1:-1] < time_series[2:]))[0] + 1
        
        # 获取对应的极大值和极小值
        maxima_values = time_series[maxima_indices]
        minima_values = time_series[minima_indices]
        
        # 将极值和索引填充到统一长度的数组中
        maxima_values_list.append(maxima_values)
        maxima_indices_list.append(maxima_indices)
        minima_values_list.append(minima_values)
        minima_indices_list.append(minima_indices)

    # 手动计算最大长度，用于填充
    max_maxima_len = 0
    max_minima_len = 0
    for max_vals in maxima_values_list:
        if len(max_vals) > max_maxima_len:
            max_maxima_len = len(max_vals)
    for min_vals in minima_values_list:
        if len(min_vals) > max_minima_len:
            max_minima_len = len(min_vals)
    
    # 填充每个时间序列的极大值和极小值，确保它们具有相同长度
    maxima_values_array = np.full((N, max_maxima_len), np.nan)
    maxima_indices_array = np.full((N, max_maxima_len), np.nan)
    minima_values_array = np.full((N, max_minima_len), np.nan)
    minima_indices_array = np.full((N, max_minima_len), np.nan)
    
    for i in range(N):
        maxima_values_array[i, :len(maxima_values_list[i])] = maxima_values_list[i]
        maxima_indices_array[i, :len(maxima_indices_list[i])] = maxima_indices_list[i]
        minima_values_array[i, :len(minima_values_list[i])] = minima_values_list[i]
        minima_indices_array[i, :len(minima_indices_list[i])] = minima_indices_list[i]
    
    return maxima_values_array, maxima_indices_array, minima_values_array, minima_indices_array


# ========= 二维平面流速场 =========
def flow_field2D(fun, params, N_vars, select_dim=(0, 1), vars_lim=(-1., 1., -1., 1.), N=100, initial_vars=None):
    """
        二维平面流速场
        fun  : 速度函数
        params: 速度函数的参数
        N_vars: 速度函数的变量数量
        select_dim: 选择的维度 (x, y)
        vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max)
        N    : 网格点数
        initial_vars: 指定所有变量的值，形状为 (N_vars,)
    """
    if initial_vars is not None:
        initial_vars = np.asarray(initial_vars)
    else:
        initial_vars = np.zeros(N_vars) + 1e-12
    vars = np.broadcast_to(initial_vars[:, np.newaxis, np.newaxis], (N_vars, N, N)).copy()
    # 生成网格
    x_min, x_max, y_min, y_max = vars_lim
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x, y)

    dim1, dim2 = select_dim
    vars[dim1] = X
    vars[dim2] = Y
    
    I = np.zeros(N_vars)
    dvars_dt = fun(vars, 0, I, params)
    dX_dt = dvars_dt[dim1]
    dY_dt = dvars_dt[dim2]

    return dX_dt, dY_dt, X, Y

def flow_field3D(fun, params, N_vars, select_dim=(0, 1, 2), vars_lim=(-1., 1., -1., 1., -1., 1.), N=100, initial_vars=None):
    """
        三维流速场
        fun  : 速度函数
        params: 速度函数的参数
        N_vars: 速度函数的变量数量
        select_dim: 选择的维度 (x, y, z)
        vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max, z_min, z_max)
        N    : 网格点数
        initial_vars: 指定所有变量的值，形状为 (N_vars,)
    """
    if initial_vars is not None:
        initial_vars = np.asarray(initial_vars)
    else:
        initial_vars = np.zeros(N_vars) + 1e-12
    vars = np.broadcast_to(initial_vars[:, np.newaxis, np.newaxis, np.newaxis], (N_vars, N, N, N)).copy()
    # 生成网格
    x_min, x_max, y_min, y_max, z_min, z_max = vars_lim
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    z = np.linspace(z_min, z_max, N)
    X, Y, Z = np.meshgrid(x, y, z)

    dim1, dim2, dim3 = select_dim
    vars[dim1] = X
    vars[dim2] = Y
    vars[dim3] = Z
    
    I = np.zeros(N_vars)
    dvars_dt = fun(vars, 0, I, params)
    dX_dt = dvars_dt[dim1]
    dY_dt = dvars_dt[dim2]
    dZ_dt = dvars_dt[dim3]

    return dX_dt, dY_dt, dZ_dt, X, Y, Z


# ========= 零斜线 nullclines =========
def find_nullclines(fun, params, N_vars, x_dim=0, y_dim=1, dv_dt_dim=0, x_range=(0., 1.), N=100, initial_guesse=None, initial_vars=None):
    """
    通用零斜线求解函数

    Parameters:
        fun         : 模型函数，返回各维度的导数
        params      : 模型参数
        N_vars      : 系统变量的数量
        x_dim       : 指定自变量的维度 int  (自变量的维度)
        y_dim       : 指定求解的目标维度 int  (应变量的维度)
        dv_dt_dim   : 自定零斜线的维度
        x_range     : 零斜线自变量范围 (x_min, x_max)
        N           : 零斜线的点数量
        initial_guesse: 指定初始值
        initial_vars: 指定所有变量的值，形状为 (N_vars,)

    Returns:
        nullcline: 零斜线的值数组，形状为 (N,)
    """
    
    if initial_vars is not None:
        initial_vars = np.asarray(initial_vars)
    else:
        initial_vars = np.zeros(N_vars) + 1e-12
    
    # 尝试多个初始猜测值
    initial_guesses = [0.1, 0.5, 1.0, 5., 10., 50., -0.1, -0.5, -1.0, -5., -10., 50.]
    if initial_guesse is not None:
        initial_guesses.insert(0, initial_guesse)

    x_min, x_max = x_range
    v_range = np.linspace(x_min, x_max, N)  # 自变量的取值范围

    I = np.zeros(N_vars)

    nullcline = []
    for v in v_range:
        # 复制初始变量值
        vars_fixed = initial_vars.copy()
        vars_fixed[x_dim] = v  # 固定自变量的值

        # 定义目标函数，仅对目标维度求解
        def target_func(x):
            vars_fixed[y_dim] = x
            return fun(vars_fixed, None, I, params)[dv_dt_dim]

        # 使用 root 求解目标维度的零斜线
        for guess in initial_guesses:
            sol = root(target_func, guess)
            if sol.success:
                nullcline.append(sol.x[0])  # 成功求解的值
                break                       # 成功求解则跳出循环
        if not sol.success:
            nullcline.append(np.nan)  # 如果求解失败，返回 NaN

    return np.array(nullcline)

