# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/1/16
# User      : WuY
# File      : base_fun.py
# 文件中包含：

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
from scipy.optimize import root
from scipy.sparse import coo_matrix
import random
from PIL import Image
import io

"""
数值模拟算法(欧拉，龙格库塔，离散):
    1). euler 方法 : Euler
    2). rk4 方法   : RK4
    3). 离散方法    : discrete
"""

"""
神经元模型中的函数工具
    1). 峰值测算函数        :  spikes_eval  
    2). 记录峰值时间函数    :   record_spike_times
    3). 计算ISI函数         :  calculate_isi
    4). 计算CV函数          :  calculate_cv
    5). 计算KOP函数(同步)   :  calculate_kuramoto
    5). 二\三维平面流速场   :  flow_field2D/flow_field3D
    6). 零斜线(nullclines) :  find_nullclines
"""

"""
突触模型中的函数工具
    1). 矩阵转稀疏矩阵      : matrix_to_sparse
    2). 稀疏矩阵转矩阵      : sparse_to_matrix 
"""

"""
常用的工具函数
    1). 延迟存储器              :   delayer
    2). 连接矩阵to拉普拉斯矩阵   :   to_laplacian
    3). 噪声产生器              :   noiser
    4). 极大极小值              :   find_extrema
    5). numba版计算特征值       :   eigval_qr
    6). 螺旋波动态显示器        :   spiral_wave_display
    7). 状态变量(膜电位)动态显示器  :   state_variable_display
"""


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
def Heun(fun, x0, t, dt, *args):
    """
    使用 Heun 方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程函数，形式为 fun(x, t, *args)
        x0: 上一个时间单位的状态变量 (numpy.ndarray)
        t: 当前时间
        dt: 时间步长
    return:
        x1 (numpy.ndarray): 下一个时间单位的状态变量
    """
    # 计算当前点的斜率
    k1 = fun(x0, t, *args)
    
    # 使用 Euler 法预测值
    x_pred = x0 + dt * k1
    
    # 在预测点上计算新的斜率
    k2 = fun(x_pred, t + dt, *args)
    
    # 加权平均斜率得到新的状态
    x0 += 0.5 * dt * (k1 + k2)
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


# ================================= 神经元模型中的函数工具 =================================
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


# ================================= 节点模型中的函数工具 =================================

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


# ================================= 突触模型中的函数工具 =================================
@njit
def matrix_to_sparse(conn, weight_matrix=None):
    """
    将矩阵转换为稀疏矩阵
  
        参数：
        - conn: 连接矩阵(二元矩阵)
        - weight_matrix: 权重矩阵(可选)
    
        返回：
        - pre_ids   : 前节点的id
        - post_ids  : 后节点的id
        - weights   : 节点对的权重
    """
    # 将 conn 转换为二元矩阵
    binary_conn = np.where(conn != 0, 1, 0)
  
    # 如果未提供权重矩阵，则默认为全1矩阵
    if weight_matrix is None:
        weight = np.ones_like(conn, dtype=np.float64)
    else:
        weight = np.asarray(weight_matrix, dtype=np.float64)

    # 确保 binary_conn 和 weight_matrix 形状一致
    if binary_conn.shape != weight.shape:
        raise ValueError("binary_conn 和 weight_matrix 的形状必须一致！")
  
    # 提取非零元素的行列索引
    post_ids, pre_ids = np.nonzero(binary_conn)

    # 提取对应权重
    rows, cols = weight.shape
    indices =  post_ids * rows + pre_ids  # 计算一维索引
    weights = weight.ravel()[indices]  # 一维索引提取权重

    # 将结果整合为一个三列矩阵
    # ids_and_weights = np.vstack((pre_ids, post_ids, weights))

    return pre_ids, post_ids, weights

@njit
def sparse_to_matrix(N_pre, N_post, pre_ids, post_ids, weights):
    """
    将稀疏矩阵信息转换为连接矩阵和权重矩阵
    
        参数：
        - N: int, 神经元总数
        - pre_ids: np.ndarray, 前节点id
        - post_ids: np.ndarray, 后节点id
        - weights: np.ndarray, 权重
    
        返回：
        - conn: np.ndarray, 二元连接矩阵 
        - weight_matrix: np.ndarray, 权重矩阵 
    """
    # 将输入转换为整数类型
    pre_ids, post_ids = pre_ids.astype(np.int32), post_ids.astype(np.int32)
    # 初始化连接矩阵和权重矩阵
    conn = np.zeros((N_post, N_pre), dtype=np.int32)
    weight_matrix = np.zeros((N_post, N_pre), dtype=np.float64)
  
    # 扁平化索引
    flat_indices = post_ids * N_pre + pre_ids

    # 将对应位置设置为连接
    conn_flat = conn.ravel()
    conn_flat[flat_indices] = 1

    # 将权重赋值
    weight_flat = weight_matrix.ravel()
    weight_flat[flat_indices] = weights

    return conn, weight_matrix

# ========= COO 稀疏化 =========
def to_sparse_matrix(matrix):
    """
        sparse_matrix.toarray() 方法将 COO 格式的稀疏矩阵转换为密集矩阵。
        pre_ids : sparse_matrix.col
        post_ids : sparse_matrix.row
        weights : sparse_matrix.data
    """
    # 找到非零元素
    nonzero_indices = np.nonzero(matrix)
    rows = nonzero_indices[0]
    cols = nonzero_indices[1]
    # 获取非零元素的值
    values = matrix[rows, cols]
    # 得到 COO-format 稀疏矩阵
    sparse_matrix = coo_matrix((values, (rows, cols)), shape=matrix.shape)
    return sparse_matrix
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


# ================================= 计算特征值(Numba版) =================================
@njit
def eigvals_qr(A, num_iterations=1000):
    """
    使用QR分解计算矩阵的特征值

        参数:
            A: 要计算特征值的矩阵
            num_iterations: 迭代次数

        返回:
            特征值数组
    """
    n = A.shape[0]
    A_k = np.ascontiguousarray(A.copy())  # 复制原始矩阵
    
    for _ in range(num_iterations):
        # QR分解
        Q, R = np.linalg.qr(A_k)
        A_k = np.dot(R, Q)  # 迭代更新矩阵
    
    # 对角元素即为特征值
    return np.diag(A_k)


# ========= 螺旋波动态显示器 =========
class show_spiral_wave:
    """
    螺旋波动态显示器
        var: 要显示的变量
        Nx:  网络的一维长度
        Ny:  网络的二维长度
        save_gif: 是否保存gif动图
    """
    def __init__(self, var, Nx, Ny, save_gif=False):
        self.var = var
        self.Nx = Nx
        self.Ny = Ny
        self.save_gif = save_gif
        self.frames = []
        plt.ion()

    def __call__(self, i, t=None, show_interval=1000):
        """
            i: 迭代次数
            t: 时间
            show_interval: 显示间隔
        """
        if i % show_interval <= 0.001:
            var = self.var.reshape(self.Nx, self.Ny)
            plt.clf()
            plt.imshow(var, cmap="jet", origin="lower", aspect="auto")
            if t is not None:
                plt.title(f"t={t:.3f}")
            else:
                plt.title(f"i={i}")
            plt.colorbar()
            plt.pause(0.0000000000000000001)

            if self.save_gif:
                buffer_ = io.BytesIO()
                plt.savefig(buffer_, format='png')
                buffer_.seek(0)
                self.frames.append(Image.open(buffer_))

    def save_image(self, filename="animation.gif", duration=50):
        """
            保存gif动图
            filename: 文件名
            duration: 每帧持续时间(ms)
        """
        if self.save_gif:
            self.frames[0].save(filename, save_all=True, append_images=self.frames[1:], duration=duration, loop=0)

    def show_final(self):
        """ 
            显示最终图像
        """
        plt.ioff()
        var = self.var.reshape(self.Nx, self.Ny)
        plt.imshow(var, cmap="jet", origin="lower", aspect="auto")
        plt.colorbar()
        plt.show()  # 停止交互模式，显示最终图像


# ========= 状态变量(膜电位)动态显示器 =========
class show_state:
    """
    状态变量动态显示器
        N_var:      要显示的变量数
        N_time:     要显示时间宽度
        dt:         计算步步长
        save_gif:   是否保存gif动图
    """ 
    def __init__(self, N_var, N_show=50_00, dt=0.01, save_gif=False):
        self.N_var = N_var
        self.N_time = N_show
        self.vars = np.full((N_show, N_var), np.nan)
        self.dt = dt
        self.count = 0
        self.save_gif = save_gif
        self.frames = []
        plt.ion()

    def __call__(self, var, i, t=None, show_interval=1000, pause=0.0000000000000000001):
        """
            var : 状态变量
            i   : 迭代次数
            t   : 时间
            show_interval: 显示间隔(更新频率)
            pause: 暂停时间
        """
        if len(var) != self.N_var:
            raise ValueError("var的长度与N_var不一致")
        
        if self.count < self.N_time:
            self.vars[self.count] = var
        else:
            self.vars[:-1] = self.vars[1:]
            self.vars[-1] = var

        self.count += 1
        
        if i % show_interval <= 0.001:
            plt.clf()   # 清除当前图像
            if t is not None:
                if t < self.N_time*self.dt:
                    self.time_vec = np.linspace(0, self.N_time*self.dt, self.N_time)
                else:
                    self.time_vec = np.linspace(t-self.N_time*self.dt, t, self.N_time)
            else:        
                if i < self.N_time:
                    self.time_vec = np.linspace(0, self.N_time*self.dt, self.N_time)  
                else:
                    self.time_vec = np.linspace((i-self.N_time)*self.dt, i*self.dt, self.N_time)
            self.vars_temp = self.vars.copy()
            plt.plot(self.time_vec, self.vars_temp)
            plt.xlim(self.time_vec[0], self.time_vec[-1])
            plt.pause(pause)

            if self.save_gif:
                buffer_ = io.BytesIO()
                plt.savefig(buffer_, format='png')
                buffer_.seek(0)
                self.frames.append(Image.open(buffer_))

    def save_image(self, filename="animation.gif", duration=50):
        """
            保存gif动图
            filename: 文件名
            duration: 每帧持续时间(ms)
        """
        if self.save_gif:
            self.frames[0].save(filename, save_all=True, append_images=self.frames[1:], duration=duration, loop=0)

    def show_final(self):
        """
            显示最终图像
        """
        plt.ioff()
        plt.plot(self.time_vec, self.vars_temp)
        plt.xlim(self.time_vec[0], self.time_vec[-1])
        plt.show()  # 停止交互模式，显示最终图像
     
