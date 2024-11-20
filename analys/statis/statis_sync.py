# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/15
# User      : WuY
# File      : statis_sync.py
# 用于同步研究的统计量


import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


# ================================= 同步因子 =================================
class synFactor:
    """
        计算同步因子  
        N: 需要计算变量的数量
        Tn: 计算次数(int), Time/dt

        描述：一个用于量化同步的归一化统计量（范围在0-1之间）。
                越趋近与1，同步越好；
                越趋近与0，同步越差。
    """
    def __init__(self, N, Tn):
        self.N = N      # 矩阵大小
        self.Tn = Tn    # 计算次数
        self.count = 0  # 统计计算次数

        # 初始化累加器
        self.up1 = 0.
        self.up2 = 0.
        self.down1 = np.zeros(N)
        self.down2 = np.zeros(N)

    def __call__(self, x):
        self.up1, self.up2, self.count = cal_synFactor(x, self.up1, self.up2, self.down1, self.down2, self.count, self.Tn)

    def return_syn(self):
        return return_synFactor(self.up1, self.up2, self.down1, self.down2, self.count, self.Tn)
    
@njit
def cal_synFactor(x, up1, up2, down1, down2, count, Tn):
    """
        计算同步因子
    """
    F = np.mean(x)
    up1 += F * F / Tn
    up2 += F / Tn
    down1 += x * x / Tn
    down2 += x / Tn
    count += 1  # 计算次数叠加W

    return up1, up2, count

@njit
def return_synFactor(up1, up2, down1, down2, count, Tn):
    """
        返回同步因子
    """
    if count != Tn:
        print(f"输入计算次数{Tn},实际计算次数{count}")

    down = np.mean(down1 - down2 ** 2)
    if down > -0.000001 and down < 0.000001:
        return 1.
    
    up = up1 - up2 ** 2

    return up / down


# ================================= Kuramoto Order Parameter (KOP) =================================
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
        raise ValueError("没有满足条件的神经元，请检查输入数据或降低 min_spikes 的值！")
    
    first_spikes = np.array(first_spikes, dtype=np.float64)
    last_spikes = np.array(last_spikes, dtype=np.float64)

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

