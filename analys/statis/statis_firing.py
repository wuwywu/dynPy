# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/20
# User      : WuY
# File      : statis_firing.py
# 用于放电相关的统计量

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


# ================================= 记录峰值时间 =================================
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


# ================================= 通过峰值时间计算有效的 ISI =================================
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


# ================================= 计算每个神经元的 CV（变异系数） =================================
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


# ================================= 极大极小值 =================================
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

