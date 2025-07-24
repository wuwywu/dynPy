# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/5/22
# User      : WuY
# File      : phases_analys.py
# 使用希尔伯特变换求出 幅度， 频率， 瞬时频率。

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
from scipy.signal import hilbert
from scipy.signal import find_peaks, detrend
import matplotlib.pyplot as plt


def tohilbert(signal, time=None):
    """
    signal : 输入信号[ndim, Ntime]
    time   : 时间 [N, ] (如果不输入则无法求出 瞬时频率)
    """
    # 进行希尔伯特变换
    analytic_signal = hilbert(signal)

    # 计算幅度
    amplitude = np.abs(analytic_signal)

    # 计算相位
    phase = np.angle(analytic_signal)

    # 计算瞬时频率
    if time is not None:
        instantaneous_frequency = np.diff(np.unwrap(phase)) / (2 * np.pi * np.diff(time))
    else : instantaneous_frequency = None

    # 归一化相位
    phase = (phase + np.pi) / (2*np.pi)

    return amplitude, phase, instantaneous_frequency


def calculate_complete_phases(phases):
    """
    计算完整的相位变化：
        2*pi*N + phase(0-1)
    注意：
        希尔伯特变换求出相位后，将首尾的误差去掉，截取中间的有效部分
    """
    complete_phases = []
    complete_phases.append(phases[0] * 2 * np.pi)
    for i in range(1, len(phases)):
        diff = phases[i] - phases[i - 1]
        if diff < -0.9:
            diff += 1
        complete_phase = complete_phases[i - 1] + diff * 2 * np.pi
        complete_phases.append(complete_phase)
    return np.array(complete_phases)


def calculate_mean_frequency(signal, time, threshold=None, min_distance=None, detrend_signal=True):
    """
    从原始信号计算平均频率
    
    参数:
    signal: 输入信号数组 [Ntime, ]
    time: 时间数组 [Ntime, ]，与信号对应
    threshold: 检测峰值的阈值，None时自动计算
    min_distance: 两个峰值之间的最小样本数(少于这个距离的峰值将被忽略)
    detrend_signal: 是否对信号进行去趋势处理
    
    返回:
    mean_freq: 平均频率 (Hz)
    peak_times: 检测到的峰值时间点
    peak_indices: 检测到的峰值在信号中的索引
    """
    # 确保输入是numpy数组
    signal = np.asarray(signal)
    time = np.asarray(time)
    
    # 检查输入长度是否匹配
    if len(signal) != len(time):
        raise ValueError("信号和时间数组长度必须相同")
    
    # 去趋势处理，减少基线漂移影响
    if detrend_signal:
        signal = detrend(signal)
    
    # 检测信号峰值（假设信号的峰值代表放电事件）
    peak_indices, _ = find_peaks(
        signal, 
        height=threshold, 
        distance=min_distance
    )

    # 如果没有检测到足够的峰值，返回0
    if len(peak_indices) < 2:
        return 0.0, np.array([]), peak_indices
    
    # 获取峰值对应的时间点
    peak_times = time[peak_indices]
    
    # 计算总时间（从第一个峰值到最后一个峰值）
    total_duration = peak_times[-1] - peak_times[0]
    
    # 避免除以零
    if total_duration <= 0:
        return 0.0, peak_times, peak_indices
    
    # 计算平均频率：(峰值数量-1) / 总持续时间
    # 减1是因为n个峰值有n-1个间隔
    mean_freq = (len(peak_times) - 1) / total_duration
    
    return mean_freq, peak_times, peak_indices


if __name__ == "__main__":
    # 生成多个示例信号
    t = np.linspace(0, 10, 1000)
    signal1 = np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)
    signal2 = np.cos(2 * np.pi * 3 * t) + 0.3 * np.cos(2 * np.pi * 4 * t)

    signals = np.array([signal1, signal2])

    amplitudes, phases, instantaneous_frequencies = tohilbert(signals, t)

    # # 计算平均频率
    t = np.linspace(0, 10, 1000)
    freq = 5
    signal = np.sin(2 * np.pi * freq * t)
    # signal = np.cos(2 * np.pi * 3 * t)
    mean_freq, peak_times, peak_indices = calculate_mean_frequency(
        signal, 
        t
    )
    
    print(f"检测到 {len(peak_times)} 个峰值")
    print(f"计算得到的平均频率: {mean_freq:.2f} Hz")
    print(f"理论频率: {freq} Hz")

    # 绘制结果
    fig, axs = plt.subplots(4, 2, figsize=(14, 8))

    axs[0, 0].plot(t, signals[0])
    axs[0, 0].set_title('Signal 1')
    axs[0, 1].plot(t, signals[1])
    axs[0, 1].set_title('Signal 2')

    axs[1, 0].plot(t, amplitudes[0])
    axs[1, 0].set_title('Amplitude of Signal 1')
    axs[1, 1].plot(t, amplitudes[1])
    axs[1, 1].set_title('Amplitude of Signal 2')

    axs[2, 0].plot(t, phases[0])
    axs[2, 0].set_title('Phase of Signal 1')
    axs[2, 1].plot(t, phases[1])
    axs[2, 1].set_title('Phase of Signal 2')

    axs[3, 0].plot(t[:-1], instantaneous_frequencies[0])
    axs[3, 0].set_title('Instantaneous Frequency of Signal 1')
    axs[3, 1].plot(t[:-1], instantaneous_frequencies[1])
    axs[3, 1].set_title('Instantaneous Frequency of Signal 2')

    plt.tight_layout()
    plt.show()
