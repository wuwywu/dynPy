# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/13
# User      : WuY
# File      : utils_f.py
# 一些工具

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# ================================= 记录神经元发放峰的时间 =================================
class spikevent:
    """
        神经元模型的峰值收集器
        N: 收集 N 个神经的尖峰事件
    """
    def __init__(self, N):
        self.N = N
        self.Tspike_list = [[] for i in range(N)]

    def __call__(self, t, flaglaunch):
        """
        t: 模拟实时时间
        flaglaunch: 是否尖峰的标志 (放电开启标志)
        """
        # 找出符合条件的神经元索引
        indices = np.where(flaglaunch > 0.9)[0]
        # 使用列表推导式直接将时间添加到符合条件的神经元中
        [self.Tspike_list[i].append(t) for i in indices]

    def pltspikes(self):
        plt.eventplot(self.Tspike_list)


# ================================= 状态变量延迟 =================================
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


# ================================= 连接矩阵to拉普拉斯矩阵 =================================
@njit
def to_laplacian(adjacency_matrix):
    """
        计算拉普拉斯矩阵
        adjacency_matrix: 邻接矩阵
    """
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    return laplacian_matrix


#  ================================= 噪声 =================================
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


if __name__ == "__main__":
    N = 1
    noisee = noiser(D_noise=0.01, dt=0.01, N=N, type="white")
    # noisee = noiser(D_noise=0.01, dt=0.01, N=N, type="color", lam_color=.1)

    noise_ = []
    for i in range(1000):
        noisee()
        noise_.append(noisee.noise.copy())

    noise_ = np.array(noise_)

    plt.plot(noise_)
    plt.show()
