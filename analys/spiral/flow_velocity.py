# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/5/20
# User      : WuY
# File      : flow_velocity.py
# 用于计算流速
# ref: Li et al.: Finding type and location of the source of cardiac arrhythmias from the averaged flow velocity field using the determinant-trace method. Phys. Rev. E. 104, 064401 (2021). https://doi.org/10.1103/PhysRevE.104.064401

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

class FlowVelocity:
    """
        用于计算流速的类。
        
        参数：
        V_dalay : 一个延迟时间的变量视图，形状为(Nx, Ny)
        alpha : 一种防止估计导数中噪声的加权因子
        n_iterations : 迭代次数 (迭代过程收敛)
    """
    def __init__(self, V_dalay, alpha=12., n_iterations=128):
        self.V_dalay = V_dalay
        self.alpha = alpha
        self.n_iterations = n_iterations

    def __call__(self, V):
        """
            计算流速的函数

            参数：
                V (ndarray): 当前时间的变量视图，形状为(Nx, Ny)。
        """
        # 计算梯度
        dv_dx, dv_dy, dv_dt = calculate_gradients(self.V_dalay, V)

        # 计算平均流速
        v_x_avg, v_y_avg = calculate_velocity(dv_dx, dv_dy, dv_dt, self.alpha, self.n_iterations)

        return v_x_avg, v_y_avg
    
    def reduce_density(self, v_x_avg, v_y_avg, v_x_size=30, v_y_size=30):
        """
            减少流速密度的函数(块平均法)

            参数：  
                v_x_avg (ndarray): 平均x方向的流速，形状为(Nx, Ny)。
                v_y_avg (ndarray): 平均y方向的流速，形状为(Nx, Ny)。
                v_x_size (int): x方向的流速密度大小。
                v_y_size (int): y方向的流速密度大小。
        """
        return reduce_density(v_x_avg, v_y_avg, v_x_size, v_y_size)
    

def reduce_density(v_x_avg, v_y_avg, v_x_size=30, v_y_size=30):
    """
        减少流速密度的函数(块平均法)

        参数：  
            v_x_avg (ndarray): 平均x方向的流速，形状为(Nx, Ny)。
            v_y_avg (ndarray): 平均y方向的流速，形状为(Nx, Ny)。
            v_x_size (int): x方向的流速密度大小。
            v_y_size (int): y方向的流速密度大小。
    """
    Ny, Nx = v_x_avg.shape

    # Step 1: 补一行一列（复制第1行、第1列）
    vx_ext = np.vstack([v_x_avg[0:1, :], v_x_avg])       # 顶部复制第1行
    vx_ext = np.hstack([vx_ext[:, 0:1], vx_ext])         # 左边复制第1列

    vy_ext = np.vstack([v_y_avg[0:1, :], v_y_avg])
    vy_ext = np.hstack([vy_ext[:, 0:1], vy_ext])

    # Step 2: 每块大小（xy） = [block_h]*v_y_size, [block_w]*v_x_size
    block_h = vx_ext.shape[0] // v_y_size
    block_w = vx_ext.shape[1] // v_x_size

    # Step 3: 裁剪为整除大小
    H_trim = block_h * v_y_size
    W_trim = block_w * v_x_size
    vx_trim = vx_ext[:H_trim, :W_trim]
    vy_trim = vy_ext[:H_trim, :W_trim]

    # Step 4: reshape 成 block，并对每块求平均
    vx_blocks = vx_trim.reshape(v_y_size, block_h, v_x_size, block_w)
    vy_blocks = vy_trim.reshape(v_y_size, block_h, v_x_size, block_w)

    vx_reduced = vx_blocks.mean(axis=(1, 3))  # 对每个 block 的行列平均
    vy_reduced = vy_blocks.mean(axis=(1, 3))

    return vx_reduced, vy_reduced

def calculate_gradients(v_dalay, v):
    """
        计算梯度的函数

        参数：
            v_dalay (ndarray): 一个延迟时间的变量视图，形状为(Nx, Ny)。
            v (ndarray): 当前时间的变量视图，形状为(Nx, Ny)。
    """
    dv_dx = (
        v_dalay[1:, :-1] - v_dalay[:-1, :-1] +
        v_dalay[1:, 1:]  - v_dalay[:-1, 1:]  +
        v[1:, :-1]       - v[:-1, :-1]       +
        v[1:, 1:]        - v[:-1, 1:]
    ) / 4.
             
    dv_dy = (
        v_dalay[:-1, 1:] - v_dalay[:-1, :-1] +
        v_dalay[1:, 1:]  - v_dalay[1:, :-1]  +
        v[:-1, 1:]       - v[:-1, :-1]       +
        v[1:, 1:]        - v[1:, :-1]
    ) / 4.

    dv_dt = (
        v_dalay[:-1, :-1] - v[:-1, :-1] +
        v_dalay[:-1, 1:]  - v[:-1, 1:]  +
        v_dalay[1:, :-1]  - v[1:, :-1]  +
        v_dalay[1:, 1:]   - v[1:, 1:]
    ) / 4.

    return dv_dx, dv_dy, dv_dt

def calculate_velocity(dv_dx, dv_dy, dv_dt, alpha, n_iterations):
    """
        计算流速的函数

        参数：
            dv_dx (ndarray): x方向的梯度。
            dv_dy (ndarray): y方向的梯度。
            dv_dt (ndarray): 时间方向的梯度。
            alpha (float): 加权因子。
            n_iterations (int): 迭代次数。
    """
    # 初始化
    v_x = np.zeros_like(dv_dx)
    v_y = np.zeros_like(dv_dy)
    v_x_avg = np.zeros_like(dv_dx)
    v_y_avg = np.zeros_like(dv_dy)

    # 迭代
    for _ in range(n_iterations):
        # 更新速度场（数据项）
        numerator = v_x_avg * dv_dx + v_y_avg * dv_dy + dv_dt
        denominator = alpha**2 + dv_dx**2 + dv_dy**2

        v_x = v_x_avg - dv_dx * numerator / denominator
        v_y = v_y_avg- dv_dy * numerator / denominator

        # 添加无流边界
        v_x_temp = np.pad(v_x, pad_width=1, mode='edge')
        v_y_temp = np.pad(v_y, pad_width=1, mode='edge')

        # 计算平均值
        v_x_avg = (
                    v_x_temp[0:-2, 1:-1] + v_x_temp[2:, 1:-1] +     # 上下
                    v_x_temp[1:-1, 0:-2] + v_x_temp[1:-1, 2:]       # 左右
                ) / 6.0 + (
                    v_x_temp[0:-2, 0:-2] + v_x_temp[0:-2, 2:] +     # 左上、右上
                    v_x_temp[2:, 0:-2] + v_x_temp[2:, 2:]           # 左下、右下
                ) / 12.0
       
        v_y_avg = (
                    v_y_temp[0:-2, 1:-1] + v_y_temp[2:, 1:-1] +
                    v_y_temp[1:-1, 0:-2] + v_y_temp[1:-1, 2:]
                ) / 6.0 + (
                    v_y_temp[0:-2, 0:-2] + v_y_temp[0:-2, 2:] +
                    v_y_temp[2:, 0:-2] + v_y_temp[2:, 2:]
                ) / 12.0

    return v_x_avg, v_y_avg


if __name__ == "__main__":
    # 定义简单输入
    v_now = np.ones((5, 5))

    v_prev = np.ones((5, 5))*2
    v_prev[0, 0] = 1
    
    dvx, dvy, dvt = calculate_gradients(v_prev, v_now)    
    # print(dvx)
    # print(dvy)
    # print(dvt)
    v_x, v_y = calculate_velocity(dvx, dvy, dvt, alpha=12., n_iterations=2)
    # vx_reduced, vy_reduced = reduce_density(v_x, v_y, v_x_size=30, v_y_size=30)
    print("Python 结果：")
    print("v_x:")
    print(v_x[:, 0])

    print("v_y:")
    print(v_y)

    # print("Python 结果：")
    # print("vx_reduced:")
    # print(np.round(vx_reduced, 4))
    # print("vy_reduced:")
    # print(np.round(vy_reduced, 4))
