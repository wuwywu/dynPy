# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/1/17
# User      : WuY
# File      : spiral_wave_conn.py
# 螺旋波连接矩阵

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numba import njit, prange
import random

# 定义一个函数将二维索引转换为一维索引
@njit
def index(i, j, N_j):
    """
        将二维索引转换为一维索引。
        i: int, 二维索引的第一个维度。
        j: int, 二维索引的第二个维度。
        N_j: int, 第二个维度的长度。
    """
    return i * N_j + j

# ================= No-flow boundary ================
@njit
def create_diffusion_No_flow2D_4(Nx, Ny):
    """
        创建一个螺旋波连接矩阵(上下左右4点, 无流边界)。
        Nx: int, 网络的一维长度。
        Ny: int, 网络的二维长度。
    """
     # 总节点数
    N = Nx * Ny
    # 初始化扩散矩阵（邻接矩阵），所有值为 0
    diffusion_matrix = np.zeros((N, N), dtype=np.int32)
    
    # 遍历每个节点，连接上下左右的邻居
    for i in range(Nx):
        for j in range(Ny):
            # 当前节点的索引
            current_index = index(i, j, Ny)    
            # 上
            if i > 0:  # 防止越界
                diffusion_matrix[current_index, index(i - 1, j, Ny)] = 1
            # 下
            if i < Nx - 1:  # 防止越界
                diffusion_matrix[current_index, index(i + 1, j, Ny)] = 1
            # 左
            if j > 0:  # 防止越界
                diffusion_matrix[current_index, index(i, j - 1, Ny)] = 1
            # 右
            if j < Ny - 1:  # 防止越界
                diffusion_matrix[current_index, index(i, j + 1, Ny)] = 1
    
    return diffusion_matrix

@njit
def create_diffusion_No_flow2D_8(Nx, Ny):
    """
        创建一个螺旋波连接矩阵(上下左右8点, 无流边界)。
        Nx: int, 网络的一维长度。
        Ny: int, 网络的二维长度。
    """
    # 总节点数
    N = Nx * Ny
    # 初始化扩散矩阵（邻接矩阵），所有值为 0
    diffusion_matrix = np.zeros((N, N), dtype=np.int32)
    
    # 遍历每个节点，连接周围的 8 个邻居，考虑无流边界
    for i in range(Nx):
        for j in range(Ny):
            # 当前节点的索引
            current_index = index(i, j, Ny)
            
            # 上
            if i > 0:  # 如果不是最上行
                up = index(i - 1, j, Ny)
                diffusion_matrix[current_index, up] = 1
            
            # 下
            if i < Nx - 1:  # 如果不是最下行
                down = index(i + 1, j, Ny)
                diffusion_matrix[current_index, down] = 1
            
            # 左
            if j > 0:  # 如果不是最左列
                left = index(i, j - 1, Ny)
                diffusion_matrix[current_index, left] = 1
            
            # 右
            if j < Ny - 1:  # 如果不是最右列
                right = index(i, j + 1, Ny)
                diffusion_matrix[current_index, right] = 1

            # 左上对角线
            if i > 0 and j > 0:  # 如果不是最左上角
                left_up = index(i - 1, j - 1, Ny)
                diffusion_matrix[current_index, left_up] = 1
            
            # 右上对角线
            if i > 0 and j < Ny - 1:  # 如果不是最右上角
                right_up = index(i - 1, j + 1, Ny)
                diffusion_matrix[current_index, right_up] = 1
            
            # 左下对角线
            if i < Nx - 1 and j > 0:  # 如果不是最左下角
                left_down = index(i + 1, j - 1, Ny)
                diffusion_matrix[current_index, left_down] = 1
            
            # 右下对角线
            if i < Nx - 1 and j < Ny - 1:  # 如果不是最右下角
                right_down = index(i + 1, j + 1, Ny)
                diffusion_matrix[current_index, right_down] = 1
    
    return diffusion_matrix


# ================= Periodic boundary ================
@njit
def create_diffusion_periodic2D_4(Nx, Ny):
    """
        创建一个螺旋波连接矩阵(上下左右4点, 周期边界)。
        Nx: int, 网络的一维长度。
        Ny: int, 网络的二维长度。
    """
    # 总节点数
    N = Nx * Ny
    # 初始化扩散矩阵（邻接矩阵），所有值为 0
    diffusion_matrix = np.zeros((N, N), dtype=np.int32)
    
    # 遍历每个节点，连接上下左右的邻居，考虑周期边界
    for i in range(Nx):
        for j in range(Ny):
            # 当前节点的索引
            current_index = index(i, j, Ny)
            
            # 上：周期性边界
            up = index((i - 1) % Nx, j, Ny)
            diffusion_matrix[current_index, up] = 1
            
            # 下：周期性边界
            down = index((i + 1) % Nx, j, Ny)
            diffusion_matrix[current_index, down] = 1
            
            # 左：周期性边界
            left = index(i, (j - 1) % Ny, Ny)
            diffusion_matrix[current_index, left] = 1
            
            # 右：周期性边界
            right = index(i, (j + 1) % Ny, Ny)
            diffusion_matrix[current_index, right] = 1
    
    return diffusion_matrix

@njit
def create_diffusion_periodic2D_8(Nx, Ny):
    """
        创建一个螺旋波连接矩阵(上下左右8点, 周期边界)。
        Nx: int, 网络的一维长度。
        Ny: int, 网络的二维长度。
    """
    # 总节点数
    N = Nx * Ny
    # 初始化扩散矩阵（邻接矩阵），所有值为 0
    diffusion_matrix = np.zeros((N, N), dtype=np.int32)
    
    # 遍历每个节点，连接周围的 8 个邻居，考虑周期边界
    for i in range(Nx):
        for j in range(Ny):
            # 当前节点的索引
            current_index = index(i, j, Ny)
            
            # 上：周期性边界
            up = index((i - 1) % Nx, j, Ny)
            diffusion_matrix[current_index, up] = 1
            
            # 下：周期性边界
            down = index((i + 1) % Nx, j, Ny)
            diffusion_matrix[current_index, down] = 1
            
            # 左：周期性边界
            left = index(i, (j - 1) % Ny, Ny)
            diffusion_matrix[current_index, left] = 1
            
            # 右：周期性边界
            right = index(i, (j + 1) % Ny, Ny)
            diffusion_matrix[current_index, right] = 1

            # 左上对角线：周期性边界
            left_up = index((i - 1) % Nx, (j - 1) % Ny, Ny)
            diffusion_matrix[current_index, left_up] = 1
            
            # 右上对角线：周期性边界
            right_up = index((i - 1) % Nx, (j + 1) % Ny, Ny)
            diffusion_matrix[current_index, right_up] = 1
            
            # 左下对角线：周期性边界
            left_down = index((i + 1) % Nx, (j - 1) % Ny, Ny)
            diffusion_matrix[current_index, left_down] = 1
            
            # 右下对角线：周期性边界
            right_down = index((i + 1) % Nx, (j + 1) % Ny, Ny)
            diffusion_matrix[current_index, right_down] = 1
    
    return diffusion_matrix


if __name__ == "__main__":
    Nx = 5
    Ny = 5
    # diffusion_matrix = create_diffusion_No_flow2D_4(Nx, Ny)
    diffusion_matrix = create_diffusion_No_flow2D_8(Nx, Ny)
    # diffusion_matrix = create_diffusion_periodic2D_4(Nx, Ny)
    # diffusion_matrix = create_diffusion_periodic2D_8(Nx, Ny)
    
    print(diffusion_matrix)
    print(diffusion_matrix.shape)
    print(diffusion_matrix.sum(1))
