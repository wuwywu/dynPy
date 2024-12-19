# encoding: utf-8
# Author    : HuXY
# Datetime  : 2024/12/19
# User      : HuXY
# File      : Random_timevaring.py
# Erdős-Rényi（E-R）模型
# ref：Xueyan Hu, Dynamical rewiring promotes synchronization in memristive FitzHugh-Nagumo neuronal networks,Chaos Soliton Fract.184,115047(2024)
# 描述：这个代码实现了随机网络模型的动态重布线，这种方式会使得网络的拓扑结构会随机发生变化。


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

# np.random.seed(2024)
# random.seed(2024)

# 随机网络，G(n, p)模型  概率p连接一条边, 单向
@njit
def create_ER_p(n, p, seed=None):
    """
    n-网络的总节点数
    p-网络中节点间连接的概率
    seed-随机数种子
    """
    if seed is not None:
        np.random.seed(seed)
    random_matrix = np.random.rand(n, n)
    connection_matrix = (random_matrix < p).astype(np.int32)
    np.fill_diagonal(connection_matrix, 0)  #不考虑自连接
    return connection_matrix

@njit
def Random_rewiring(connMat, N, eta):  # 重布线规则
    """
    connMat: 连接矩阵
    N:       网络总节点数
    eta:     重布线范围(eta部分的节点进行重布线)
    """
    # 获取连接矩阵元素为1的位置
    one_row_indices, one_col_indices = np.where(connMat == 1)  

    # 获取连接矩阵元素为0的位置
    zero_connMat = np.eye(N)+connMat       
    zero_row_indices, zero_col_indices = np.where(zero_connMat == 0)

    # 选择要重布线的元素数量
    num_elements_to_change = int(one_row_indices.size*eta)   

    # 计算连接矩阵元素为1和0的数量
    one_size = one_row_indices.size
    zero_size = zero_row_indices.size

    # 保证要重布线的元素数量小于等于连接矩阵元素为1和0的数量
    if one_size >= num_elements_to_change and zero_size >= num_elements_to_change:
        # 随机选择要重布线的元素
        one_random = np.random.choice(one_size, num_elements_to_change, replace=False)
        # 重布线:元素为1的变为0
        for i in range(one_random.size):
            connMat[one_row_indices[one_random[i]], one_col_indices[one_random[i]]] = 0

        # 重布线:元素为0的变为1
        zero_random = np.random.choice(zero_size, num_elements_to_change, replace=False)
        for i in range(zero_random.size):
            connMat[zero_row_indices[zero_random[i]], zero_col_indices[zero_random[i]]] = 1

    return connMat


if __name__ == "__main__":
    n = 100
    p = 0.005
    eta = 0.1
    
    connMat = create_ER_p(n, p)
    print(connMat)
    Random_rewiring(connMat, n, eta)
    print(connMat)
