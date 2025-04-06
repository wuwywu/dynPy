# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/04/06
# User      : WuY 
# File      : HOI.py
# Description : 包含高阶交互作用(Higher Order Interaction)的相关函数

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


# 构建高阶网络(二阶，三维矩阵)
## 找出一级交互中的所有三角形
@njit
def find_triangles(adjacency_matrix):
    """
    找到给定邻接矩阵中的所有三角形。
    
    参数:
        adjacency_matrix (np.ndarray): 邻接矩阵
    
    返回:
        list: 三角形的列表，每个三角形由三个节点组成
    """
    n = adjacency_matrix.shape[0]
    triangles = []
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] == 1:
                for k in range(j + 1, n):
                    if adjacency_matrix[i, k] == 1 and adjacency_matrix[j, k] == 1:
                        triangles.append((i, j, k))
    return triangles


## 构建二阶高阶网络
@njit
def build_second_order_network(N, triangles):
    """
    构建二阶网络。
    参数:
        N (int): 网络中节点的数量
        triangles (list): 三角形的列表

    返回:
        np.ndarray: 二阶网络的连接矩阵(post, pre1, pre2)
    """
    # 创建一个三维矩阵，大小为 (N, N, N)，用于存储二阶网络的连接关系
    second_order_network = np.zeros((N, N, N), dtype=np.int32)

    for index, (i, j, k) in enumerate(triangles):
        second_order_network[i, j, k] = 1
        second_order_network[i, k, j] = 1
        second_order_network[j, i, k] = 1
        second_order_network[j, k, i] = 1
        second_order_network[k, i, j] = 1
        second_order_network[k, j, i] = 1

    return second_order_network


## 三角形计数矩阵 (降维高阶网络)
@njit
def create_triangle_count_matrix(N, triangles):
    """
    创建一个二阶网络的三角形计数矩阵。
    
    参数:
        N (int): 网络中节点的数量
        triangles (list): 三角形的列表
    
    返回:
        np.ndarray: 二阶网络的三角形计数矩阵
    """
    triangle_count_matrix = np.zeros((N, N), dtype=np.int32)
    for index, (i, j, k) in enumerate(triangles):
        triangle_count_matrix[i, j] += 1
        triangle_count_matrix[j, i] += 1
        triangle_count_matrix[i, k] += 1
        triangle_count_matrix[k, i] += 1
        triangle_count_matrix[j, k] += 1
        triangle_count_matrix[k, j] += 1

    return triangle_count_matrix 


## 稀疏三维矩阵
@njit
def matrix_to_sparse_3d(conn, weight_matrix=None):
    """
    将三维矩阵转换为稀疏矩阵
    
    参数:
        conn (np.ndarray): 三维连接矩阵
        weight_matrix (np.ndarray): 权重矩阵(可选)
    
    返回:
        tuple: 稀疏矩阵的行索引、列索引和权重
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
    post_ids, pre_ids, pre2_ids = np.nonzero(binary_conn)

    # 提取对应权重
    rows, cols, depth = weight.shape
    indices = post_ids * rows * cols + pre_ids * cols + pre2_ids
    weights = weight.ravel()[indices]  # 一维索引提取权重
    # 将结果整合为一个三列矩阵
    # ids_and_weights = np.vstack((pre_ids, pre2_ids, post_ids, weights))

    return pre_ids, pre2_ids, post_ids, weights

