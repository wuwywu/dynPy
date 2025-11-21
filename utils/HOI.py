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
    post_ids, pre_ids1, pre_ids2 = np.nonzero(binary_conn)

    # 提取对应权重
    rows, cols, depth = weight.shape
    indices = post_ids * rows * cols + pre_ids1 * cols + pre_ids2
    weights = weight.ravel()[indices]  # 一维索引提取权重
    # 将结果整合为一个三列矩阵
    # ids_and_weights = np.vstack((pre_ids, pre2_ids, post_ids, weights))

    return pre_ids1, pre_ids2, post_ids, weights


## 通过张量写出广义的 Laplacian 矩阵
### numba版
@njit
def laplacian_from_tensor(A):
    """
    A: (N, N, ..., N) 的 (d+1) 阶邻接张量
       A[i, j1, ..., jd] = a^{(d)}_{i j1...jd}

    返回:
        L  : (N, N)  的广义 Laplacian
        Ki : (N,)    的 K_i^{(d)}
        Kij: (N, N)  的 K_{ij}^{(d)}
    """
    D = A.ndim          # = d+1
    N = A.shape[0]
    d = D - 1           # 公式里的 d

    # ---- 阶乘 d! 和 (d-1)!，纯循环 ----
    fact_d = 1
    for k in range(2, d + 1):
        fact_d *= k

    fact_d1 = 1
    for k in range(2, d):
        fact_d1 *= k
    if d == 1:
        fact_d1 = 1     # 0! = 1

    # ---- 结果数组 ----
    Ki  = np.zeros(N)          # (N,)
    Kij = np.zeros((N, N))     # (N, N)

    # ---- 一次性遍历 A 的所有元素，线性索引 -> 多维索引 ----
    A_flat = A.ravel()
    total  = A_flat.size

    idx = np.empty(D, np.int64)

    for p in range(total):
        v = A_flat[p]

        # 还原 p 对应的 (i, j1, ..., jd)
        r = p
        for dim in range(D - 1, -1, -1):
            idx[dim] = r % N
            r //= N

        i0 = idx[0]   # 第一个指标 i
        j0 = idx[1]   # 第二个指标 j

        # ∑_{j1...jd} a_{i j1...jd}  -> Ki
        Ki[i0] += v

        # ∑_{j1...j_{d-1}} a_{i j j1...} -> Kij
        # 这里统一处理：不区分 d=1 或 d>1，第二个指标就是 j
        Kij[i0, j0] += v

    # ---- 归一化得到 K_i^{(d)} 和 K_{ij}^{(d)} ----
    Ki = Ki / fact_d

    Kij = Kij / fact_d1

    # ---- 构造 L^{(d)} ----
    L = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                # off-diagonal: - (d-1)! * K_ij^{(d)}
                L[i, j] = -fact_d1 * Kij[i, j]
            else:
                # diagonal: d! * K_i^{(d)}
                L[i, j] = fact_d * Ki[i]

    return L, Ki, Kij

### 非numba版
def laplacian_from_tensor(A):
    """
    A: (N, N, ..., N) 的 (d+1) 阶邻接张量
       A[i, j1, ..., jd] = a^{(d)}_{i j1...jd}

    返回:
        L  : (N, N)  的广义 Laplacian
        Ki : (N,)    的 K_i^{(d)}  （已经除以 d!）
        Kij: (N, N)  的 K_{ij}^{(d)}（已经除以 (d-1)!）
    """
    A = np.asarray(A)
    D = A.ndim          # = d+1
    if D < 2:
        raise ValueError("A 至少需要是 2 阶张量 (N,N,...)")

    N = A.shape[0]
    d = D - 1           # 公式里的 d

    # ---- 阶乘 d! 和 (d-1)! ----
    fact_d  = math.factorial(d)
    fact_d1 = math.factorial(d - 1) if d > 1 else 1  # 0! = 1

    # ---- Ki: 按 j1...jd 求和，再除以 d! ----
    # 轴 1..D-1 全部求和，保留 i 这一维
    sum_axes_Ki = tuple(range(1, D))
    Ki = A.sum(axis=sum_axes_Ki) / fact_d           # 形状 (N,)

    # ---- Kij: 按 j2...jd 求和，再除以 (d-1)! ----
    if d == 1:
        # d=1 时，Kij^{(1)} = a_{ij} 本身（除以 0! = 1 不变）
        Kij = A.astype(float, copy=True)            # 形状 (N,N)
    else:
        # 对轴 2..D-1 求和，保留 i 和 j1（此时记为 j）
        sum_axes_Kij = tuple(range(2, D))
        Kij = A.sum(axis=sum_axes_Kij) / fact_d1    # 形状 (N,N)

    # ---- 构造 L^{(d)} ----
    # 先用非对角公式：L_ij = - (d-1)! * K_ij^{(d)}
    L = -fact_d1 * Kij

    # 再把对角改成：L_ii = d! * K_i^{(d)}
    # np.fill_diagonal 会就地修改
    np.fill_diagonal(L, fact_d * Ki)

    return L, Ki, Kij

