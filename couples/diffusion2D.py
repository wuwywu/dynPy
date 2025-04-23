# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/4/24
# User      : WuY
# File      : diffusion2D.py
# 2D扩散网络(晶格网络，无流边界，周期边界，4邻接，8邻接)
# 使用5点中心差分法，9点差分法计算耦合

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


class Diffusion2D:
    """
        2D扩散网络(晶格网络，无流边界，周期边界，4邻接，8邻接)
        使用5点中心差分法，9点差分法计算耦合
        D: (numpy.ndarray or float), 扩散系数(耦合强度)
        boundary: str, 边界条件，可选值为"No_flow", "periodic"
        adjacency: int, 邻接数，可选值为4, 8
    """
    def __init__(self, D, boundary="No_flow", adjacency=4):
        self.D = D
        boundary_list = ["No_flow", "periodic"]
        adjacency_list = [4, 8]
        if boundary not in boundary_list:
            raise ValueError("boundary must be in {}".format(boundary_list))
        if adjacency not in adjacency_list:
            raise ValueError("adjacency must be in {}".format(adjacency_list))
        self.boundary = boundary
        self.adjacency = adjacency
        if boundary == "No_flow":
            if adjacency == 4:
                self.diffusion = diffusion_No_flow2D_4
            elif adjacency == 8:
                self.diffusion = diffusion_No_flow2D_8
        elif boundary == "periodic":
            if adjacency == 4:
                self.diffusion = diffusion_periodic2D_4
            elif adjacency == 8:
                self.diffusion = diffusion_periodic2D_8

    def __call__(self, vars_view):
        """
            vars_view: np.array, 变量视图，形状为(Nx, Ny)
        """
        I = self.D * self.diffusion(vars_view)

        return I.ravel()
        

# 不同的邻接数和周期方式
@njit
def diffusion_No_flow2D_4(vars_view):
    """
        无流边界的4邻接扩散网络
        vars_view: np.array, 变量视图，形状为(Nx, Ny)
    """
    up = np.vstack((vars_view[0:1, :], vars_view[:-1, :]))
    down = np.vstack((vars_view[1:, :], vars_view[-1:, :]))
    left = np.hstack((vars_view[:, 0:1], vars_view[:, :-1]))
    right = np.hstack((vars_view[:, 1:], vars_view[:, -1:]))

    return up + down + left + right - 4 * vars_view

@njit
def diffusion_No_flow2D_8(vars_view):
    """
        无流边界的8邻接扩散网络
        vars_view: np.array, 变量视图，形状为(Nx, Ny)
    """
     # 上
    up = np.vstack((vars_view[0:1, :], vars_view[:-1, :]))
    # 下
    down = np.vstack((vars_view[1:, :], vars_view[-1:, :]))
    # 左
    left = np.hstack((vars_view[:, 0:1], vars_view[:, :-1]))
    # 右
    right = np.hstack((vars_view[:, 1:], vars_view[:, -1:]))
    # 左上
    up_left = np.vstack((left[0:1, :], left[:-1, :]))
    # 右上
    up_right = np.vstack((right[0:1, :], right[:-1, :]))
    # 左下
    down_left = np.vstack((left[1:, :], left[-1:, :]))
    # 右下
    down_right = np.vstack((right[1:, :], right[-1:, :]))

    return up + down + left + right + up_left + up_right + down_left + down_right - 8 * vars_view

@njit
def diffusion_periodic2D_4(vars_view):
    """
        周期边界的4邻接扩散网络
        vars_view: np.array, 变量视图，形状为(Nx, Ny)
    """
    # 上移
    up = np.vstack((vars_view[-1:, :], vars_view[:-1, :]))
    # 下移
    down = np.vstack((vars_view[1:, :], vars_view[:1, :]))
    # 左移
    left = np.hstack((vars_view[:, -1:], vars_view[:, :-1]))
    # 右移
    right = np.hstack((vars_view[:, 1:], vars_view[:, :1]))

    return up + down + left + right - 4 * vars_view

@njit
def diffusion_periodic2D_8(vars_view):
    """
        周期边界的8邻接扩散网络
        vars_view: np.array, 变量视图，形状为(Nx, Ny)
    """
    # 上移
    up = np.vstack((vars_view[-1:, :], vars_view[:-1, :]))
    # 下移
    down = np.vstack((vars_view[1:, :], vars_view[:1, :]))
    # 左移
    left = np.hstack((vars_view[:, -1:], vars_view[:, :-1]))
    # 右移
    right = np.hstack((vars_view[:, 1:], vars_view[:, :1]))

    # 左上移
    up_left = np.vstack((left[-1:, :], left[:-1, :]))
    # 右上移
    up_right = np.vstack((right[-1:, :], right[:-1, :]))
    # 左下移
    down_left = np.vstack((left[1:, :], left[:1, :]))
    # 右下移
    down_right = np.vstack((right[1:, :], right[:1, :]))

    return up + down + left + right + up_left + up_right + down_left + down_right - 8 * vars_view
