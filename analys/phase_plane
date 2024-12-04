# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/12/04
# User      : WuY
# File      : phase_plane.py
# 用于放电相关的统计量

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange


# ================================= 二维平面流速场=================================
def flow_field(fun, params, N_vars, select_dim=(0, 1), vars_lim=(-1., 1., -1., 1.), N=100):
    """
        二维平面流速场
        fun  : 速度函数
        params: 速度函数的参数
        N_vars: 速度函数的变量数量
        select_dim: 选择的维度 (x, y)
        vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max)
        N    : 网格点数
    """
    vars = np.zeros((N_vars, N, N))
    # 生成网格
    x_min, x_max, y_min, y_max = vars_lim
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x, y)

    dim1, dim2 = select_dim
    vars[dim1] = X
    vars[dim2] = Y
    
    I = np.zeros(N_vars)
    dvars_dt = fun(vars, 0, I, params)
    dX_dt = dvars_dt[dim1]
    dY_dt = dvars_dt[dim2]

    return dX_dt, dY_dt, X, Y
