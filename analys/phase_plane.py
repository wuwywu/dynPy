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
from scipy.optimize import root


# ================================= 二维平面流速场=================================
def flow_field2D(fun, params, N_vars, select_dim=(0, 1), vars_lim=(-1., 1., -1., 1.), N=100):
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

def flow_field3D(fun, params, N_vars, select_dim=(0, 1, 2), vars_lim=(-1., 1., -1., 1., -1., 1.), N=100):
    """
        三维流速场
        fun  : 速度函数
        params: 速度函数的参数
        N_vars: 速度函数的变量数量
        select_dim: 选择的维度 (x, y, z)
        vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max, z_min, z_max)
        N    : 网格点数
    """
    vars = np.zeros((N_vars, N, N, N))
    # 生成网格
    x_min, x_max, y_min, y_max, z_min, z_max = vars_lim
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    z = np.linspace(z_min, z_max, N)
    X, Y, Z = np.meshgrid(x, y, z)

    dim1, dim2, dim3 = select_dim
    vars[dim1] = X
    vars[dim2] = Y
    vars[dim3] = Z
    
    I = np.zeros(N_vars)
    dvars_dt = fun(vars, 0, I, params)
    dX_dt = dvars_dt[dim1]
    dY_dt = dvars_dt[dim2]
    dZ_dt = dvars_dt[dim3]

    return dX_dt, dY_dt, dZ_dt, X, Y, Z


# ================================= 零斜线 nullclines =================================
def find_nullclines(fun, params, N_vars, x_dim=0, y_dim=1, dv_dt_dim=0, x_range=(0., 1.), N=100, initial_guesse=None, initial_vars=None):
    """
    通用零斜线求解函数

    Parameters:
        fun         : 模型函数，返回各维度的导数
        params      : 模型参数
        N_vars      : 系统变量的数量
        x_dim       : 指定自变量的维度 int  (自变量的维度)
        y_dim       : 指定求解的目标维度 int  (应变量的维度)
        dv_dt_dim   : 自定零斜线的维度
        x_range     : 零斜线自变量范围 (x_min, x_max)
        N           : 零斜线的点数量
        initial_guesse: 指定初始值
        initial_vars: 指定所有变量的值，形状为 (N_vars,)

    Returns:
        nullcline: 零斜线的值数组，形状为 (N,)
    """
    
    if initial_vars is not None:
        initial_vars = np.asarray(initial_vars)
    else:
        initial_vars = np.zeros(N_vars) + 1e-12
    
    # 尝试多个初始猜测值
    initial_guesses = [0.1, 0.5, 1.0, 5., 10., 50., -0.1, -0.5, -1.0, -5., -10., 50.]
    if initial_guesse is not None:
        initial_guesses.insert(0, initial_guesse)

    x_min, x_max = x_range
    v_range = np.linspace(x_min, x_max, N)  # 自变量的取值范围

    I = np.zeros(N_vars)

    nullcline = []
    for v in v_range:
        # 复制初始变量值
        vars_fixed = initial_vars.copy()
        vars_fixed[x_dim] = v  # 固定自变量的值

        # 定义目标函数，仅对目标维度求解
        def target_func(x):
            vars_fixed[y_dim] = x
            return fun(vars_fixed, None, I, params)[dv_dt_dim]

        # 使用 root 求解目标维度的零斜线
        for guess in initial_guesses:
            sol = root(target_func, guess)
            if sol.success:
                nullcline.append(sol.x[0])  # 成功求解的值
                break                       # 成功求解则跳出循环
        if not sol.success:
            nullcline.append(np.nan)  # 如果求解失败，返回 NaN

    return np.array(nullcline)


