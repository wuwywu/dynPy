# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/13
# User      : WuY
# File      : base_mods.py
# 文件中包含：
"""
本模块包含：
    1、节点动力学基础模块(神经元，一般节点)
    2、突触动力学基础模块 
"""

"""
    神经元基础模块包含的功能：
        1、计算ISI (cal_isi)
        2、计算CV (cal_cv)
        3、计算Kuramoto Order Parameter (cal_kop)
        4、计算峰值时间 (return_spike_times)
        注：这些功能使用前需要打开峰值时间记录器(record_spike_times = True)

        5、cal_flow_field2D/flow_field3D (计算2/3维的速度场)
        6、cal_nullclines (计算零斜线)   
"""

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit, prange
from scipy.optimize import root
import random
# 导入函数工具
from base_fun import *

# os.environ['NUMBA_NUM_THREADS'] = '4' # 限制使用的线程数量
# np.random.seed(2024)
# random.seed(2024)

# ================================= 神经元模型的基类 =================================
"""
    注：
        1、模拟的理论时间 : t
        2、模拟的时间步长 : dt
        3、神经元基团数量 : N
    关于放电：
        4、放电开始的阈值 : th_up
        5、放电结束的阈值 : th_down
        6、放电的标志 flag (只要大于0就处于放电过程中)
        7、放电开启的标志 flaglaunch (只有在放电的时刻为1， 其他时刻为0)
        8、放电的时间 firingTime (与放电的标志时刻一致)
        9、峰值时间记录器 spike_times (size: [N, max_spikes])
        10、峰值时间计数器 spike_counts (size: [N])
"""
class Neurons:
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，("euler", "heun", "rk4", "discrete")
        dt : 计算步长

        params_f (dict): 节点模型参数
        
        t (float): 模拟的理论时间
    """
    def __init__(self, N, method="euler", dt=0.01):
        self.N = N  # 神经元数量
        self.dt = dt
        # 选择数值计算方法
        self._method = method
        method_map = {"euler": Euler, "heun": Heun, "rk4": RK4, "discrete": discrete}
        if method not in method_map:
            raise ValueError(f"无效选择，method 必须是 {list(method_map.keys())}")
        self.method = method_map[method]

        self.model = model  # 模型的微分方程

        self.fun_switch()
        self.fun_sets()
        self._params_f()
        self._vars_f()

    def _params_f(self):
        self.th_up = 0      # 放电阈上值
        self.th_down = -10  # 放电阈下值

    def _vars_f(self):
        self.t = 0  # 运行时间
        # 模型放电变量
        self.flag = np.zeros(self.N, dtype=np.int32)           # 模型放电标志(>0, 放电)
        self.flaglaunch = np.zeros(self.N, dtype=np.int32)     # 模型开始放电标志(==1, 放电刚刚开始)
        self.firingTime = np.zeros(self.N)                     # 记录放电时间(上次放电)
        # 初始化峰值时间记录相关变量
        self.max_spikes = 1000                                 # 假设每个神经元最多记录 1000 次放电
        self.spike_times = np.full((self.N, self.max_spikes), np.nan)
        self.spike_counts = np.zeros(self.N, dtype=np.int32)   # 放电次数计数

    def __call__(self, Io=0, axis=[0]):
        """
        args:
            Io: 输入到神经元模型的外部激励，
                shape:
                    (len(axis), self.num)
                    (self.num, )
                    float
            axis: 需要加上外部激励的维度
                list
        """
        axis = np.asarray(axis)
        I = np.zeros((self.N_vars, self.N))
        I[axis, :] += Io

        params_list = list(self.params_nodes.values())
        self.method(self.model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        if self.spiking: 
            self._spikes_eval(self.vars_nodes[0], self.t, self.th_up, self.th_down, self.flag, self.flaglaunch, self.firingTime)  # 放电测算

            if self.record_spike_times:
                # 调用单独的记录峰值时间的函数
                self._record_spike_times(self.flaglaunch, self.t, self.spike_times, self.spike_counts, self.max_spikes)

        self.t += self.dt  # 时间前进

    def set_vars_vals(self, vars_vals=[0, 0, ...]):
        """
            用于自定义所有状态变量的值，可以用于更改网络大小(节点数)
            vars_vals:
                1、长度为 N_vars 的列表
                2、维度为 (N_vars, N) 的数组      
        """
        self._vars()
        self._vars_f()
        for i, val in enumerate(vars_vals):
            self.vars_nodes[i] = val * np.ones(self.N)

    def fun_switch(self):
        """
            功能开关
        """
        self.spiking = True                 # spiking 计算
        self.record_spike_times = False     # 记录spiking时间
    
    def fun_sets(self):
        """
            功能集合
        """
        self._spikes_eval = spikes_eval                 # spiking 计算函数
        self._record_spike_times = record_spike_times   # 记录spiking时间函数

    def return_spike_times(self):
        """
            返回有效的spiking时间
        """
        if not self.record_spike_times:
            raise ValueError("未启用峰值时间记录功能，无法返回spiking时间。")
        
        valid_spike_times = [self.spike_times[i, ~np.isnan(self.spike_times[i])] for i in range(self.N)]

        return valid_spike_times
    
    def cal_isi(self):
        """
            计算每个神经元的有效 ISI（脉冲间隔），去除 NaN 值，保留相同长度的 ISI 数组。

            返回：
                isi_array (ndarray): 二维数组，形状为 (N, max_valid_isi_count)，包含每个神经元的 ISI，未使用的元素填充为 0。
        """
        if not self.record_spike_times:
            raise ValueError("未启用峰值时间记录功能，无法计算 ISI。")

        isi_array = calculate_isi(self.spike_times, self.spike_counts, self.N)
        
        return isi_array
    
    def cal_cv(self):
        """
            计算每个神经元的有效 CV
            变异系数 The coefficient of variation (CV)
                CV=1，会得到泊松尖峰序列（稀疏且不连贯的尖峰）。
                CV<1，尖峰序列变得更加规则，并且对于周期性确定性尖峰，CV 趋近与0。
                CV>1，对应于比泊松过程变化更大的尖峰点过程。

            返回：
                cv_array (ndarray): 一维数组，长度为 N，包含每个神经元的 CV，未使用的元素填充为 0。
        """
        if not self.record_spike_times:
            raise ValueError("未启用峰值时间记录功能，无法计算 CV。")   

        cv_array = calculate_cv(self.spike_times, self.spike_counts, self.N)

        return cv_array
    
    def cal_kop(self, min_spikes=10):
        """
            调用 Kuramoto Order Parameter 计算。
            返回：
                mean_kop (float): 平均 Kuramoto Order Parameter。
                kuramoto (ndarray): 每个时间点的 Kuramoto Order Parameter。
                phase (ndarray): 每个神经元的相位矩阵。
                peak_id (ndarray): 每个神经元的峰值编号（计算完整相位变化）。
                valid_interval (tuple): (first_last_spk, last_first_spk)，有效计算的时间区间。
        """
        if not self.record_spike_times:
            raise ValueError("未启用峰值时间记录功能，无法计算 Kuramoto Order Parameter。")   
        
        return calculate_kuramoto(self.spike_times, self.dt, min_spikes=min_spikes)
    
    def cal_flow_field2D(self, select_dim=(0, 1), vars_lim=(-1., 1., -1., 1.), N=100, initial_vars=None, plt_flag=False):
        """
            二维平面流速场
            select_dim: 选择的维度 (x, y)
            vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max)
            initial_vars: 指定所有变量的值，形状为 (N_vars,)
            N    : 网格点数
            plt_flag : 是否绘制图像
        """
        params_list = list(self.params_nodes.values())
        dX_dt, dY_dt, X, Y = flow_field2D(self.model, params_list, self.N_vars, select_dim=select_dim, vars_lim=vars_lim, N=N, initial_vars=initial_vars)

        if plt_flag:
            plt.streamplot(X, Y, dX_dt, dY_dt, density=1.5, linewidth=1., arrowsize=1.2, arrowstyle='->', color='C1', zorder=0)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Vector Field')

        return dX_dt, dY_dt, X, Y
    
    def cal_flow_field3D(self, select_dim=(0, 1, 2), vars_lim=(-1., 1., -1., 1., -1., 1.), N=100, initial_vars=None):
        """
            三维流速场
            select_dim: 选择的维度 (x, y, z)
            vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max, z_min, z_max)
            N    : 网格点数
            initial_vars: 指定所有变量的值，形状为 (N_vars,)
        """
        params_list = list(self.params_nodes.values())
        dX_dt, dY_dt, dZ_dt, X, Y, Z = flow_field3D(self.model, params_list, self.N_vars, select_dim=select_dim, vars_lim=vars_lim, N=N, initial_vars=initial_vars)

        return dX_dt, dY_dt, dZ_dt, X, Y, Z

    def cal_nullclines(self, x_dim=0, y_dim=1, x_range=(0., 1.), dvar_dt=(0, 1), N=100, initial_guesse=None, initial_vars=None, plt_flag=False):
        """
            通用零斜线求解函数
            x_dim       : 指定自变量的维度 int  (自变量的维度)
            y_dim       : 指定求解的目标维度 int  (应变量的维度)
            x_range     : 零斜线自变量范围 (x_min, x_max)
            dvar_dt     : 求解函数的目标维度 
            N           : 零斜线的点数量
            initial_guesse: 指定初始值
            initial_vars: 指定所有变量的值，形状为 (N_vars,)
            plt_flag    : 是否绘制零斜线
        """
        nullclines_list = np.full((2, N), np.nan)
        params_list = list(self.params_nodes.values())
        for i, dv_dt_dim in enumerate(dvar_dt):
            nullclines = find_nullclines(self.model, params_list, self.N_vars, x_dim=x_dim, y_dim=y_dim, dv_dt_dim=dv_dt_dim, x_range=x_range, N=N, initial_guesse=initial_guesse, initial_vars=initial_vars)

            nullclines_list[i] = nullclines

        if plt_flag:
            v_range = np.linspace(x_range[0], x_range[1], N)
            plt.plot(v_range, nullclines_list[0], label=f'var{dvar_dt[0]} nullcline(dvar{dvar_dt[0]}_dt=0)')
            plt.plot(v_range, nullclines_list[1], label=f'var{dvar_dt[1]} nullcline (dvar{dvar_dt[1]}_dt=0)')
            plt.xlabel(f'var{dvar_dt[0]}')
            plt.ylabel(f'var{dvar_dt[1]}')
            plt.title('Nullclines')
            plt.legend()

        return nullclines_list

@njit
def model(vars, t, I, params):
    res = np.zeros_like(vars)
    # 常数参数
    # params
    # 状态变量
    # vars

    return res
    

# ================================= 一般nodes的基类 =================================
"""
注：
    1、模拟的理论时间 : t
    2、模拟的时间步长 : dt
    3、节点基团数量 : N
"""
class Nodes:
    """
        N: 创建节点的数量
        method : 计算非线性微分方程的方法，("euler", "heun", "rk4", "discrete")
        dt : 计算步长

        t (float): 模拟的理论时间
    """
    def __init__(self, N, method="euler", dt=0.01):
        self.N = N  # 神经元数量
        self.dt = dt
        # 选择数值计算方法
        self._method = method
        method_map = {"euler": Euler, "heun": Heun, "rk4": RK4, "discrete": discrete}
        if method not in method_map:
            raise ValueError(f"无效选择，method 必须是 {list(method_map.keys())}")
        self.method = method_map[method]

        self.model = model  # 模型的微分方程

        self._params_f()
        self._vars_f()

    def _params_f(self):
        pass

    def _vars_f(self):
        self.t = 0.  # 运行时间

    def __call__(self, Io=0, axis=[0]):
        """
        args:
            Io: 输入到神经元模型的外部激励，
                shape:
                    (len(axis), self.num)
                    (self.num, )
                    float
            axis: 需要加上外部激励的维度
                list
        """
        # I = np.zeros((self.N_vars, self.N))
        # I[axis, :] += Io

        # params_list = list(self.params_nodes.values())
        # self.method(model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        self.t += self.dt  # 时间前进

    def set_vars_vals(self, vars_vals=[0, 0, ...]):
        """
            用于自定义所有状态变量的值
        """
        for i, val in enumerate(vars_vals):
            self.vars_nodes[i] = val

    def cal_flow_field2D(self, select_dim=(0, 1), vars_lim=(-1., 1., -1., 1.), N=100, plt_flag=False):
        """
            二维平面流速场
            select_dim: 选择的维度 (x, y)
            vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max)
            N    : 网格点数
        """
        params_list = list(self.params_nodes.values())
        dX_dt, dY_dt, X, Y = flow_field2D(self.model, params_list, self.N_vars, select_dim=select_dim, vars_lim=vars_lim, N=N)

        if plt_flag:
            plt.streamplot(X, Y, dX_dt, dY_dt, density=1.5, linewidth=1., arrowsize=1.2, arrowstyle='->', color='C1', zorder=0)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('Vector Field')

        return dX_dt, dY_dt, X, Y
    
    def cal_flow_field3D(self, select_dim=(0, 1, 2), vars_lim=(-1., 1., -1., 1., -1., 1.), N=100):
        """
            三维流速场
            select_dim: 选择的维度 (x, y, z)
            vars_lim: 速度函数的变量范围 (x_min, x_max, y_min, y_max, z_min, z_max)
            N    : 网格点数
        """
        params_list = list(self.params_nodes.values())
        dX_dt, dY_dt, dZ_dt, X, Y, Z = flow_field3D(self.model, params_list, self.N_vars, select_dim=select_dim, vars_lim=vars_lim, N=N)

        return dX_dt, dY_dt, dZ_dt, X, Y, Z

    def cal_nullclines(self, x_dim=0, y_dim=1, x_range=(0., 1.), dvar_dt=(0, 1), N=100, initial_guesse=None, initial_vars=None, plt_flag=False):
        """
            通用零斜线求解函数
            x_dim       : 指定自变量的维度 int  (自变量的维度)
            y_dim       : 指定求解的目标维度 int  (应变量的维度)
            x_range     : 零斜线自变量范围 (x_min, x_max)
            dvar_dt     : 求解函数的目标维度 
            N           : 零斜线的点数量
            initial_guesse: 指定初始值
            initial_vars: 指定所有变量的值，形状为 (N_vars,)
            plt_flag    : 是否绘制零斜线
        """
        nullclines_list = np.full((2, N), np.nan)
        params_list = list(self.params_nodes.values())
        for i, dv_dt_dim in enumerate(dvar_dt):
            nullclines = find_nullclines(self.model, params_list, self.N_vars, x_dim=x_dim, y_dim=y_dim, dv_dt_dim=dv_dt_dim, x_range=x_range, N=N, initial_guesse=initial_guesse, initial_vars=initial_vars)

            nullclines_list[i] = nullclines

        if plt_flag:
            v_range = np.linspace(x_range[0], x_range[1], N)
            plt.plot(v_range, nullclines_list[0], label=f'var{dvar_dt[0]} nullcline(dvar{dvar_dt[0]}_dt=0)')
            plt.plot(v_range, nullclines_list[1], label=f'var{dvar_dt[1]} nullcline (dvar{dvar_dt[1]}_dt=0)')
            plt.xlabel(f'var{dvar_dt[0]}')
            plt.ylabel(f'var{dvar_dt[1]}')
            plt.title('Nullclines')
            plt.legend()

        return nullclines_list
    
    
@njit
def model(vars, t, I, params):
    res = np.zeros_like(vars)
    # 常数参数
    # params
    # 状态变量
    # vars

    return res


# ================================= 突触模型的基类 =================================
"""
注：
    1、突触权重 w           (size  : [post_N, pre_N])
    2、模拟的理论时间 t     (与突触后的运行时间一致)
    3、模拟的时间步长 dt    (与突触后的运行时间一致)
    4、连接矩阵 conn        (size  : [post_N, pre_N])
    5、突触前和突触后 pre, post
    6、突触前和突触后的ids,以及对应的权重  : pre_ids, post_ids, w_sparse
"""
class Synapse:
    """
        pre     :    网络前节点
        post    :    网络后节点
        conn    :    连接矩阵   (size  : [post_N, pre_N])
        synType :    突触类型   ("electr", "chem")
        method  :    计算非线性微分方程的方法，("euler", "heun", "rk4", "discrete")
    """
    def __init__(self, pre, post, conn=None, synType="electr", method="euler"):
        # 选择数值计算方法
        self._method = method
        method_map = {"euler": Euler, "heun": Heun, "rk4": RK4, "discrete": discrete}
        if method not in method_map:
            raise ValueError(f"无效选择，method 必须是 {list(method_map.keys())}")
        self.method = method_map[method]

        # 选择突触类型
        self.synType = synType
        if self.synType == "electr":
            self.syn = syn_electr_sparse       # 电突触(稀疏)
        elif self.synType == "chem":
            self.syn = syn_chem_sparse         # 化学突触(稀疏)

        self.pre = pre                  # 网络前节点
        self.post = post                # 网络后节点
        self.conn = conn                # 连接矩阵
        self.dt = post.dt               # 计算步长
        self._params_f()
        self._vars_f()
        self.to_sparse()
        # self.to_dense()

    def _params_f(self):
        # 0维度--post，1维度--pre
        self.w = .1 * np.ones((self.post.N, self.pre.N))  # 设定连接权重

    def _vars_f(self):
        self.t = self.post.t = 0.  # 运行时间

    def __call__(self):
        # 触前和突触后的状态
        pre_state = (self.pre.vars_nodes[0], self.pre.firingTime, self.pre.flaglaunch)
        post_state = (self.post.vars_nodes[0], self.post.firingTime, self.post.flaglaunch)

        return self.forward_sparse(pre_state, post_state)

    def forward_sparse(self, pre_state, post_state):
        """
            开头和结尾更新时间(重要)
            self.t = self.post.t
            self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t  # 这个是非常重要的
        # params_list = list(self.params_syn.values())

        I_post = self.syn(pre_state, post_state, self.pre_ids, self.post_ids, self.w_sparse, self.t)  # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post
          
    def forward_dense(self, pre_state, post_state):
        """
            使用矩阵形式计算突触电流(不建议使用) 
            开头和结尾更新时间(重要)
            self.t = self.post.t
            self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t  # 这个是非常重要的
        # params_list = list(self.params_syn.values())

        I_post = self.syn(pre_state, post_state, self.w, self.conn, self.t)  # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post
        
    def to_sparse(self):
        self.pre_ids, self.post_ids, self.w_sparse = matrix_to_sparse(self.conn, self.w)

    def to_dense(self):
        self.conn, self.w = sparse_to_matrix(self.pre.N, self.post.N, self.pre_ids, self.post_ids, self.w_sparse)


@njit
def syn_electr_sparse(pre_state, post_state, pre_ids, post_ids, weights, *args):
    """
        电突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        pre_ids: 突触前神经元的索引
        post_ids: 突触后神经元的索引
        weights: 突触权重
    """
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state

    # 计算膜电位差 (vj - vi)
    vj_vi = pre_mem[pre_ids] - post_mem[post_ids]

    # 计算突触电流贡献
    currents = weights * vj_vi

    # 神经元数量
    num_neurons = len(post_mem)  # 突触后神经元总数

    # 累积电流贡献到突触后神经元
    Isyn = np.bincount(post_ids, weights=currents, minlength=num_neurons)

    return Isyn

@njit
def syn_chem_sparse(pre_state, post_state, pre_ids, post_ids, weights, *args):
    """
        化学突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        pre_ids: 突触前神经元的索引
        post_ids: 突触后神经元的索引
        weights: 突触权重
    """
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state

    # 神经元数量
    num_neurons = len(post_mem)  # 突触后神经元总数

    # 累积电流贡献到突触后神经元
    # Isyn = np.bincount(post_ids, weights=currents, minlength=num_neurons)

    # return I_syn

# =============================== 矩阵计算神经网络(不建议使用) ===============================
@njit
def syn_electr(pre_state, post_state, w, conn, *args):
    """
        电突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        w: 突触权重
        conn: 连接矩阵
    """
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state

    # 计算突触前和突触后的膜电位差值
    vj_vi = pre_mem-post_mem[:, None]     # pre减post
    # 计算突触电流
    Isyn = (w*conn*vj_vi).sum(axis=1)   # 0维度--post，1维度--pre

    return Isyn

@njit
def syn_chem(pre_state, post_state, w, conn, *args):
    """
        化学突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        w: 突触权重
        conn: 连接矩阵
    """
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state

    # return I_syn

