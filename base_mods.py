# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/13
# User      : WuY
# File      : base_mods.py
# 文件中包含：
# 1、节点动力学基础模块(神经元，一般节点)
# 2、突触动力学基础模块
# 3、数值模拟算法(欧拉，龙格库塔，离散)
# 4、常用的工具函数(延迟存储器)


import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random

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
"""
class Neurons:
    """
        N : 建立神经元的数量
        method : 计算非线性微分方程的方法，("euler", "rk4", "discrete")
        dt : 计算步长
        spiking : 是否计算神经元的放电(True, False)

        params_f (dict): 节点模型参数
        
        t (float): 模拟的理论时间
    """
    def __init__(self, N, method="euler", dt=0.01, spiking=True):
        self.N = N  # 神经元数量
        self.dt = dt
        # 选择数值计算方法
        self.method = method
        method_options = ["euler", "rk4", "discrete"]
        if method not in method_options:
            raise ValueError(f"无效选择，method在{method_options}选择")
        if method == "euler":   self.method = Euler
        elif method == "rk4":   self.method = RK4
        elif method == "discrete":   self.method = discrete

        if spiking:  
            self.spiking = spiking
            self._spikes_eval = spikes_eval
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
        self.firingTime = np.zeros(self.N)                # 记录放电时间(上次放电)

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
        # I[0, :] = self.Iex  # 恒定的外部激励
        # I[axis, :] += Io

        # params_list = list(self.params_nodes.values())
        # self.method(model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        # if self.spiking: 
        #     self._spikes_eval(self.vars_nodes[0], self.params_f, self.flag, self.flaglaunch, self.firingTime)  # 放电测算

        self.t += self.dt  # 时间前进

@njit
def spikes_eval(mem, t, th_up, th_down, flag, flaglaunch, firingTime):
    """
        在非人工神经元中，计算神经元的 spiking
    """
    # -------------------- 放电开始 --------------------
    flaglaunch[:] = 0                                           # 重置放电开启标志
    firing_StartPlace = np.where((mem > th_up) & (flag == 0))   # 放电开始的位置
    flag[firing_StartPlace] = 1                                 # 放电标志改为放电
    flaglaunch[firing_StartPlace] = 1                           # 放电开启标志
    firingTime[firing_StartPlace] = t                           # 记录放电时间

    #  -------------------- 放电结束 -------------------
    firing_endPlace = np.where((mem < th_down) & (flag == 1))   # 放电结束的位置
    flag[firing_endPlace] = 0                                   # 放电标志改为放电


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
        method : 计算非线性微分方程的方法，("euler", "rk4", "discrete")
        dt : 计算步长

        t (float): 模拟的理论时间
    """
    def __init__(self, N, method="euler", dt=0.01):
        self.N = N  # 神经元数量
        self.dt = dt
        # 选择数值计算方法
        self.method = method
        method_options = ["euler", "rk4", "discrete"]
        if method not in method_options:
            raise ValueError(f"无效选择，method在{method_options}选择")
        if method == "euler":   self.method = Euler
        elif method == "rk4":   self.method = RK4
        elif method == "discrete":   self.method = discrete

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
        # Iex = self.params_nodes["Iex"]      # 恒定的外部激励
        # I = np.zeros((self.N_vars, self.N))
        # I[0, :] = self.Iex  # 恒定的外部激励
        # I[axis, :] += Io

        # params_list = list(self.params_nodes.values())
        # self.method(model, self.vars_nodes, self.t, self.dt, I, params_list)  #

        self.t += self.dt  # 时间前进


# ================================= 突触模型的基类 =================================
"""
注：
    1、突触权重 w           (size  : [post_N, pre_N])
    2、模拟的理论时间 t     (与突触后的运行时间一致)
    3、模拟的时间步长 dt    (与突触后的运行时间一致)
    4、连接矩阵 conn        (size  : [post_N, pre_N])
    5、突触前和突触后 pre, post
"""
class Synapse:
    """
        pre     :    网络前节点
        post    :    网络后节点
        conn    :    连接矩阵   (size  : [post_N, pre_N])
        synType :    突触类型   ("electr", "chem")
        method  :    计算非线性微分方程的方法，("euler", "rk4", "discrete")
    """
    def __init__(self, pre, post, conn=None, synType="electr", method="euler"):
        # 选择数值计算方法
        self.method = method
        method_options = ["euler", "rk4", "discrete"]
        if method not in method_options:
            raise ValueError(f"无效选择，method在{method_options}选择")
        if method == "euler":   self.method = Euler
        elif method == "rk4":   self.method = RK4
        elif method == "discrete":   self.method = discrete

        # 选择突触类型
        self.synType = synType
        if self.synType == "electr":
            self.syn = syn_electr  # 电突触
        elif self.synType == "chem":
            self.syn = syn_chem  # Alpha_化学突触

        self.pre = pre                  # 网络前节点
        self.post = post                # 网络后节点
        self.conn = conn                # 连接矩阵
        self.dt = post.dt               # 计算步长
        self._params_f()
        self._vars_f()

    def _params_f(self):
        # 0维度--post，1维度--pre
        self.w = .1 * np.ones((self.post.N, self.pre.N))  # 设定连接权重

    def _vars_f(self):
        self.t = self.post.t = 0.  # 运行时间

    def __call__(self):
        """
            开头和结尾更新时间(重要)
            self.t = self.post.t
            self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t  # 这个是非常重要的

        # 触前和突触后的状态
        pre_state = [self.pre.vars_nodes[0], self.pre.firingTime, self.pre.flaglaunch.astype(np.float64)]
        post_state = [self.post.vars_nodes[0], self.post.firingTime, self.post.flaglaunch.astype(np.float64)]

        I_post = self.syn(pre_state, post_state, self.w, self.conn)  # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post

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


# ================================= 演算法 =================================
@njit
def Euler(fun, x0, t, dt, *args):
    """
    使用 euler 方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        x0: 上一个时间单位的状态变量
        t: 运行时间
        dt: 时间步长
    :return: 
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    # 计算下一个时间单位的状态变量
    x0 += dt * fun(x0, t, *args)
    return x0

@njit
def RK4(fun, x0, t, dt, *args):
    """
    使用 Runge-Kutta 方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        x0: 上一个时间单位的状态变量
        t: 运行时间
        dt: 时间步长
    :return:
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    k1 = fun(x0, t, *args)
    k2 = fun(x0 + (dt / 2.) * k1, t + (dt / 2.), *args)
    k3 = fun(x0 + (dt / 2.) * k2, t + (dt / 2.), *args)
    k4 = fun(x0 + dt * k3, t + dt, *args)

    x0 += (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x0

@njit
def discrete(fun, x0, t, dt, *args):
    """
    使用离散方法计算一个时间步后系统的状态。
    args:
        fun: 微分方程
        x0: 上一个时间单位的状态变量
        t: 运行时间
        dt: 时间步长(设定为1)
    :return:
        x0 (numpy.ndarray): 下一个时间单位的状态变量
    """
    x0 = fun(x0, t, *args)
    return x0


# ================================= 常用工具 =================================
# ========= 延迟存储器 =========
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

# ========= 连接矩阵to拉普拉斯矩阵 =========
@njit
def to_laplacian(adjacency_matrix):
    """
        计算拉普拉斯矩阵
        adjacency_matrix: 邻接矩阵
    """
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    return laplacian_matrix

