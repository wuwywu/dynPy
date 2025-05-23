# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/15
# User      : WuY
# File      : synapse_sigmoidal.py
# ref: D. Somers and N. Kopell, Rapid synchronization through fast threshold modulation. Biol. Cybernet. 68, 393 (1993).

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(r"../")
import copy
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random
from base_mods import Synapse
from base_mods import delayer

# np.random.seed(2024)
# random.seed(2024)

@njit
def syn_chem_sigmoidal_sparse(pre_state, post_state, pre_ids, post_ids, weights, params):
    """
    sigmoidal 化学突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        pre_ids: 突触前神经元的索引
        post_ids: 突触后神经元的索引
        weights: 突触权重
        params: 突触参数
    """
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state

    e, theta, epsi = params

    # the sigmoidal function (a limiting version is the Heaviside function)
    s = 1 / (1 + np.exp(-epsi*(pre_mem - theta)))
    
    # 计算突触电流贡献
    currents = weights * s[pre_ids] * (e - post_mem[post_ids])

    # 神经元数量
    num_neurons = len(post_mem)  # 突触后神经元总数

    # 累积电流贡献到突触后神经元
    Isyn = np.bincount(post_ids, weights=currents, minlength=num_neurons)

    return Isyn


class syn_sigmoidal(Synapse):
    """
        pre     :    网络前节点
        post    :    网络后节点
        conn    :    连接矩阵   (size  : [post_N, pre_N])
        synType :    突触类型   ("electr", "chem_sigmoidal")
        method  :    计算非线性微分方程的方法，("euler", "heun", "rk4", "discrete")
    """
    def __init__(self, pre, post, conn=None, synType="chem_sigmoidal", method="euler"):
        super().__init__(pre=pre, post=post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.synType = synType
        if self.synType == "chem_sigmoidal":
            self.syn = syn_chem_sigmoidal_sparse  # 化学突触

        self._params()
        self._vars()

    def _params(self):
        self.params_syn = {
            "e": 0.,     # 化学突触的平衡电位(mV)
            "theta": 0., # 放电阈值
            "epsi": 7.,  # 放电下滑斜率
        }

    def _vars(self):
        # 0维度--post，1维度--pre
        pass

    def __call__(self):
        """
            开头和结尾更新时间（重要）
            self.t = self.post.t
            self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t  # 这个是非常重要的

        # 触前和突触后的状态
        pre_state = (self.pre.vars_nodes[0], self.pre.firingTime, self.pre.flaglaunch)
        post_state = (self.post.vars_nodes[0], self.post.firingTime, self.post.flaglaunch)
        params_list = list(self.params_syn.values())

        I_post = self.syn(pre_state, post_state, self.pre_ids, self.post_ids, self.w_sparse, params_list)  # 突触后神经元接收的突触电流, params_list

        self.t += self.dt  # 时间前进

        return I_post


class syn_sigmoidal_delay(Synapse):
    """
        pre     :    网络前节点
        post    :    网络后节点
        conn    :    连接矩阵   (size  : [post_N, pre_N])
        synType :    突触类型   ("electr", "chem_sigmoidal")
        method  :    计算非线性微分方程的方法，("euler", "heun", "rk4", "discrete")
        delayer :    突触延迟器
    """
    def __init__(self, pre, post, conn=None, synType="chem_sigmoidal", method="euler", delayN=0):
        super().__init__(pre=pre, post=post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.synType = synType
        if self.synType == "chem_sigmoidal":
            self.syn = syn_chem_sigmoidal_sparse  # 化学突触

        self.delayee = delayer(pre.N, int(delayN))

        self._params()
        self._vars()

    def _params(self):
        self.params_syn = {
            "e": 0.,     # 化学突触的平衡电位(mV)
            "theta": 0., # 放电阈值
            "epsi": 7.,  # 放电下滑斜率
        }

    def _vars(self):
        # 0维度--post，1维度--pre
        pass

    def __call__(self):
        """
            开头和结尾更新时间（重要）
            self.t = self.post.t
            self.t += self.dt
        """
        # 保证syn不管何时创建都能与突触后有相同的时间
        self.t = self.post.t  # 这个是非常重要的

        # 触前和突触后的状态
        pre_mem = self.delayee(self.pre.vars_nodes[0])              # 存储延迟，并给出延迟的值
        pre_state = (pre_mem, self.pre.firingTime, self.pre.flaglaunch)
        post_state = (self.post.vars_nodes[0], self.post.firingTime, self.post.flaglaunch)

        # 将突触参数转化为列表
        params_list = list(self.params_syn.values())

        I_post = self.syn(pre_state, post_state, self.pre_ids, self.post_ids, self.w_sparse, params_list)  # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post


# =============================== 矩阵计算神经网络(不建议使用) ===============================
@njit
def syn_chem_sigmoidal(pre_state, post_state, w, conn, params):
    """
    sigmoidal 化学突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        w: 突触权重
        conn: 连接矩阵
        params: 突触参数
    """
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state
    e, theta, epsi = params

    # the sigmoidal function (a limiting version is the Heaviside function)
    s = 1 / (1 + np.exp(-epsi*(pre_mem - theta)))
    
    Isyn = (w * conn * (e - post_mem)[:, None] * s[None, :]).sum(1) # 0维度--post，1维度--pre

    return Isyn

