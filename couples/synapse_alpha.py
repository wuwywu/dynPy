# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/12/13
# User      : WuY
# File      : synapse_alpha.py
# ref: W. Rall, Distinguishing theoretical synaptic potentials computed for different soma-dendritic distributions of synaptic inputs, J. Neurophysiol. 30(5), 1138-1168 (1967).

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
def syn_chem_alpha_sparse(pre_state, post_state, pre_ids, post_ids, weights, t, params):
    """
    alpha 化学突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        pre_ids: 突触前神经元的索引
        post_ids: 突触后神经元的索引
        weights: 突触权重
        t: 时间
        params: 突触参数
    """
    # 前节点的状态
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state

    tau_syn, e = params

    alpha = (t - pre_firingTime) / tau_syn * np.exp((pre_firingTime - t) / tau_syn)
    g_syn = alpha[pre_ids] * (e - post_mem[post_ids])

    # 计算突触电流贡献
    currents = weights * g_syn

    # 神经元数量
    num_neurons = len(post_mem)  # 突触后神经元总数

    # 累积电流贡献到突触后神经元
    Isyn = np.bincount(post_ids, weights=currents, minlength=num_neurons)

    return Isyn

@njit
def syn_chem_alpha_delay_sparse(pre_state, post_state, pre_ids, post_ids, weights, t, params, alpha_delay):
    """
    alpha 化学突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        w: 突触权重
        conn: 连接矩阵
        t: 时间
        params: 突触参数
        alpha_delay: 延迟的alpha函数
    """
    # 前节点的状态
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state

    tau_syn, e = params

    g_syn = alpha_delay[pre_ids] * (e - post_mem[post_ids])

    # 计算突触电流贡献
    currents = weights * g_syn

    # 神经元数量
    num_neurons = len(post_mem)  # 突触后神经元总数

    # 累积电流贡献到突触后神经元
    Isyn = np.bincount(post_ids, weights=currents, minlength=num_neurons)

    return Isyn


class syn_alpha(Synapse):
    """
        pre     :    网络前节点
        post    :    网络后节点
        conn    :    连接矩阵   (size  : [post_N, pre_N])
        synType :    突触类型   ("electr", "chem_alpha")
        method  :    计算非线性微分方程的方法，("euler", "heun", "rk4", "discrete")
    """
    def __init__(self, pre, post, conn=None, synType="chem_alpha", method="euler"):
        super().__init__(pre=pre, post=post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.synType = synType
        if self.synType == "chem_alpha":
            self.syn = syn_chem_alpha_sparse  # 化学突触

        self._params()
        self._vars()

    def _params(self):
        self.params_syn = {
            "tau_syn":2.,   # 化学突触的时间常数
            "e": 0.,        # 化学突触的平衡电位(mV)
        }

    def _vars(self):
        # 0维度--post，1维度--pre
        pass

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
        params_list = list(self.params_syn.values())

        I_post = self.syn(pre_state, post_state, self.pre_ids, self.post_ids, self.w_sparse, self.t, params_list)  # 突触后神经元接收的突触电流

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
        params_list = list(self.params_syn.values())

        I_post = self.syn(pre_state, post_state, self.w, self.conn, self.t, params_list)  # 突触后神经元接收的突触电流

        self.t += self.dt  # 时间前进

        return I_post
    

class syn_alpha_delay(Synapse):
    """
        pre     :    网络前节点
        post    :    网络后节点
        conn    :    连接矩阵   (size  : [post_N, pre_N])
        synType :    突触类型   ("electr", "chem_alpha")
        method  :    计算非线性微分方程的方法，("euler", "heun", "rk4", "discrete")
        delayN  :    突触延迟的数量
    """
    def __init__(self, pre, post, conn=None, synType="chem_alpha", method="euler", delayN=0):
        super().__init__(pre=pre, post=post, conn=conn, synType=synType, method=method)
        # 选择突触类型
        self.synType = synType
        if self.synType == "chem_alpha":
            self.syn = syn_chem_alpha_delay_sparse  # 化学突触

        self.delayee = delayer(pre.N, int(delayN))

        self._params()
        self._vars()

    def _params(self):
        self.params_syn = {
            "tau_syn":2.,   # 化学突触的时间常数
            "e": 0.,        # 化学突触的平衡电位(mV)
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
        if self.synType == "electr":
            pre_mem = self.delayer(self.pre.vars_nodes[0])  
        elif self.synType == "chem_alpha":
            pre_mem = self.pre.vars_nodes[0]
            pre_firingTime = self.pre.firingTime
            tau_syn = self.params_syn["tau_syn"]
            alpha = (self.t - pre_firingTime) / tau_syn * np.exp((pre_firingTime - self.t) / tau_syn)
            alpha = self.delayee(alpha)    # 存储延迟，并给出延迟的值
        else:
            pre_mem = self.pre.vars_nodes[0]

        pre_state = (pre_mem, self.pre.firingTime, self.pre.flaglaunch)
        post_state = (self.post.vars_nodes[0], self.post.firingTime, self.post.flaglaunch)
        params_list = list(self.params_syn.values())

        I_post = self.syn(pre_state, post_state, self.pre_ids, self.post_ids, self.w_sparse, self.t, params_list, alpha)  # 突触后神经元接收的突触电流, params_list

        self.t += self.dt  # 时间前进

        return I_post
    

# =============================== 矩阵计算神经网络(不建议使用) ===============================
@njit
def syn_chem_alpha(pre_state, post_state, w, conn, t, params):
    """
    alpha 化学突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        w: 突触权重
        conn: 连接矩阵
        t: 时间
        params: 突触参数
    """
    # 前节点的状态
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state
    tau_syn, e = params

    alpha = (t - pre_firingTime) / tau_syn * np.exp((pre_firingTime - t) / tau_syn)
    g_syn = alpha * np.expand_dims((e - post_mem), axis=1)

    Isyn = (w * conn * g_syn).sum(axis=1)  # 0维度--post，1维度--pre

    return Isyn

@njit
def syn_chem_alpha_delay(pre_state, post_state, w, conn, t, params, alpha_delay):
    """
    alpha 化学突触
        pre_state: 突触前的状态
        post_state: 突触后的状态
        w: 突触权重
        conn: 连接矩阵
        t: 时间
        params: 突触参数
        alpha_delay: 延迟的alpha函数
    """
    # 前节点的状态
    pre_mem, pre_firingTime, pre_flaglaunch = pre_state
    post_mem, post_firingTime, post_flaglaunch = post_state
    tau_syn, e = params

    g_syn = alpha_delay * np.expand_dims((e - post_mem), axis=1)

    Isyn = (w * conn * g_syn).sum(axis=1)  # 0维度--post，1维度--pre

    return Isyn

