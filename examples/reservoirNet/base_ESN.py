# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/2/14
# User      : WuY
# File      : ESN.py
# 回声状态网络（ESN）模型

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random

# np.random.seed(2024)
# random.seed(2024)

class ESN_base:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, alpha=0.3):
        """
        args:
            input_size:         输入数量
            reservoir_size:     储池中的节点数
            output_size:        输出数量
            spectral_radius:    谱半径
            alpha:              衰减率
        """
        self.input_size = input_size            # 输入数量
        self.reservoir_size = reservoir_size    # 储池中的节点数
        self.output_size = input_size           # 输出数量
        self.spectral_radius = spectral_radius  # 谱半径（特征值的最大值，大于1就一定不具有回声状态）
        self.alpha = alpha                      # 储池状态的衰减率

        self._init_states()                     # 初始化储池状态
        self._init_weights()                    # 初始化权重矩阵

    def _init_states(self):
        """
            初始化储池状态, 输入权重矩阵, 输入偏置, 输出权重矩阵, 输出偏置
        """
        # 初始化储池状态
        self.r = np.zeros(self.reservoir_size)

        # 初始化储池权重矩阵
        self.w_res2res = np.random.randn(self.reservoir_size, self.reservoir_size)

        # 初始化输入权重矩阵
        self.w_in2res = np.random.randn(self.reservoir_size, self.input_size)

        # 初始化输出权重矩阵（目的: 调整，拟合）
        self.w_res2out = np.random.randn(self.output_size, self.reservoir_size)

    def _init_weights(self):
        """
            初始化权重矩阵
        """ 
        # 初始化输入权重
        self.w_in2res = np.random.uniform(-.1, .1, size=self.w_in2res.shape)
        self.w_in2res *= (np.random.rand(*self.w_in2res.shape) < 0.05)               # 稀疏化

        # 初始化回声状态层权重
        self.w_res2res = np.random.normal(0, .1, size=self.w_res2res.shape)
        self.w_res2res *= (np.random.rand(*self.w_res2res.shape) < 0.05)             # 稀疏化

        # 调整谱半径
        rho_w = max(abs(np.linalg.eigvals(self.w_res2res)))               # 计算矩阵的最大特征值
        self.w_res2res *= self.spectral_radius / rho_w
        
    def __call__(self, inputs):
        """
        input-->reservoir(->reservoir)-->output
        args:
            inputs: 输入数据(input_size, )
        return:
            outputs: 输出数据(output_size, )
        """
        self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(self.w_in2res @ inputs + self.w_res2res @ self.r)
        outputs = self.w_res2out @ self.r

        return outputs
    
    def train_readout(self, inputs):
        """
        输出训练用的储池状态
        args:
            inputs: 输入数据(input_size, )
        """
        self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(self.w_in2res @ inputs + self.w_res2res @ self.r)
        return self.r
    