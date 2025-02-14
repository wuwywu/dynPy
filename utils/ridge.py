# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/2/14
# User      : WuY
# File      : ridge.py
# 岭回归算法--> ridge regression


import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import random

# np.random.seed(2024)
# random.seed(2024)


@njit
def Ridge(X, Y, alpha=1e-6):
    """
    实现多输出岭回归的闭式解。
    权重矩阵的形状为 (n_outputs, n_inputs)。
    岭回归的闭式解为：
        W = (X^T X + alpha I)^-1 X^T Y
        (Y = XW^T)
    其中，I 是单位矩阵，alpha 是正则化系数。

    args:
        X: 输入特征张量, 形状为 (n_samples, n_inputs)
        Y: 输出张量, 形状为 (n_samples, n_outputs)
        alpha: 正则化系数, 默认值为 1e-6
    return:
        权重张量 (n_outputs, n_inputs)
    """
    n_inputs = X.shape[1]
    I = np.eye(n_inputs)
    x_transpose = X.T

    w = np.linalg.pinv(x_transpose.dot(X) + alpha * I).dot(x_transpose).dot(Y)
    return w.T


if __name__ == "__main__":
    in_num = 95
    out_num = 2
    n_samples = 100

    input = np.random.randn(n_samples, in_num)  # 输入 (n_samples, n_inputs)
    output = np.sin(input[:, :out_num])          # 输出 (n_samples, n_outputs)
    w = Ridge(input, output)                    # 权重 (n_outputs, n_inputs)

    print(w.shape)
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))
    axs[0].plot(np.arange(n_samples), output.T[0], label="origin")
    axs[0].plot(np.arange(n_samples), (w@input.T)[0], label="fitting")
    axs[0].legend()

    axs[1].plot(np.arange(n_samples), output.T[1], label="origin")
    axs[1].plot(np.arange(n_samples), (w@input.T)[1], label="fitting")
    axs[1].legend()

    plt.show()
