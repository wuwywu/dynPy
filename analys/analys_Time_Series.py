# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2025/12/5
# User      : WuY
# File      : analys_Time_Series.py
# 分析时间序列的相关函数

import numpy as np

import numpy as np

# ================= 寻找穿过点 =================
# 可以用来找 MSF 的阈值
def find_crossings(y, v=0, x=None):
    """
    找到序列 y 穿过阈值 v 的位置（线性插值）

    参数:
        y : 1D 数组或列表，数据列
        v : float，阈值
        x : 1D 数组或列表，可选，对应的横坐标；
            如果为 None，则使用 x = 0,1,2,...,len(y)-1

    返回:
        x_cross : 1D numpy 数组，所有穿过点的 x 坐标
        y_cross : 1D numpy 数组，对应的 y 坐标（恒等于 v）
        idx     : 1D numpy 数组，每个穿过点所在的区间左端索引 i，
                  表示该点位于 [x[i], x[i+1]] 之间
    """
    # 转为 numpy 数组
    y = np.asarray(y, dtype=float)

    if x is None:
        x = np.arange(len(y), dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        if x.shape != y.shape:
            raise ValueError("x 和 y 的长度必须相同")

    if len(y) < 2:
        return np.array([]), np.array([]), np.array([], dtype=int)

    # 与阈值的差
    diff = y - v

    # 穿过条件：相邻两点在阈值两侧 → diff[i] * diff[i+1] < 0
    mask = diff[:-1] * diff[1:] < 0
    idx = np.where(mask)[0]  # 每个穿过点在区间 [i, i+1] 内

    if len(idx) == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)

    # 线性插值：
    # x_cross = x_i + (x_{i+1} - x_i) * (v - y_i) / (y_{i+1} - y_i)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]

    x_cross = x0 + (x1 - x0) * (v - y0) / (y1 - y0)
    y_cross = np.full_like(x_cross, fill_value=v, dtype=float)

    return x_cross, y_cross, idx


# ================= 使用示例 =================
if __name__ == "__main__":
    # 示例数据
    y = [2, 4, 1, 5, 0]
    v = 3

    # 例1：没有给 x，默认 x = 0,1,2,...
    x_cross, y_cross, idx = find_crossings(y, v)

    print("穿过点所在区间左索引 idx =", idx)
    print("穿过点 x 坐标 =", x_cross)
    print("穿过点 y 坐标 =", y_cross)

    # 例2：给定不等距 x
    x = [0.0, 0.5, 2.0, 3.0, 4.5]
    x_cross2, y_cross2, idx2 = find_crossings(y, v, x=x)

    print("（不等距 x）穿过点 x 坐标 =", x_cross2)
