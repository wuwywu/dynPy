# encoding: utf-8
# Author    : WuY<wuyong@mails.ccnu.edu.com>
# Datetime  : 2024/11/13
# User      : WuY
# File      : utils_f.py
# 一些工具

import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

# ================================= 记录神经元发放峰的时间 =================================
class spikevent:
    """
        神经元模型的峰值收集器
        N: 收集 N 个神经的尖峰事件
    """
    def __init__(self, N):
        self.N = N
        self.Tspike_list = [[] for i in range(N)]

    def __call__(self, t, flaglaunch):
        """
        t: 模拟实时时间
        flaglaunch: 是否尖峰的标志 (放电开启标志)
        """
        # 找出符合条件的神经元索引
        indices = np.where(flaglaunch > 0.9)[0]
        # 使用列表推导式直接将时间添加到符合条件的神经元中
        [self.Tspike_list[i].append(t) for i in indices]

    def pltspikes(self):
        plt.eventplot(self.Tspike_list)


# ================================= 状态变量延迟 =================================
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


# ================================= 连接矩阵to拉普拉斯矩阵 =================================
@njit
def to_laplacian(adjacency_matrix):
    """
        计算拉普拉斯矩阵
        adjacency_matrix: 邻接矩阵
    """
    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix
    return laplacian_matrix


# ================================= 计算特征值(Numba版) =================================
@njit
def eigvals_qr(A, num_iterations=1000):
    n = A.shape[0]
    A_k = np.ascontiguousarray(A.copy())  # 复制原始矩阵
    
    for _ in range(num_iterations):
        # QR分解
        Q, R = np.linalg.qr(A_k)
        A_k = np.dot(R, Q)  # 迭代更新矩阵
    
    # 对角元素即为特征值
    return np.diag(A_k)


# ================================= 噪声 =================================
class noiser:
    """
        噪声产生
        D_noise:    噪声强度
        dt:         计算步步长
        N:          要产生噪声的尺度，可以是整数或元组
        type:       噪声类型，可选：["white"、"color"]
        lam_color:  色噪声的相关率
    """
    def __init__(self, D_noise, dt, N, type="white", lam_color=0.1):
        self.D_noise = D_noise
        self.dt = dt
        self.N = N
        self.lam_color = lam_color

        self.type = type
        type_map = {"white": Gaussian_white_noise, "color": Gaussian_colour_noise}
        if type not in type_map:
            raise ValueError(f"无效选择，type 必须是 {list(type_map.keys())}")
        self.noisee = type_map[type]

        # 初始化噪声
        self.noise = dt*np.random.normal(loc=0., scale=np.sqrt(D_noise*lam_color), size=N)

    def __call__(self):
        """
            产生噪声
        """
        if self.type == "white":
            self.noise = self.noisee(self.D_noise, self.noise, self.dt, self.N)
        elif self.type == "color":
            self.noise = self.noisee(self.D_noise, self.noise, self.dt, self.N, self.lam_color)

@njit
def Gaussian_white_noise(D_noise, noise, dt, size):
    """
        高斯白噪声
        D_noise:    噪声强度
        dt:         计算步步长
        size:       要产生噪声的尺度，可以是整数或元组
    """
    # a = np.random.rand(size)
    # b = np.random.rand(size)
    # noise = np.sqrt(-4*D_noise*dt*np.log(a)) * np.cos(2*np.pi*b)
    noise = np.random.normal(loc=0., scale=np.sqrt(2*D_noise*dt), size=size)
    return noise

@njit
def Gaussian_colour_noise(D_noise, noise, dt, size, lam_color):
    """
        高斯色噪声
        D_noise:    噪声强度
        noise:      上一步的噪声
        dt:         计算步步长
        size:       要产生噪声的尺度，可以是整数或元组
        lam_color:  色噪声的相关率
    """
    # a = np.random.rand(size)
    # b = np.random.rand(size)
    # g_w = np.sqrt(-4*D_noise*dt*np.log(a)) * np.cos(2*np.pi*b)
    g_w = np.random.normal(loc=0., scale=np.sqrt(2*D_noise*dt), size=size) 
    noise = noise - dt * lam_color * noise + lam_color*g_w
    noise = dt * noise

    return noise


# ================================= 稀疏函数工具模块 =================================
@njit
def matrix_to_sparse(conn, weight_matrix=None):
    """
    将矩阵转换为稀疏矩阵
  
        参数：
        - conn: 连接矩阵(二元矩阵)
        - weight_matrix: 权重矩阵(可选)
    
        返回：
        - pre_ids   : 前节点的id
        - post_ids  : 后节点的id
        - weights   : 节点对的权重
    """
    # 将 conn 转换为二元矩阵
    binary_conn = np.where(conn != 0, 1, 0)
  
    # 如果未提供权重矩阵，则默认为全1矩阵
    if weight_matrix is None:
        weight_matrix = np.ones_like(conn, dtype=np.float64)
    else:
        weight_matrix = np.asarray(weight_matrix, dtype=np.float64)

    # 确保 binary_conn 和 weight_matrix 形状一致
    if binary_conn.shape != weight_matrix.shape:
        raise ValueError("binary_conn 和 weight_matrix 的形状必须一致！")
  
    # 提取非零元素的行列索引
    post_ids, pre_ids = np.nonzero(binary_conn)

    # 提取对应权重
    rows, cols = weight_matrix.shape
    indices =  post_ids * rows + pre_ids  # 计算一维索引
    weights = weight_matrix.ravel()[indices]  # 一维索引提取权重

    # 将结果整合为一个三列矩阵
    # ids_and_weights = np.vstack((pre_ids, post_ids, weights))

    return pre_ids, post_ids, weights

@njit
def sparse_to_matrix(N_pre, N_post, pre_ids, post_ids, weights):
    """
    将稀疏矩阵信息转换为连接矩阵和权重矩阵
    
        参数：
        - N: int, 神经元总数
        - pre_ids: np.ndarray, 前节点id
        - post_ids: np.ndarray, 后节点id
        - weights: np.ndarray, 权重
    
        返回：
        - conn: np.ndarray, 二元连接矩阵 
        - weight_matrix: np.ndarray, 权重矩阵 
    """
    # 将输入转换为整数类型
    pre_ids, post_ids = pre_ids.astype(np.int32), post_ids.astype(np.int32)
    # 初始化连接矩阵和权重矩阵
    conn = np.zeros((N_post, N_pre), dtype=np.int32)
    weight_matrix = np.zeros((N_post, N_pre), dtype=np.float64)
  
    # 扁平化索引
    flat_indices = post_ids * N_post + pre_ids

    # 将对应位置设置为连接
    conn_flat = conn.ravel()
    conn_flat[flat_indices] = 1

    # 将权重赋值
    weight_flat = weight_matrix.ravel()
    weight_flat[flat_indices] = weights

    return conn, weight_matrix


# ========= 螺旋波动态显示器 =========
class show_spiral_wave:
    """
    螺旋波动态显示器
        var: 要显示的变量
        Nx:  网络的一维长度
        Ny:  网络的二维长度
        save_gif: 是否保存gif动图
    """
    def __init__(self, var, Nx, Ny, save_gif=False):
        self.var = var
        self.Nx = Nx
        self.Ny = Ny
        self.save_gif = save_gif
        self.frames = []
        plt.ion()

    def __call__(self, i, t=None, show_interval=1000):
        """
            i: 迭代次数
            t: 时间
            show_interval: 显示间隔
        """
        if i % show_interval <= 0.001:
            var = self.var.reshape(self.Nx, self.Ny)
            plt.clf()
            plt.imshow(var, cmap="jet", origin="lower", aspect="auto")
            if t is not None:
                plt.title(f"t={t:.3f}")
            else:
                plt.title(f"i={i}")
            plt.colorbar()
            plt.pause(0.0000000000000000001)

            if self.save_gif:
                buffer_ = io.BytesIO()
                plt.savefig(buffer_, format='png')
                buffer_.seek(0)
                self.frames.append(Image.open(buffer_))

    def save_image(self, filename="animation.gif", duration=50):
        """
            保存gif动图
            filename: 文件名
            duration: 每帧持续时间(ms)
        """
        if self.save_gif:
            self.frames[0].save(filename, save_all=True, append_images=self.frames[1:], duration=duration, loop=0)

    def show_final(self):
        """ 
            显示最终图像
        """
        plt.ioff()
        var = self.var.reshape(self.Nx, self.Ny)
        plt.imshow(var, cmap="jet", origin="lower", aspect="auto")
        plt.colorbar()
        plt.show()  # 停止交互模式，显示最终图像


# ========= 状态变量(膜电位)动态显示器 =========
class show_state:
    """
    状态变量动态显示器
        N_var:      要显示的变量数
        N_time:     要显示时间宽度
        dt:         计算步步长
        save_gif:   是否保存gif动图
    """ 
    def __init__(self, N_var, N_show=50_00, dt=0.01, save_gif=False):
        self.N_var = N_var
        self.N_time = N_show
        self.vars = np.full((N_show, N_var), np.nan)
        self.dt = dt
        self.count = 0
        self.save_gif = save_gif
        self.frames = []
        plt.ion()

    def __call__(self, var, i, t=None, show_interval=1000, pause=0.0000000000000000001):
        """
            var : 状态变量
            i   : 迭代次数
            t   : 时间
            show_interval: 显示间隔(更新频率)
            pause: 暂停时间
        """
        if len(var) != self.N_var:
            raise ValueError("var的长度与N_var不一致")
        
        if self.count < self.N_time:
            self.vars[self.count] = var
        else:
            self.vars[:-1] = self.vars[1:]
            self.vars[-1] = var

        self.count += 1
        
        if i % show_interval <= 0.001:
            plt.clf()   # 清除当前图像
            if t is not None:
                if t < self.N_time*self.dt:
                    self.time_vec = np.linspace(0, self.N_time*self.dt, self.N_time)
                else:
                    self.time_vec = np.linspace(t-self.N_time*self.dt, t, self.N_time)
            else:        
                if i < self.N_time:
                    self.time_vec = np.linspace(0, self.N_time*self.dt, self.N_time)  
                else:
                    self.time_vec = np.linspace((i-self.N_time)*self.dt, i*self.dt, self.N_time)
            self.vars_temp = self.vars.copy()
            plt.plot(self.time_vec, self.vars_temp)
            plt.xlim(self.time_vec[0], self.time_vec[-1])
            plt.pause(pause)

            if self.save_gif:
                buffer_ = io.BytesIO()
                plt.savefig(buffer_, format='png')
                buffer_.seek(0)
                self.frames.append(Image.open(buffer_))

    def save_image(self, filename="animation.gif", duration=50):
        """
            保存gif动图
            filename: 文件名
            duration: 每帧持续时间(ms)
        """
        if self.save_gif:
            self.frames[0].save(filename, save_all=True, append_images=self.frames[1:], duration=duration, loop=0)

    def show_final(self):
        """
            显示最终图像
        """
        plt.ioff()
        plt.plot(self.time_vec, self.vars_temp)
        plt.xlim(self.time_vec[0], self.time_vec[-1])
        plt.show()  # 停止交互模式，显示最终图像


if __name__ == "__main__":
    N = 1
    noisee = noiser(D_noise=0.01, dt=0.01, N=N, type="white")
    # noisee = noiser(D_noise=0.01, dt=0.01, N=N, type="color", lam_color=.1)

    noise_ = []
    for i in range(1000):
        noisee()
        noise_.append(noisee.noise.copy())

    noise_ = np.array(noise_)

    plt.plot(noise_)
    plt.show()
