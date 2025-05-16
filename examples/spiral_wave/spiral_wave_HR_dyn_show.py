import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from import_fun import HR, Synapse, Diffusion2D, show_spiral_wave

Nx = 200
Ny = 200
method = "euler"
dt = 0.01

# 生成节点，初始值设定
nodes = HR(N=Nx*Ny, method=method, dt=dt)
nodes.vars_nodes[0, :] = -1.31742
nodes.vars_nodes[1, :] = -7.67799
nodes.vars_nodes[2, :] = 1.12032

# 生成一个方阵视图，这个视图与原始数组共享内存
x_view = nodes.vars_nodes[0, :].reshape(Nx, Ny)
y_view = nodes.vars_nodes[1, :].reshape(Nx, Ny)
z_view = nodes.vars_nodes[2, :].reshape(Nx, Ny)

# 生成一个方阵视图，这个视图与原始数组共享内存
x_view = nodes.vars_nodes[0, :].reshape(Nx, Ny)
y_view = nodes.vars_nodes[1, :].reshape(Nx, Ny)
z_view = nodes.vars_nodes[2, :].reshape(Nx, Ny)

# 设定楔形初始值
x_view[91:93, 0:100] = 2.
y_view[91:93, 0:100] = 2.
z_view[91:93, 0:100] = -1.

x_view[94:96, 0:100] = 0.
y_view[94:96, 0:100] = 0.
z_view[94:96, 0:100] = 0.

x_view[97:99, 0:100] = -1.
y_view[97:99, 0:100] = -1.
z_view[97:99, 0:100] = 2.

nodes.params_nodes["Iex"] = 1.315
nodes.params_nodes["s"] = 3.9

# 设定扩散耦合
syn = Diffusion2D(D=2., boundary="No_flow", adjacency=4)

# 设定动态显示器
spiral_wave = show_spiral_wave(nodes.vars_nodes[0], Nx, Ny, save_gif=True)

#设定突触和连接
# conn = create_diffusion_No_flow2D_4(Nx, Ny)
# conn = create_diffusion_No_flow2D_8(Nx, Ny)
# syn = Synapse(nodes, nodes, conn, method=method)
# syn.w.fill(2.)  # 设定耦合强度
# syn.to_sparse() # 换了耦合配置后重设稀疏化

for i in range(500_00):
    spiral_wave(i, nodes.t, show_interval=1000)
    Isyn = syn(x_view)
    nodes(Isyn)

spiral_wave.save_image()
spiral_wave.show_final()
