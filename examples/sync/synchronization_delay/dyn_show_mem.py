import matplotlib.pyplot as plt
import numpy as np
from import_fun import HH, syn_sigmoidal, syn_sigmoidal_delay, create_sf, delayer, synFactor, show_state

dt = 0.01
method = 'euler'    # （"euler", "rk4"）
N = 100
tau = 11.5

conn = create_sf(N)
delayN = int(tau/dt)
# delayee = delayer(N, delayN)

nodes = HH(N, method, dt)
nodes.params_nodes["Iex"] = 20.
# syn = syn_sigmoidal(nodes, nodes, conn, method=method)
syn = syn_sigmoidal_delay(nodes, nodes, conn, method=method, delayN=delayN)
syn.w.fill(0.1)

# 动态显示器
N_state = 3
N_show = 50_00
state = show_state(N_state, N_show, dt, save_gif=True)

for i in range(100_00):
    nodes()

nodes.t = 0
for i in range(500_00):
    state(nodes.vars_nodes[0, [1, 50, 99]], i, nodes.t, show_interval=100, pause=.001)
    Isyn = syn()
    nodes(Isyn)

state.save_image()

state.show_final()
