import os
import sys
sys.path.append(os.path.dirname(__file__))  # 将文件所在地址放入系统调用地址中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from base_mods import *
from nodes.HH import *
from nodes.HR import *
from nodes.FHN import *
from connect.spiral_wave_conn import *
from couples.diffusion2D import *
from analys.spiral.flow_velocity import *
