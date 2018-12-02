import numpy as np

from sys import path

path.append('/home/hklee/work/fourier_quad/')

from F_Q_P_S import F_Q_P_S

import tool_box

from mpi4py import MPI

import time

size =48

ran_poi = 45

gal_num = 10000

obj =  F_Q_P_S(size,123)

pos = np.zeros((gal_num,2,ran_poi))


for i in range(gal_num):
    pos[i]= obj.ran_gal_walk(ran_poi,6)

path ="/home/shenzhi/data/ran_pos"
np.savez(path,pos)