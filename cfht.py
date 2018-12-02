import numpy as np
import os
import pandas as pd
from sys import path
path.append('/home/hklee/work/fourier_quad/')
from F_Q_P_S import F_Q_P_S
from Fourier_Quad import Fourier_Quad
from tool_box import *
from mpi4py import MPI
import lsstetc
import time

from sys  import argv
for i in range(3):
    path = '/home/shenzhi/CFHT/w1m0m%d_shear.cat'%i
