from subprocess import Popen
import numpy as  np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

gal_num = 100

cmd = 'mpiexec -n 10 python Testing.py %d '%gal_num

a = Popen(cmd, shell=True)

a.wait()

