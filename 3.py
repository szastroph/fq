from subprocess import Popen
import numpy as  np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

gal_num = 100000
ind = 3
cmd = 'mpiexec -n 10 python Testing.py %d %d'%(gal_num,ind)

a = Popen(cmd, shell=True)

a.wait()