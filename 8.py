from subprocess import Popen
import numpy as  np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
gal_num = 10000
for i in range(20):

    cmd = 'mpiexec -n 5 python Testing3.py  %d %d'%(i+140,gal_num)

    a = Popen(cmd, shell=True)

    a.wait()