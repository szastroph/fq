from subprocess import Popen
import numpy as  np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

ses_num = 6
for i in range(ses_num):
    cmd = 'tmux kill-session -t 0%d'%i

    a = Popen(cmd, shell=True)

    a.wait()