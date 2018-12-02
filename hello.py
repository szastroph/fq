

import numpy as np

from sys import path
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# path.append('/home/hklee/work/fourier_quad/')

from F_Q_P_S import F_Q_P_S
from Fourier_Quad import Fourier_Quad

from tool_box import *

obj = Fourier_Quad(48,101)
size = 52
psf_scale = 3
flux = 10
num_po,radius = 50,5
g1 ,g2 = 0.01,0.01
she_pos = obj.ran_pos(num_po, radius, (g1, g2))[1]
gal_image = obj.draw_noise(0,1)+obj.convolve_psf(she_pos,psf_scale,flux)
snr = obj.rw_snr(gal_image,1)
# print snr
gal_ps = obj.pow_spec(gal_image)

fit_p = fit_2o2(gal_ps)
fit_ps = fit_2o2(fit_p)


gal_log = np.log(gal_ps)
cur_log = smooth(gal_log,size-4)
cur_ps = np.exp(cur_log)

plt.subplot(222)
plt.imshow(fit_ps)
plt.colorbar()
plt.subplot(223)
plt.imshow(gal_ps)

plt.subplot(224)
plt.imshow((fit_p-cur_ps)/cur_ps)
plt.colorbar()
plt.subplot(221)
plt.imshow(gal_image)
plt.show()
