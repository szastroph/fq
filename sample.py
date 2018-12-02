# encoding: utf-8
from sys import path
path.append("/home/shenzhi/test")
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import optimize
from astropy.io import fits
from scipy.optimize import leastsq
from F_Q_P_S import F_Q_P_S
from Fourier_Quad import Fourier_Quad
import time

start = time.clock()

gal_num = 10000
imagesize = 64
psf_num = 7
pic_num = psf_num*gal_num
g1=0.01
g2=0.02
a =np.zeros((3,10))
for k in range(10):
    obj = F_Q_P_S()
    p=0
    G1 = np.zeros(pic_num)
    G2 = np.zeros(pic_num)
    N = np.zeros(pic_num)
    u = np.zeros(pic_num)
    p=0
    for i in range(gal_num):
        ran_pos = obj.ran_gal_walk(num=45 ,imagesize=imagesize,limit=8)
        for j in np.linspace(3,4.2,psf_num):
            psf_scale = j
            psf = obj.cre_psf(psf_scale=psf_scale, imagesize=imagesize,flux=4,model="GAUSS")
            # fwhm = obj.get_radius(image=psf,scale=2.)

            rot_pos = obj.rot_pos(ran_pos, 0)
            shear_pos = obj.add_shear(rot_pos, g1 = g1, g2 = g2)
            psf_image = obj.convol_psf(shear_pos, psf_scale=psf_scale , imagesize=imagesize,flux=4,psf="GAUSS")
            # if i ==1 and j ==3:
            #     plt.imshow(psf_image)
            #     plt.show()
            G1[p], G2[p], N[p],u[p]= obj.shear_est( psf_image, psf, imagesize=imagesize,background_noise = None)
            p += 1
        # print(type(g1))
    a[0,k],a[1,k] = obj.fmin_g(G1,N,u,1,4)
    a[2,k]=abs(a[0,k]-g1)/a[1,k]
    print(a[0,k],a[1,k],abs(a[0,k]-g1)/a[1,k])
    # print(g1,g2)
# np.savez()
np.savetxt("180331g_hg_sig.txt",a)

end = time.clock()
print "Whole time =",str(end)
