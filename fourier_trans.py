# encoding: utf-8
import numpy as np
from numpy import fft
from scipy.optimize import fmin_cg
from scipy import ndimage, signal
import copy
import matplotlib.pyplot as plt
import random
from scipy import optimize
from astropy.io import fits
from Fourier_Quad import Fourier_Quad
class Fourier_trans:

    def random_gallexy(self,radius,num):
        r   = np.random.uniform(low=0,high= radius, size=num)
        theta = np.random.uniform(low=0,high=2*np.pi,size=num)
        pos = np.matrix([r*np.cos(theta), r*np.sin(theta)])
        return pos#2*50

    def rot_pos(self,theta,pos):
        rot_pos = np.matrix([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])*pos
        return rot_pos

    def add_shear(self,g1,g2,pos):
        shear = np.matrix(([1+g1/(1-g1**2-g2**2),g2/(1-g1**2-g2**2)],[g2/(1-g1**2-g2**2),1-g1/(1-g1**2-g2**2)]))
        shear_pos = shear * pos#.reshape(2,50)
        return shear_pos

    def gaussian_psf(self,shear_pos,imagesize,sigma):
        x = shear_pos.shape[1]
        mx,my= np.mgrid[0:imagesize, 0:imagesize]
        pos = np.array(shear_pos) + imagesize/2.      ##将起始坐标（原点）设在图片中心
        arr = np.zeros((imagesize, imagesize))
        for i in range(x):
            z = np.exp(-((mx-pos[0, i])**2+(my-pos[1, i])**2)/sigma**2/2./np.pi)
            arr += z
    #       d = fits.PrimaryHDU(np.array(z))
    #       d.writeto('gauss_psf.fits',overwrite=True)
    #    plt.figure()
     #   plt.imshow(arr)
      #  plt.colorbar()
       # plt.show()
        return arr

    def poission_n(self,gal_imag,imagesize):
        poi_noi = np.random.poisson(50,imagesize**2).reshape(imagesize,imagesize)
        arr = gal_imag+poi_noi
        return arr

#image_ps = fft.fftshift((np.abs(fft.fft2(arr)))**2)#使用Matplotlib绘制变换后的信号
#使用fft函数对余弦波信号进行傅里叶变换。
#对变换后的结果应用ifft函数，应该可以近似地还原初始信号。
    def pow_spec(self, image):
        image_ps = fft.fftshift((np.abs(fft.fft2(image)))**2)
        return image_ps
    def shear_est(self, imag, psf_image, imagesize, background_noise=None, F=False, N=False):
        x = imagesize
        my, mx = np.mgrid[0:x, 0:x]
        image_ps = self.pow_spec(imag)
        if background_noise is not None: # to deduct the noise
            nbg = self.pow_spec(background_noise)
            if N == False:
                rim = self.border(1, x)
                n   = np.sum(rim)
                gal_pnoise = np.sum(image_ps*rim)/n               #the Possion noise of galaxy image
                nbg_pnoise = np.sum(nbg*rim)/n                   #the  Possion noise of background noise image
                image_ps = image_ps - nbg + nbg_pnoise - gal_pnoise
            else:
                image_ps = image_ps - nbg

        if F==True:
            psf_ps = psf_image
        else:
            psf_ps = self.pow_spec(psf_image)

        hlr = self.get_radius_new(psf_ps, 2., x)[0]
        wb, beta = self.wbeta(hlr, x)
        maxi = np.max(wb)
        idx = wb < maxi / 100000.
        wb[idx] = 0.
        maxi = np.max(psf_ps)
        idx = psf_ps < maxi / 100000.
        psf_ps[idx] = 1.

        tk = wb/psf_ps * image_ps
        alpha = 2.*np.pi/x
        kx = (mx-0.5*x)
        ky = (my-0.5*x)
        mn1 = (-0.5)*(kx**2 - ky**2)
        mn2 = -kx*ky
        mn3 = kx**2 + ky**2 - 0.5*beta**2*(kx**2 + ky**2)**2
        mn4 = kx**4 - 6*kx**2*ky**2 + ky**4
        mn5 = kx**3*ky - kx*ky**3

        mg1 = np.sum((mn1 * tk)*(alpha**4))
        mg2 = np.sum((mn2 * tk)*(alpha**4))
        mn  = np.sum((mn3 * tk)*(alpha**4))
        u  = np.sum(mn4 * tk)*(-0.5*beta**2)*(alpha**4)
        v  = np.sum(mn5 * tk)*(-2.*beta**2)*(alpha**4)
        return mg1, mg2,mn,wb

    def cre_psf(self, psf_scale, imagesize, model="GAUSS", x=0, y=0):
        xx = np.linspace(0, imagesize - 1, imagesize)
        mx, my = np.meshgrid(xx, xx)
        arr = np.exp(-((mx -imagesize/2.+x)**2+(my-imagesize/2.+y)**2)/2./np.pi/psf_scale**2)

        return arr


    def wbeta(self, beta, imagesize):
        my, mx = np.mgrid[0:imagesize, 0:imagesize]
        sigma = beta/np.sqrt(2)
        w_temp = np.exp(-((mx-0.5*imagesize)**2+(my-0.5*imagesize)**2)/2./sigma**2)
        beta = 1./beta
        return w_temp, beta

    def get_radius(self, image, scale):
        # get the radius of the flux descends to the maximum/scale
        radi_arr = copy.copy(image)
        maxi = np.max(radi_arr)
        y, x = np.where(radi_arr == maxi)
        idx = radi_arr < maxi / scale
        radi_arr[idx] = 0.
        idx = radi_arr > 0.
        radi_arr[idx] = 1.
        half_radi_pool = []
        half_radi_pool.append((int(y[0]), int(x[0])))

        def check(x):  # x is a two components  tuple -like input
            if (x[0] - 1, x[1]) not in half_radi_pool and radi_arr[x[0] - 1, x[1]] == 1:
                half_radi_pool.append((x[0] - 1, x[1]))
            if (x[0] + 1, x[1]) not in half_radi_pool and radi_arr[x[0] + 1, x[1]] == 1:
                half_radi_pool.append((x[0] + 1, x[1]))
            if (x[0], x[1] + 1) not in half_radi_pool and radi_arr[x[0], x[1] + 1] == 1:
                half_radi_pool.append((x[0], x[1] + 1))
            if (x[0], x[1] - 1) not in half_radi_pool and radi_arr[x[0], x[1] - 1] == 1:
                half_radi_pool.append((x[0], x[1] - 1))
            return len(half_radi_pool)

        while True:
            for cor in half_radi_pool:
                num0 = len(half_radi_pool)
                num1 = check(cor)
            if num0 == num1:
                break
        hrp = np.sqrt(len(half_radi_pool) / np.pi)
        return hrp

    def get_radius_new(self, image, scale, size):
        # get the radius of the flux descends to the maximum/scale
        radi_arr = copy.copy(image)
        maxi = np.max(radi_arr)
        y, x = np.where(radi_arr == maxi)
        idx = radi_arr < maxi / scale
        radi_arr[idx] = 0.
        half_radius_pool = []
        flux = []

        def detect(mask, ini_y, ini_x, signal, signal_val):
            if mask[ini_y, ini_x] > 0:
                signal.append((ini_y, ini_x))
                signal_val.append(mask[ini_y, ini_x])
                mask[ini_y, ini_x] = 0
                for cor in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if ini_y + cor[0] < size and ini_x + cor[1] < size and mask[ini_y + cor[0], ini_x + cor[1]] > 0:
                        detect(mask, ini_y + cor[0], ini_x + cor[1], signal, signal_val)
            return signal, signal_val

        half_radius_pool, flux = detect(radi_arr, y[0], x[0], half_radius_pool, flux)

        return np.sqrt(len(half_radius_pool)/np.pi), half_radius_pool, np.sum(flux), maxi, (y, x)

obj = Fourier_trans()
psf = obj.cre_psf(psf_scale=4, imagesize=32)

a = b =c =0
pos = obj.random_gallexy(radius=4, num=60)

for j in range(4):
        rot_pos = obj.rot_pos(theta=j*np.pi/4, pos=pos)
        shear_pos = obj.add_shear(g1=0.01, g2=0.03, pos=rot_pos)
        imag = obj.gaussian_psf(shear_pos, imagesize=32, sigma=4)
        mg1, mg2,mn,wb = obj.shear_est(imag=imag,psf_image=psf, imagesize=32)

        a += mg1
        b += mg2
        c += mn
g1 = a/c
g2 = b/c
print (g1,g2)




"""
obj = Fourier_Quad()
psf = obj.cre_psf(psf_scale=4, imagesize=60)
psf_ps = obj.pow_spec(image=psf)

a = b =c =0
pos_o = obj.ran_pos(6, 40)
for i in range(4):
    pos_r = obj.rotate(pos_o, i*np.pi/4)
    pos_s = obj.shear(pos_r, 0.01, 0.03)

    final = obj.convolve_psf(pos_s, 4, 60,flux=1, psf="GAUSS")
    mg1, mg2, mn = obj.shear_est(final, psf, 60)

    a += mg1
    b += mg2
    c += mn
g1 = a/c
g2 = b/c

"""