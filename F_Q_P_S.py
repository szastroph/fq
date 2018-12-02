
# encoding: utf-8

import numpy as np
from numpy import fft
from scipy.optimize import fmin_cg
from scipy import ndimage, signal
import copy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
import random
from scipy import optimize
from astropy.io import fits
import time
class F_Q_P_S:
    def __init__(self, size, seed):
        self.ran = np.random.RandomState(seed)
        self.size = size
        self.alpha = (2.*np.pi/size)**4
        self.my = np.mgrid[0: size, 0: size][0] - size/2.
        self.mx = np.mgrid[0: size, 0: size][1] - size/2.

        # self.hlr = self.get_radius_new(psf_ps, 2.)[0]  # hlr = self.get_radius_new(psf_ps, 2., x)[0]


    def gauss_noi(self, mean, sigma):
        noise_img = self.ran.normal(loc=mean, scale=sigma, size=self.size * self.size).reshape(self.size, self.size)
        return noise_img

    def poiss_noi(self, gal_imag, lamd):
        poi_noi = np.random.poisson(lamd, self.size ** 2).reshape(self.size, self.size)
        arr = gal_imag + poi_noi
        return arr, poi_noi

    # def ps(self ,image):
    #     image_ps = fft.fftshift(fft.fft2(image))
    #     return image_ps

    def pow_spec(self, image):
        image_ps = fft.fftshift((np.abs(fft.fft2(image))) ** 2)
        return image_ps


    def ran_pos(self ,num ,limit,g=None):
        pos = np.zeros((2 ,num))
        theta = np.random.uniform(low=0, high=2 * np.pi ,size=num)
        x= 0.
        y= 0.
        xn =np.cos(theta)
        yn =np.sin(theta)
        for i in range(num):
            x += xn[i]
            y += yn[i]
            if pow(x,2) + pow(y,2) >= limit*limit:
                x = xn[i]
                y = yn[i]
            pos[0, i] = x
            pos[1, i] = y

        pos[0] = pos[0] - np.mean(pos[0])
        pos[1] = pos[1] - np.mean(pos[1])

        if g:
            # a = (1 - g[0] ** 2 - g[1] ** 2)
            sheared = np.dot(np.array(([(1 + g[0]), g[1]], [g[1], (1 - g[0])] )), pos)
            return sheared
        else:
            return pos

    def add_shear(self,pos,g):
        sheared = np.dot(np.array(([(1 + g[0]), g[1]], [g[1], (1 - g[0])])), pos)
        return sheared


    def ran_gal_gaussian(self, radius, num):
        step = np.random.uniform(low=0, high=radius, size=num)
        theta = np.random.uniform(low=0, high=2 * np.pi, size=num)
        pos = np.matrix([step * np.cos(theta), step * np.sin(theta)])
        return pos

    def rot_pos(self, pos, theta):
        rot_mat = np.matrix([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        return np.dot(rot_mat, pos)

    def cre_psf(self, psf_scale,flux=1., model="GAUSS"):
        if model is 'GAUSS':
            # factor = flux*np.sqrt(1./2/np.pi/psf_scale**2)
            factor = flux*1./2/np.pi/psf_scale**2

            arr = factor*np.exp(-(self.mx**2 + self.my**2)/2./psf_scale**2)
            return arr

        elif model is 'Moffat':
            r_scale_sq = 9
            m = 3.5
            factor = 1./(np.pi*psf_scale**2*((1. + r_scale_sq)**(1.-m) - 1.)/(1.-m))
            rsq = (self.mx**2 + self.my**2) / psf_scale**2
            idx = rsq > r_scale_sq
            rsq[idx] = 0.
            arr = factor*(1. + rsq)**(-m)
            arr[idx] = 0.
            return arr

    def convolve_psf(self, pos, psf_scale, flux=1., psf="GAUSS"):
        x = pos.shape[1]
        arr = np.zeros((self.size, self.size))

        if psf == 'GAUSS':
            # factor = flux*np.sqrt(1./2/np.pi/psf_scale**2)
            factor = flux/2/np.pi/psf_scale**2

            for i in range(x):
                arr += factor*np.exp(-((self.mx-pos[0, i])**2+(self.my-pos[1, i])**2)/2./psf_scale**2)

        elif psf == "Moffat":
            r_scale_sq = 9
            m = 3.5
            factor = flux/(np.pi*psf_scale**2*((1. + r_scale_sq)**(1.-m) - 1.)/(1.-m))
            for l in range(x):
                rsq = ((self.mx-pos[0, l])**2+(self.my-pos[1, l])**2)/psf_scale**2
                idx = rsq > r_scale_sq
                pfunction = factor*(1. + rsq)**(-m)
                pfunction[idx] = 0.
                arr += pfunction
        return arr


    def wbeta(self, radius):
        w_temp = np.exp(-(self.mx ** 2 + self.my ** 2) / radius ** 2)
        return w_temp, 1. / radius

    def w_beta(self, psf_radius):
        beta = psf_radius * 1.20  ###1.17 former
        w_temp = np.exp(-((self.mx) **2 + (self.my)**2)/beta**2)
        beta = 1. / beta
        return w_temp, beta


    def shear_est(self, gal_image, psf,hlr, bg_n=None):
        ky, kx = self.my, self.mx
        psf_ps = self.pow_spec(psf)
        gal_ps = self.pow_spec(gal_image)
        # if background_noise is "poi_noi":  # to deduct the noise
        #     gal_image, background_noise = self.poi_noi(gal_imag=psf_image, lamd=10)
        #     nbg_ps = self.pow_spec(background_noise)
        #     gal_ps = gal_ps - nbg_ps
        if bg_n is not None:
            bgn = self.pow_spec(bg_n)
            gal_ps = gal_ps - bgn
        else:
            gal_ps = self.pow_spec(gal_image)

        # hlr = self.get_radius_new(psf_ps, 2.)[0]  # hlr = self.get_radius_new(psf_ps, 2., x)[0]
        wb, beta = self.wbeta(hlr)  # fourier transpormed w_beta and beta


        maxi = np.max(psf_ps)
        idx = psf_ps < maxi / 10000.
        tk = wb /psf_ps * gal_ps
        tk[idx] = 0.

        mn1 = (-0.5)*(kx**2 - ky**2)
        mn2 = -kx*ky
        mn3 = kx**2 + ky**2 - 0.5*beta**2*(kx**2 + ky**2)**2
        mn4 = kx**4 - 6*kx**2*ky**2 + ky**4
        mn5 = kx**3*ky - kx*ky**3
        mg1 = np.sum(mn1 * tk)*self.alpha
        mg2 = np.sum(mn2 * tk)*self.alpha
        mn  = np.sum(mn3 * tk)*self.alpha
        mu  = np.sum(mn4 * tk)*(-0.5*beta**2)*self.alpha
        mv  = np.sum(mn5 * tk)*(-2.*beta**2)*self.alpha
        return mg1, mg2, mn, mu


    def get_radius_new(self, image, scale):
        # get the radius of the flux descends to the maximum/scale
        radi_arr = copy.copy(image)
        maxi = np.max(radi_arr)
        y, x = np.where(radi_arr == maxi)
        idx = radi_arr < maxi / scale
        radi_arr[idx] = 0.
        half_radius_pool = []
        flux = []

        def detect(mask, ini_y, ini_x, signal, signal_val, size):
            if mask[ini_y, ini_x] > 0:
                signal.append((ini_y, ini_x))
                signal_val.append(mask[ini_y, ini_x])
                mask[ini_y, ini_x] = 0
                for cor in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if -1 < ini_y + cor[0] < size and -1 < ini_x + cor[1] < size \
                            and mask[ini_y + cor[0], ini_x + cor[1]] > 0:
                        detect(mask, ini_y + cor[0], ini_x + cor[1], signal, signal_val, size)
            return signal, signal_val

        half_radius_pool, flux = detect(radi_arr, y[0], x[0], half_radius_pool, flux, self.size)

        return np.sqrt(len(half_radius_pool) / np.pi), half_radius_pool, np.sum(flux), maxi, (y, x)


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
        return np.sqrt(len(half_radi_pool) / np.pi)

    def set_bin(self, data, bin_num, abs_sort=True):
        if abs_sort:
            temp_data = np.sort(np.abs(data))
        else:
            temp_data = np.sort(data[data > 0])
        bin_size = len(temp_data) / bin_num * 2
        bins = np.array([temp_data[int(i * bin_size)] for i in range(1, int(bin_num / 2))])
        bins = np.sort(np.append(np.append(-bins, [0.]), bins))
        bound = np.max(np.abs(data)) * 100.
        bins = np.append(-bound, np.append(bins, bound))
        return bins

    def G_bin(self, g, nu, g_h, bins, ig_num):  # checked 2017-7-9!!!
        r"""
        to calculate the symmetry the shear estimators
        :param g: estimators from Fourier quad, 1-D numpy array
        :param nu: N + U for g1, N - U for g2, 1-D numpy array
        :param g_h: pseudo shear (guess)
        :param bins: bin of g for calculation of the symmetry, 1-D numpy array
        :param ig_num: the number of inner grid of bin to be neglected
        :return: chi square
        """
        bin_num = len(bins) - 1
        inverse = range(int(bin_num / 2 - 1), -1, -1)
        G_h = g - nu * g_h
        num = np.histogram(G_h, bins)[0]
        n1 = num[0:int(bin_num / 2)]
        n2 = num[int(bin_num / 2):][inverse]
        xi = (n1 - n2) ** 2 / (n1 + n2)
        return np.sum(xi[:len(xi)-ig_num]) * 0.5

    def fmin_g(self, g, nu, bin_num, ig_num=0, pic_path=False, left=-0.1, right=0.1):  # checked 2017-7-9!!!
        # nu = N + U for g1
        # nu = N - U for g2
        # temp_data = numpy.sort(numpy.abs(g))[:int(len(g)*0.99)]
        # bin_size = len(temp_data)/bin_num*2
        # bins = numpy.array([temp_data[int(i*bin_size)] for i in range(1, int(bin_num / 2))])
        # bins = numpy.sort(numpy.append(numpy.append(-bins, [0.]), bins))
        # bound = numpy.max(numpy.abs(g)) * 100.
        # bins = numpy.append(-bound, numpy.append(bins, bound))
        bins = self.set_bin(g, bin_num)
        same = 0
        iters = 0
        # m1 chi square & left & left chi square & right & right chi square
        records = np.zeros((15, 5))
        while True:
            templ = left
            tempr = right
            m1 = (left + right) / 2.
            m2 = (m1 + left) / 2.
            m3 = (m1 + right) / 2.
            fL = self.G_bin(g, nu, left, bins, ig_num)
            fR = self.G_bin(g, nu, right, bins, ig_num)
            fm1 = self.G_bin(g, nu, m1, bins, ig_num)
            fm2 = self.G_bin(g, nu, m2, bins, ig_num)
            fm3 = self.G_bin(g, nu, m3, bins, ig_num)
            values = [fL, fm2, fm1, fm3, fR]
            points = [left, m2, m1, m3, right]
            records[iters, ] = fm1, left, fL, right, fR
            if max(values) < 30:
                temp_left = left
                temp_right = right
            if fL > max(fm1, fm2, fm3) and fR > max(fm1, fm2, fm3):
                if fm1 == fm2:
                    left = m2
                    right = m1
                elif fm1 == fm3:
                    left = m1
                    right = m3
                elif fm2 == fm3:
                    left = m2
                    right = m3
                elif fm1 < fm2 and fm1 < fm3:
                    left = m2
                    right = m3
                elif fm2 < fm1 and fm2 < fm3:
                    right = m1
                elif fm3 < fm1 and fm3 < fm2:
                    left = m1
            elif fR > fm3 >= fL:
                if fL == fm3:
                    right = m3
                elif fL == fm1:
                    right = m1
                elif fL == fm2:
                    right = m2
                elif fm1 == fm2:
                    right = right
                elif fm1 < fm2 and fm1 < fL:
                    left = m2
                    right = m3
                elif fm2 < fL and fm2 < fm1:
                    right = m1
                elif fL < fm1 and fL < fm2:
                    right = m2
            elif fL > fm2 >= fR:
                if fR == fm2:
                    left = m2
                elif fR == fm1:
                    left = m1
                elif fR == fm3:
                    left = m3
                elif fm1 < fR and fm1 < fm3:
                    left = m2
                    right = m3
                elif fm3 < fm1 and fm3 < fR:
                    left = m1
                elif fR < fm1 and fR < fm3:
                    left = m3
                elif fm1 == fm3:
                    left = m1
                    right = m3

            if abs(left-right) < 1.e-5:
                g_h = (left+right)/2.
                break
            iters += 1
            if left == templ and right == tempr:
                same += 1
            if iters > 12 and same > 2 or iters > 14:
                g_h = (left+right)/2.
                break
                # print(left,right,abs(left-right))
        # fitting
        left_x2 = np.min(np.abs(records[:iters, 2] - fm1 - 20))
        label_l = np.where(left_x2 == np.abs(records[:iters, 2] - fm1 - 20))[0]
        if len(label_l > 1):
            label_l = label_l[0]

        right_x2 = np.min(np.abs(records[:iters, 4] - fm1 - 20))
        label_r = np.where(right_x2 == np.abs(records[:iters, 4] - fm1 - 20))[0]
        if len(label_r > 1):
            label_r = label_r[0]

        if left_x2 > right_x2:
            right = records[label_l, 3]
            left = 2*m1 - right
        else:
            left = records[label_r, 1]
            right = 2*m1 - left

        g_range = np.linspace(left, right, 80)
        xi2 = np.array([self.G_bin(g, nu, g_hat, bins, ig_num) for g_hat in g_range])

        gg4 = np.sum(g_range ** 4)
        gg3 = np.sum(g_range ** 3)
        gg2 = np.sum(g_range ** 2)
        gg1 = np.sum(g_range)
        xigg2 = np.sum(xi2 * (g_range ** 2))
        xigg1 = np.sum(xi2 * g_range)
        xigg0 = np.sum(xi2)
        cov = np.linalg.inv(np.array([[gg4, gg3, gg2], [gg3, gg2, gg1], [gg2, gg1, len(g_range)]]))
        paras = np.dot(cov, np.array([xigg2, xigg1, xigg0]))

        if pic_path:
            plt.scatter(g_range, xi2)
            plt.plot(g_range, paras[0]*g_range**2+paras[1]*g_range+paras[2])
            s = str(round(paras[0],3)) + " " + str(round(paras[1],3)) + " " + str(round(paras[2],3))
            plt.title(s)
            plt.savefig(pic_path)
            plt.close()

        g_sig = np.sqrt(1 / 2. / paras[0])
        g_h = -paras[1] / 2 / paras[0]

        return g_h, g_sig

    # def  bin_sig(self,data,img_num,mode,bin):
    #     data1 =np.zeros((4,img_num-1,img_num))
    #     g_s = np.zeros((2,img_num))
    #     for i in range(img_num):
    #         data1[:,:,i]=np.delete(data, i, axis=1)
    #         g_s[0,i],g_s[1,i] = self.fmin_g(data1[0],data[2],data[3],mode,bin)
    #     gsig_mean = np.mean(g_s[1])
    #     gh_sig = ((img_num-1)/img_num)*abs(g_s[1]-gsig_mean)**2
    #     return gh_sig






    def bootstr(self,data,spl_num,mode="PDF_SYM",shear =1):
        # sample_data = np.zeros((4,pic_num,spl_num))
        # if shear ==1:
        pic_num = len(data[0])
        samp =np.zeros((spl_num,4,pic_num))

        g_pdf = np.zeros((2,spl_num))
        g_ave = np.zeros((2,spl_num))
        for i in range(spl_num):
            idx = np.random.randint(0,pic_num,pic_num)
            sample =np.array(data[:,[idx]]).reshape(4,pic_num)
            g_pdf[0,i],g_pdf[1,i]= self.fmin_g(sample[shear-1],sample[2],sample[3],1,4)
            g_ave[0,i] = np.mean(sample[shear-1]) / np.mean(sample[2])
            g_ave[1,i] = np.sqrt(np.mean(sample[shear-1]**2)/np.mean(sample[2])**2/pic_num)
            samp[i] = sample
        # np.savetxt("g_pdf.txt",g_pdf)
        # np.savetxt("g_ave.txt",g_ave)
        # g1_pdf_sig =np.std(g_pdf[0])
        # g1_ave_sig =np.std(g_ave[0])
        # g1_p_sig=np.sqrt(np.sum((g_pdf[0]-g[0])**2)/len(g_pdf[0]))
        # g1_a_sig=np.sqrt(np.sum((g_ave[0]-g[0])**2)/len(g_ave[0]))
        # k2_p = np.sum(((g_pdf[0]-g[0])/g_pdf[1])**2)
        # k2_a = np.sum(((g_ave[0]-g[0])/g_ave[1])**2)
        return g_pdf,g_ave,samp#g1_p_sig,g1_a_sig,k2_p,k2_a

        # if mode == "average":
        #     for i in range(spl_num):
        #         idx = np.random.randint(0, pic_num, pic_num)
        #         sample = data[:, [idx]]
        #         g1_spl [i]= np.mean(sample[0])/np.mean(sample[2])
        #      g1_sig = (g1_spl - np.mean(g1_spl)) ** 2 / spl_num


######################
    def move(self, image, x, y):
        imagesize = image.shape[0]
        cent = np.where(image == np.max(image))
        dx = int(np.max(cent[1]) - x)
        dy = int(np.max(cent[0]) - y)
        if dy > 0:
            if dx > 0:
                arr_y = image[0:dy, 0:imagesize]
                arr = image[dy:imagesize, 0:imagesize]
                arr = np.row_stack((arr, arr_y))
                arr_x = arr[0:imagesize, 0:dx]
                arr = arr[0:imagesize, dx:imagesize]
                arr = np.column_stack((arr, arr_x))
                return arr
            elif dx == 0:
                arr_y = image[0:dy, 0:imagesize]
                arr = image[dy:imagesize, 0:imagesize]
                arr = np.row_stack((arr, arr_y))
                return arr
            else:
                arr_y = image[0:dy, 0:imagesize]
                arr = image[dy:imagesize, 0:imagesize]
                arr = np.row_stack((arr, arr_y))
                arr_x = arr[0:imagesize, 0:imagesize + dx]
                arr = arr[0:imagesize, imagesize + dx:imagesize]
                arr = np.column_stack((arr, arr_x))
                return arr
        elif dy == 0:
            if dx > 0:
                arr_x = image[0:imagesize, 0:dx]
                arr = image[0:imagesize, dx:imagesize]
                arr = np.column_stack((arr, arr_x))
                return arr
            elif dx == 0:
                return image
            else:
                arr = image[0:imagesize, 0:imagesize + dx]
                arr_x = image[0:imagesize, imagesize + dx:imagesize]
                arr = np.column_stack((arr_x, arr))
                return arr
        elif dy < 0:
            if dx > 0:
                arr_y = image[imagesize + dy:imagesize, 0:imagesize]
                arr = image[0:imagesize + dy, 0:imagesize]
                arr = np.row_stack((arr_y, arr))
                arr_x = arr[0:imagesize, 0:dx]
                arr = arr[0:imagesize, dx:imagesize]
                arr = np.column_stack((arr, arr_x))
                return arr
            elif dx == 0:
                arr_y = image[imagesize + dy:imagesize, 0:imagesize]
                arr = image[0:imagesize + dy, 0:imagesize]
                arr = np.row_stack((arr_y, arr))
                return arr
            else:
                arr_y = image[imagesize + dy:imagesize, 0:imagesize]
                arr = image[0:imagesize + dy, 0:imagesize]
                arr = np.row_stack((arr_y, arr))
                arr_x = arr[0:imagesize, 0:imagesize + dx]
                arr = arr[0:imagesize, imagesize + dx:imagesize]
                arr = np.column_stack((arr, arr_x))
                return arr

    def get_centroid(self, image, filt=False, radius=2):
        y0,x0 = self.mfpoly(image)
        y, x = np.mgrid[0:image.shape[0], 0:image.shape[1]]
        # m0 = numpy.sum(image)
        # mx  = numpy.sum(x * image)
        # my  = numpy.sum(y * image)
        # x0 = mx / m0
        # y0 = my / m0
        #yc,xc = numpy.where(image==numpy.max(image))
        y = y - y0
        x = x - x0
        if filt == True:
            gaus_filt = np.exp(-(y**2+x**2)/2/np.pi/radius**2)
            image = image *gaus_filt
        mxx = np.sum(x*x*image)
        mxy = np.sum(x * y * image)
        myy = np.sum(y* y * image)
        e1 = (mxx-myy)/(mxx+myy)
        e2 = 2*mxy/(mxx+myy)
        return y0, x0,e1,e2

    def psf_align(self, image):
        imagesize = image.shape[0]
        arr = self.move(image, 0, 0)
        image_f = fft.fft2(arr)
        yy, xx = np.mgrid[0:48, 0:48]
        xx = np.mod(xx + 24, 48) - 24
        yy = np.mod(yy + 24, 48) - 24
        fk = np.abs(image_f) ** 2
        line = np.sort(np.array(fk.flat))
        idx = fk < 4 * line[int(imagesize ** 2/ 2)]
        fk[idx] = 0
        weight = fk / line[int(imagesize ** 2/ 2)]
        kx = xx * 2 * np.pi / imagesize
        ky = yy * 2 * np.pi / imagesize

        def pha(p):
            x, y = p
            return np.sum((np.angle(image_f*np.exp(-1.0j*(kx*x+ky*y))))**2*weight)

        res = fmin_cg(pha, [0, 0], disp=False)
        inve = fft.fftshift(np.real(fft.ifft2(image_f*np.exp(-1.0j*(kx*res[0]+ky*res[1])))))
        return inve

    def gaussfilter(self, psfimage):
        x, y = np.mgrid[0:3, 0:3]
        xc, yc = 1.0, 1.0
        w = 1.3
        ker = (1.0/2.0/np.pi/w/w)*np.exp(-0.5*((x-xc)**2+(y-yc)**2)/2.0/w/w)
        imcov = signal.convolve(psfimage, ker, mode='same')
        return imcov

    def mfpoly(self,psf):
        p = np.where(psf == np.max(psf))
        yr, xr = p[0][0], p[1][0]
        yp, xp = np.mgrid[yr - 1:yr + 2, xr - 1:xr + 2]
        patch = psf[yr - 1:yr + 2, xr - 1:xr + 2]
        zz = patch.reshape(9)
        xx = xp.reshape(9)
        yy = yp.reshape(9)
        xy = xx * yy
        x2 = xx * xx
        y2 = yy * yy
        A = np.array([np.ones_like(zz), xx, yy, x2, xy, y2]).T
        cov = np.linalg.inv(np.dot(A.T, A))
        a, b, c, d, e, f = np.dot(cov, np.dot(A.T, zz))
        coeffs = np.array([[2.0 * d, e], [e, 2.0 * f]])
        mult = np.array([-b, -c])
        xc, yc = np.dot(np.linalg.inv(coeffs), mult)
        return yc, xc

    def segment(self, image):
        shape = image.shape
        y = int(shape[0] / self.size)
        x = int(shape[1] / self.size)
        star = [image[iy * self.size:(iy + 1) * self.size, ix * self.size:(ix + 1) * self.size]
                for iy in range(y) for ix in range(x)]
        for i in range(x):
            if np.sum(star[-1]) == 0:
                star.pop()
        return star  # a list of psfs

    def stack(self, image_array, columns):
        # the inverse operation of divide_stamps
        # the image_array is a three dimensional array of which the length equals the number of the stamps
        num = len(image_array)
        row_num, c = divmod(num, columns)
        if c != 0:
            row_num += 1
        arr = np.zeros((row_num*self.size, columns * self.size))
        for j in range(row_num):
            for i in range(columns):
                tag = i + j * columns
                if tag > num - 1:
                    break
                arr[j*self.size:(j+1)*self.size, i*self.size:(i+1)*self.size] = image_array[tag]
        return arr

    def fit(self, star_stamp, noise_stamp, star_data, mode=1):
        psf_pool = self.segment(star_stamp)
        noise_pool = self.segment(noise_stamp)
        x = star_data[:, 0]
        y = star_data[:, 1]
        sxx = np.sum(x * x)
        syy = np.sum(y * y)
        sxy = np.sum(x * y)
        sx = np.sum(x)
        sy = np.sum(y)
        sz = 0.
        szx = 0.
        szy = 0.
        d=1
        rim = self.border(d)
        n = np.sum(rim)
        for i in range(len(psf_pool)):
            if mode==1:
                pmax = np.max(psf_pool[i])
                arr = psf_pool[i]/pmax#[p[0][0]-1:p[0][0]+1,p[1][0]-1:p[1][0]+1])
                conv = self.gaussfilter(arr)
                dy,dx = self.mfpoly(conv)
                psf = ndimage.shift(arr,(24-dy,24-dx),mode='reflect')
            elif mode==2:
                psf = self.pow_spec(psf_pool[i])
                noise = self.pow_spec(noise_pool[i])
                # psf_pnoise = numpy.sum(rim*psf)/n
                # noise_pnoise = numpy.sum(rim*noise)/n
                psf =psf - noise# -psf_pnoise+noise_pnoise
                pmax = (psf[23,24]+psf[24,23]+psf[24,25]+psf[25,24])/4
                #pmax = numpy.sum(psf)
                psf = psf / pmax
                psf[24,24]=1
            else:
                psf = self.psf_align(psf_pool[i])
                psf = psf/np.max(psf)
            sz += psf
            szx += psf * x[i]
            szy += psf * y[i]
        a = np.zeros((self.size, self.size))
        b = np.zeros((self.size, self.size))
        c = np.zeros((self.size, self.size))
        co_matr = np.array([[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, len(x)]])
        for m in range(self.size):
            for n in range(self.size):
                re = np.linalg.solve(co_matr, np.array([szx[m, n], szy[m, n], sz[m, n]]))
                a[m, n] = re[0]
                b[m, n] = re[1]
                c[m, n] = re[2]
        return a, b, c

    def border(self, edge):
        if edge >= self.size/2.:
            print("Edge must be smaller than half of  the size!")
        else:
            arr = np.ones((self.size, self.size))
            arr[edge: self.size - edge, edge: self.size - edge] = 0.
            return arr