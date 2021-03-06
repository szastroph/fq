import platform
if platform.system() == 'Linux':
    import matplotlib
    matplotlib.use('Agg')
import numpy
import copy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import scipy
import configparser
import logging
import time

################################################################
# the detection methods
################################################################

def detect(mask, ini_y, ini_x, signal, signal_val, y_size, x_size, sflag):
    if mask[ini_y, ini_x] > 0:
        signal.append((ini_y, ini_x))
        signal_val.append(mask[ini_y, ini_x])
        mask[ini_y, ini_x] = 0
        for cor in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            if -1 < ini_y + cor[0] < y_size and -1 < ini_x + cor[1] < x_size and mask[ini_y + cor[0], ini_x + cor[1]]>0:
                if ini_y + cor[0] == 0 or ini_y + cor[0] == y_size-1 or ini_x + cor[1] == 0 or ini_x + cor[1]==x_size-1:
                    sflag = 1
                detect(mask, ini_y + cor[0], ini_x + cor[1], signal, signal_val, y_size, x_size, sflag)
    return signal, signal_val, sflag


def stamp_detector(image, thres, y_size, x_size, ra=10):
    # get the source object
    image_c = copy.copy(image)
    img_idx = image_c < thres
    image_c[img_idx] = 0.

    center = image_c[int(y_size/2-ra):int(y_size/2+ra), int(x_size/2-ra):int(x_size/2+ra)]
    y, x = numpy.where(center > 0)
    y_sour = y + int(y_size/2 - ra)
    x_sour = x + int(x_size/2 - ra)

    final_obj = []
    final_flux = []
    flag = 0
    for m in range(len(x_sour)):
        sour_pool = []
        sour_flux = []
        yp = y_sour[m]
        xp = x_sour[m]

        if image_c[yp, xp] > 0:
            sour_pool, sour_flux, flag = detect(image_c, yp, xp, sour_pool, sour_flux, y_size, x_size, flag)

        if len(sour_pool) > len(final_obj):
            final_obj = sour_pool
            final_flux = sour_flux
        elif len(sour_pool) == len(final_obj):
            if numpy.max(final_flux) < numpy.max(sour_flux):
                final_obj = sour_pool
                final_flux = sour_flux
    if len(final_flux) == 0:
        peak = 0
    else:
        peak = numpy.max(final_flux)
    return final_obj, numpy.sum(final_flux), numpy.sum((numpy.array(final_flux))**2), peak, flag


def find_binary(image, ysize, xsize, sig):
    my, mx = numpy.mgrid[0:5,0:5]
    w = 3
    gauss = 1./2/numpy.pi/w/w*numpy.exp(-((my-2.5)**2+(mx-2.5)**2)/2/w**2)
    # the padding is to avoid the boundary crossing in the local peak searching
    image_c = numpy.lib.pad(scipy.signal.convolve(image,gauss,mode="same"),2,mode="constant",constant_values=0)
    mask = copy.deepcopy(image_c)
    idx = mask < 1.5*sig
    mask[idx] = 0
    mask_0 = copy.deepcopy(mask)
    p = numpy.where(mask > 0)
    yp, xp = p[0], p[1]
    objects = []
    peaks = []
    for j in range(len(xp)):
        if mask[yp[j], xp[j]] > 0:
            cache = [(yp[j], xp[j])]
            peak = []
            mask[yp[j], xp[j]] = 0
            num = 2
            num0 = 1
            while True:
                num_new = num0 - num
                if num == num0:
                    break
                num0 = len(cache)
                p_new = []
                for k in range(num_new, 0):
                    xy = cache[k]
                    for coord in ((xy[0] + 1, xy[1]), (xy[0] - 1, xy[1]), (xy[0], xy[1] - 1), (xy[0], xy[1] + 1),
                          (xy[0] -1, xy[1] -1), (xy[0] - 1, xy[1]+1), (xy[0]+1, xy[1] - 1), (xy[0]+1, xy[1] + 1)):
                        if -1 < coord[0] < ysize and -1 < coord[1] < xsize and mask[coord[0], coord[1]] > 0:
                            p_new.append(coord)
                            mask[coord[0], coord[1]] = 0
                            if image_c[coord[0], coord[1]] >= numpy.max(image_c[coord[0] - 2:coord[0] + 3, coord[1] - 2:coord[1] + 3]):
                                peak.append(coord)
                cache.extend(p_new)
                num = len(cache)
            if num > 5:
                objects.append(cache)
                peaks.append(peak)
    return objects, peaks, image_c, mask_0


def source_detector(img_arr, area_thresh, noise_level, cross=True):
    """
    to find the sources in the 2-D image
    :param img_arr: the 2-D numpy array, image
    :param area_thresh: the threshold for a source detection
    :param noise_level: the threshold of source value above which the pixel will be regarded as source
    :param cross: If True, the algorithm will check the nearest 4 pixels of the pixels of source
                  if False, the algorithm will check the nearest 8 pixels around the pixels of source
    :return: list of source coordinates [[..., (y_i, x_i),..], ..., [...]], each sublist contains tuples of the
            x- and y- coordinates
    """
    ysize, xsize = img_arr.shape
    mask = img_arr.copy()
    idx = mask < noise_level
    mask[idx] = 0
    objects = []
    p = numpy.where(mask > 0)
    yp, xp = p[0], p[1]

    # the relative coordinates of the nearest pixels
    if cross:
        # "+"
        relative_y, relative_x = [-1, 1, 0, 0], [0, 0, -1, 1]
        check_num = 4
    else:
        # "+" & "x"
        relative_y, relative_x = [-1, -1, -1, 0, 0, 1, 1, 1], [-1, 0, 1, -1, 1, -1, 0, 1]
        check_num = 8
    for j in range(len(xp)):
        if mask[yp[j], xp[j]] > 0:
            cache = [(yp[j], xp[j])]
            mask[yp[j], xp[j]] = 0
            num = 2
            num0 = 1
            while True:
                num_new = num0 - num
                if num == num0:
                    break
                num0 = len(cache)
                p_new = []
                for k in range(num_new, 0):
                    xy = cache[k]
                    #for coord in ((xy[0] + 1, xy[1]), (xy[0] - 1, xy[1]), (xy[0], xy[1] - 1), (xy[0], xy[1] + 1)):
                    for i in range(check_num):
                        coord_y, coord_x = xy[0] + relative_y[i], xy[1] + relative_x[i]
                        if -1 < coord_y < ysize and -1 < coord_x < xsize and mask[coord_y, coord_x] > 0:
                            p_new.append((coord_y, coord_x))
                            mask[coord_y, coord_x] = 0
                cache.extend(p_new)
                num = len(cache)
            if num >= area_thresh:
                objects.append(cache)
    return objects


def get_hlr(image, scale, size, thres=None):
    # get the source object, to avoid the overflow of the stack
    mask = copy.copy(image)
    maxi = numpy.max(image[int(size/2-6):int(size/2+6), int(size/2-6):int(size/2+6)])
    y, x = numpy.where(mask == maxi)
    if thres:
        idx = mask < thres
        mask[idx] = 0.
    else:
        idx = mask < maxi / scale
        mask[idx] = 0.
    flux = maxi
    flux_sq = maxi**2

    cache = [(x, y)]
    mask[y, x] = 0
    num = 2
    num0 = 1
    while True:
        num_new = num0 - num
        if num == num0:
            break
        num0 = len(cache)
        p_new = []
        for k in range(num_new, 0):
            xy = cache[k]
            for coord in ((xy[0] + 1, xy[1]), (xy[0] - 1, xy[1]), (xy[0], xy[1] - 1), (xy[0], xy[1] + 1)):
                if -1 < coord[0] < size and -1 < coord[1] < size and mask[coord[0], coord[1]] > 0:
                    p_new.append(coord)
                    flux += mask[coord[0], coord[1]]
                    flux_sq += mask[coord[0], coord[1]] ** 2

                    mask[coord[0], coord[1]] = 0
        cache.extend(p_new)
        num = len(cache)

    return numpy.sqrt(len(cache)/numpy.pi), flux, flux_sq, num, maxi


def smooth(image,size):
    my, mx = numpy.mgrid[-2:3, -2:3]
    x, y = mx.reshape((1, 25)), my.reshape((1, 25))
    cen = int((size * size + size) / 2)
    fit_img = numpy.zeros_like(image)
    for i in range(size):
        for j in range(size):
            arr = numpy.zeros((size, size))
            arr[int(size / 2), int(size / 2)] = 0.5
            pos = []
            tag = 0
            pk = 0
            z = []
            x_d = []
            for m in range(-2, 3):
                p = (i + m + size) % size
                for n in range(-2, 3):
                    q = (j + n + size) % size

                    if tag not in [0, 4, 20, 24]:  # abs(m) != 2 or abs(n) != 2:
                        if p * size + q != cen:
                            pos.append((p, q))
                            z.append(image[p, q])
                            x_d.append((x[0, tag], y[0, tag]))
                        else:
                            pk = tag
                    tag += 1

            x_d = numpy.array(x_d).T
            # if i ==j ==0:
            #     print x_d
            a1, a2 = curve_fit(fxy, x_d, z)
            fit_img[i, j] = a1[0]
    return fit_img

def fit_2o(image):

    ma = numpy.max(image)/10**12.
    a = numpy.log(image+ma)
    m = len(image)

    alpha = numpy.load('alpha.npz')['arr_0']
    alpha = numpy.array(alpha)
    # print alpha,alpha.shape
    img_new = numpy.zeros_like(image)
    img_cur = numpy.zeros_like(image)
    data = numpy.zeros((m+4,m+4))
    data[2:m+2, 2:m+2] = a
    data[0, 2:m + 2] = a[-2]
    data[1, 2:m + 2] = a[-1]
    data[m + 2, 2:m + 2] = a[0]
    data[m + 3, 2:m + 2] = a[1]
    data[:, 0] = data[:, m]
    data[:, 1] = data[:, m + 1]
    data[:, -1] = data[:, 3]
    data[:, -2] = data[:, 2]
    #get  one bigger image to fit the transformation
    mx, my = numpy.mgrid[-2:3, -2:3]
    for i in range(m):
        for j in range(m):
            mask = data[i:i+5, j:j+5]
            alph = alpha[i*m+j,:,:]

            fi = get_alpha(i,j,m,mask,my,mx)

            if i ==3 and j ==10:
                print alph.shape,fi.shape
            para = numpy.dot(alph, fi)

            img_new[i,j] = para[0]

    img_new = numpy.exp(img_new-ma)
    # img_cur = numpy.exp(img_cur-ma)

    return img_new

def get_alpha(a, b, size, mask,my,mx,task = False):  # a,b is  the index of the centre point of power spectrum
    x, y,mask = mx.reshape((1, 25)), my.reshape((1, 25)),mask.reshape((1,25))
    # if task == True:

    if (size/2 +2-a-2) ** 2 + (size/2 +2-b-2) ** 2 < 6 and a<=size/2+2 and b<=size/2+2:
        # print (a,b)
        index = [0, 4, 20, 24, int(size/2+2 - a)*5+ int(size/2+2 -b)]
        # print int(size/2+2 - a)*5 + int(size/2+2 -b)
        x_new, y_new, mask_new = numpy.delete(x, index),numpy.delete(y, index), numpy.delete(mask, index)
    else:
        index = [0, 4, 20, 24]
        x_new, y_new, mask_new = numpy.delete(x, index), numpy.delete(y, index),numpy.delete(mask,index)
    a1 = len(x_new)                           # sum of        1
    a2 = numpy.sum(x_new)                     # sum of        xi
    a3 = numpy.sum(y_new)                     # sum of        yi
    a4 = numpy.sum(x_new ** 2)                # sum of       xi^2
    a5 = numpy.sum(x_new * y_new)             # sum 0f       xi*yi
    a6 = numpy.sum(y_new ** 2)                # sum of       yi^2
    a7 = numpy.sum(x_new ** 3)                # sum of       xi^3
    a8 = numpy.sum(x_new ** 2 * y_new)        # sum of       xi^2*yi
    a9 = numpy.sum(x_new * y_new ** 2)        # sum of       xi*yi^2
    a10 = numpy.sum(y_new ** 3)               # sum of       yi^3
    a11 =numpy. sum(x_new ** 4)               # sum of       xi^4
    a12 = numpy.sum(x_new ** 3 * y_new)       # sum of       xi^3*yi
    a13 = numpy.sum(x_new ** 2 * y_new ** 2)  # sum of       xi^2*yi^2
    a14 = numpy.sum(x_new * y_new ** 3)       # sum of       xi*yi^3
    a15 =numpy.sum(y_new ** 4)                # sum of       yi^4
    alp = numpy.matrix([[a1, a2, a3, a4,  a5,  a6],
                        [a2, a4, a5, a7,  a8,  a9],
                        [a3, a5, a6, a8,  a9,  a10],
                        [a4, a7, a8, a11, a12, a13],
                        [a5, a8, a9, a12, a13, a14],
                        [a6, a9, a10,a13, a14, a15]])
    alph = numpy.linalg.inv(alp)

    f  = numpy.sum(mask_new)
    fx = numpy.sum(mask_new*x_new)
    fy = numpy.sum(mask_new*y_new)
    fx2 = numpy.sum(mask_new*x_new**2)
    fxy = numpy.sum(mask_new*x_new*y_new)
    fy2 = numpy.sum(mask_new*y_new**2)
    f_z = numpy.array([[f], [fx], [fy], [fx2], [fxy], [fy2]])

    # x_d = numpy.vstack((x_new,y_new))
    # a1, a2 = curve_fit(fxy, x_d, mask_new)
    # im_d=a1[5]

    return f_z

def fit_2o2(image):
    size = len(image)+4
    m = len(image)

    a = numpy.log10(image)
    data = numpy.zeros((size, size))
    data[2:m + 2, 2:m + 2] = a
    data[0, 2:m + 2] = a[-2]
    data[1, 2:m + 2] = a[-1]
    data[m + 2, 2:m + 2] = a[0]
    data[m + 3, 2:m + 2] = a[1]
    data[:, 0] = data[:, m]
    data[:, 1] = data[:, m + 1]
    data[:, -1] = data[:, 3]
    data[:, -2] = data[:, 2]

    gal_log = data        # we need 52*52,now it's 52*52

    # gal_log_fit = tool_box.smooth(gal_log, size)
    # gal_fit = 10**gal_log_fit
    inv = numpy.load('alpha.npz')['arr_0']
    new = numpy.zeros_like(gal_log)
    for row in range(2, size - 2):
        for col in range(2, size - 2):
            tag = 0
            x = []
            y = []
            z = []
            pk = 0
            for i in range(-2, 3):
                for j in range(-2, 3):
                    if tag not in [0, 4, 20, 24]:
                        if (i + row) * size + j + col != (size + 1) * size / 2:
                            x.append(j)
                            y.append(i)
                            z.append(gal_log[i + row, j + col])
                        else:
                            pk = tag
                    tag += 1
            # print(pk, len(x))
            # arr = numpy.zeros((size, size))
            # arr[int(size/2), int(size/2)] = 2
            # for ii in range(len(x)):
            #     arr[row+y[ii], col+x[ii]] = 1
            #
            # plt.imshow(arr)
            # plt.show()
            x = numpy.array(x)
            y = numpy.array(y)
            z = numpy.array(z)

            f = z.sum()
            fx = numpy.sum(z * x)
            fy = numpy.sum(z * y)
            fx2 = numpy.sum(z * x * x)
            fxy = numpy.sum(z * x * y)
            fy2 = numpy.sum(z * y * y)
            # n = len(x)
            # x1 = x.sum()
            # x2 = numpy.sum(x * x)
            # x3 = numpy.sum(x * x * x)
            # x4 = numpy.sum(x * x * x * x)
            #
            # y1 = y.sum()
            # y2 = numpy.sum(y * y)
            # y3 = numpy.sum(y * y * y)
            # y4 = numpy.sum(y * y * y * y)
            #
            # xy = numpy.sum(x * y)
            # x2y = numpy.sum(x * x * y)
            # xy2 = numpy.sum(x * y * y)
            # x3y = numpy.sum(x * x * x * y)
            # xy3 = numpy.sum(x * y * y * y)
            # x2y2 = numpy.sum(x * x * y * y)
            #
            # cov = numpy.array([[n, x1, y1, x2, xy, y2],
            #                    [x1, x2, xy, x3, x2y, xy2],
            #                    [y1, xy, y2, x2y, xy2, y3],
            #                    [x2, x3, x2y, x4, x3y, x2y2],
            #                    [xy, x2y, xy2, x3y, x2y2, xy3],
            #                    [y2, xy2, y3, x2y2, xy3, y4]])
            # inv_cov = numpy.linalg.inv(cov)
            inv_cov = inv[(row-2)*48+col-2,:,:]
            f_z = numpy.array([[f], [fx], [fy], [fx2], [fxy], [fy2]])
            # if row ==49 and col == 48:
            #     print f_z,inv[(row-2)*48+col-2]
            para = numpy.dot(inv_cov, f_z)
            new[row, col] = para[0, 0]

    new_p = new[2:size-2,2:size-2]
    new_ps =10**(new_p)#-numpy.max(image)/10**12.
    return new_ps


################################################################
# the fitting methods
################################################################


def exp_fun(x, ampli, sig, mu):
    # for fitting
    return ampli * numpy.exp(-(x - mu) ** 2 / 2 / sig ** 2)


def fxy(x, a, b, c, d, e, f):
    return a + b*x[0] + c*x[1] + d*x[0]**2 + e*x[0]*x[1] + f*x[1]**2
    # return a*x[0]**2 + b*x[0]*x[1]+c*x[1]**2+d *x[0]+e*x[1]+f

def gaussnosie_fit(data, bin_num):
    r"""
    fit the Gaussian distribution (amplitude, sigma, mu)
    :param data: the
    :param bin_num:
    :return:
    """
    num, bins = numpy.histogram(data, bin_num)
    num = num/numpy.sum(num)
    mp = numpy.where(num == numpy.max(num))[0][0]
    coeff, coerr = curve_fit(f=exp_fun, xdata=bins[1:], ydata=num)
    # the fitted sigma can be negative
    return coeff, coerr, bins, num


def gauss_fit(x, f, method='scipy'):
    r"""
    to fit the Gaussian function
    f(x,y,...) = A*EXP(-SUM (x_i - mu_i)**2/2/sig_i**2)

    it is designed for the case that the scipy's curve_fit() fails

    :param x: a list of coordinates, (n,) numpy array,
    :param f: the measured value, (n,) numpy array
    :param method: scipy curve fitting or the least square method
    :return: list of coefficients, [...[mean_i, sigma_i^2],...,[A]]
            the last component is the coefficient A.
    """
    # dimension
    idx = f >= f.max()*0.05
    nd = len(x)
    ln_f = numpy.log(f[idx])
    if method == "scipy":
        X = numpy.array([x[i][idx]**j for i in range(nd) for j in range(2,-1,-1)]).T
        ones = numpy.ones((X.shape[0], 1))
        X = numpy.column_stack((X, ones))
        res = scipy.linalg.lstsq(X,ln_f)[0]
        coeff = []
        for i in range(nd):
            ai,bi,ci = res[i*3], res[i*3+1], res[i*3+2]
            mu_i = -0.5*bi/ai
            sig_i2 = mu_i/bi
            coeff.append([mu_i, sig_i2])
        coeff.append([numpy.exp(2*res[-1])]) # 2!!
        return coeff
    else:
        # developing
        return None


def fit_1d(x, y, order, method):
    r"""
    fit a polynomial to n-order
    a1 + a2*x + a3*x^2  .....
    the powers of the polynomial can be written as \SUM_{i~N} X^{i}

    :param x: 1-D numpy array, x
    :param y: 1-D numpy array, the measured values
    :param order: the highest order of target polynomial
    :param method: leastsq or scipy, all the subroutines do fitting basing on least square method
    :return: (n,1) numpy array, the target coefficients
    """
    turns = order + 1
    x = x*1.0
    if method == "lsq":
        pows = [[i + j for i in range(turns)] for j in range(turns)]
        fxy = [numpy.sum(y * (x ** pows[0][i])) for i in range(turns)]
        cov = [[numpy.sum(x**pows[i][j]) for i in range(turns)] for j in range(turns)]
        res = numpy.dot(numpy.linalg.inv(numpy.array(cov)), numpy.array(fxy))
        return res
    elif method == "scipy":
        x = numpy.array([x**i for i in range(turns)]).T
        res = scipy.linalg.lstsq(x,y)[0]
        return res
    else:
        raise ValueError("method must be one of \"lsq, scipy\"")


def fit_2d(x, y, fun_val, order):
    r"""
    fit a polynomial to n-order, a1 + a2*x + a2*y + a3*x^2 + a4*x*y + a5*y^2 .....
    the powers of the polynomial can be written as \SUM_{i~N}\SUM_{j~i} X^{i-j}*y^{j}
    :param x: 1-D numpy array, x
    :param y: 1-D numpy array, y
    :param fun_val: 1-D numpy array, the measured values
    :param order: the highest order of the target polynominal
    :return: (n,1) numpy array, coefficients of the target polynomial
    """
    x, y = x * 1.0, y * 1.0
    turns = int((order + 1) * (order + 2) / 2)
    pows = [(i-j, j) for i in range(order+1) for j in range(i+1)]
    fxy = [[numpy.sum(fun_val * (x ** pows[i][0]) * (y ** pows[i][1]))] for i in range(turns)]
    cov = [[numpy.sum(x**(pows[i][0]+pows[j][0])*y**(pows[i][1]+pows[j][1])) for i in range(turns)] for j in range(turns)]
    res = numpy.dot(numpy.linalg.inv(numpy.array(cov)), numpy.array(fxy))
    return res, pows


def fit_background(image, pix_num, function, pix_lb, pix_ub, my, mx, seqs, yblock=1, xblock=1, order=1, sort=True, ax=None):
    r"""
    fit the background noise with n-order polynomial
    !!! sort=True is highly recommended for either the background fitting or noise fitting
    :param image: the image
    :param pix_num: the number of pixel for fitting
    :param function: "flat" for fitting background noise value in the plane,
                    "gauss" for fitting the sigma of the noise,
                    "sort=True" is suggested,
                    and the sigma fitting should be called after the background removing
    :param pix_lb, pix_ub : the lower (upper) bound of value of the target chosen pixel for fitting,
                            to avoid the bad and the saturated pixels.
    :param my, mx, seqs: my, mx are the grids of coordinates made by numpy.mgrid[...],
                        seqs is a list of the No. of the each cross of grids, seqs = numpy.arange(0, len(my))
                        these three parameters are made outside for time saving
    :param yblock, xblock : number of the block in y-axis, x-axis
    :param order: the highest order of polynomial
    :param sort: Boolean, True: fitting with the mid-part (0.3~0.7) of the PDF of pixel values
                        False: fitting with all the pixels chosen
    :return: list of coefficients in each block,
            the sub-lists of the coefficients of each block are arranged like the 1-D numpy
            array which is reshape from the 2-D numpy array.
            if function=="gauss", the list looks like [.., [[mean,sigma],[amplitude]], [[],[]], ...]
    """
    y, x = image.shape
    ystep, xstep = int(y/yblock), int(x/xblock)
    fit_paras = []
    # rng = numpy.random.RandomState(num)
    for i in range(yblock):
        for j in range(xblock):
            # my, mx = numpy.mgrid[i*ystep:(i+1)*ystep, j*xstep:(j+1)*xstep]
            # my = my.flatten()
            # mx = mx.flatten()
            # tags = numpy.arange(0,len(my))
            # ch_tag = rng.choice(tags, num, replace=False)
            ch_seqs = numpy.random.choice(seqs, pix_num, replace=False)
            ys = my[ch_seqs]
            xs = mx[ch_seqs]
            fz = image[i*ystep:(i+1)*ystep, j*xstep:(j+1)*xstep].flatten()[ch_seqs]
            if sort:
                fz_s = numpy.sort(fz)
                bottom, upper = fz_s[int(pix_num*0.2)], fz_s[int(pix_num*0.8)]
                idx_1 = fz >= max(pix_lb, bottom)
                idx_2 = fz <= min(pix_ub, upper)
                # print(function, pix_lb, bottom, pix_ub, upper)
                if function == "flat":
                    para = fit_2d(xs[idx_1&idx_2],ys[idx_1&idx_2],fz[idx_1&idx_2],order)
                elif function == "gauss":
                    nums, bins = numpy.histogram(fz[idx_1&idx_2], 100)
                    para = gauss_fit([bins[:-1]], nums)
                    if ax:
                        a, b, c = para[1][0], para[0][0], para[0][1]
                        px = bins[:-1]
                        ax.hist(fz[idx_1&idx_2], 100, alpha=0.5, color="green")
                        ax.plot(px, a*numpy.exp(-(px-b)**2/2/abs(c)), c="orange")
                        ax.text(0.3, 0.35, "sig: %.2f (%.2f)"%(numpy.sqrt(abs(c)),c), horizontalalignment='center',
                                verticalalignment='center',transform=ax.transAxes,color="dimgray")
                        ax.text(0.3, 0.25, "mu: %.2f"%b, horizontalalignment='center',
                                verticalalignment='center',transform=ax.transAxes,color="dimgray")
                        ax.text(0.3, 0.15, "A: %.2f"%a, horizontalalignment='center',
                                verticalalignment='center',transform=ax.transAxes,color="dimgray")

                else:
                    raise ValueError("function must be one of \"flat, gauss\"")
            else:
                if function == "flat":
                    para = fit_2d(xs,ys,fz,order)
                elif function == "gauss":
                    nums, bins = numpy.histogram(fz, 100)
                    para = gauss_fit([bins[:-1]],nums)
                else:
                    raise ValueError("function must be one of \"flat, gauss\"")
            fit_paras.append(para)

    return fit_paras

def data_fit(x_data, y_data, y_err):
    r"""
    Y = A*X ,   y = m*x+c
    Y = [y1,y2,y3,...].T  the measured data
    A = [[1,1,1,1,...]
         [x1,x2,x3..]].T
    X = [c,m].T
    C = diag[sig1^2, sig2^2, sig3^2, .....]
    the inverse of C is used as weight of data
    X = [A.T*C^(-1)*A]^(-1) * [A.T*C^(-1) *Y]
    :param x_data:
    :param y_data:
    :param y_err:
    :return:
    """
    A1 = numpy.column_stack((numpy.ones_like(x_data.T), x_data.T))
    Y1 = y_data.T
    C1 = numpy.diag(y_err ** 2)
    L1 = numpy.linalg.inv(numpy.dot(numpy.dot(A1.T, numpy.linalg.inv(C1)), A1))
    R1 = numpy.dot(numpy.dot(A1.T, numpy.linalg.inv(C1)), Y1)
    sig_m1 = numpy.sqrt(L1[1, 1])
    sig_c1 = numpy.sqrt(L1[0, 0])
    mc = numpy.dot(L1, R1)
    return mc[1], sig_m1, mc[0], sig_c1


################################################################
# the methods relate to distribution of data
################################################################
def rand_gauss2(x_range, y_range, num, cov):
    # return a 2-variables gaussian distribution
    # cxy is the correlation between the two variables and must be smaller than the sigma!
    xs = []
    ys = []
    sigx, sigy, cxy = cov
    A = (sigx * sigy) ** 2 - cxy ** 2
    coeff = 0.5/numpy.pi/numpy.sqrt(A)

    while len(xs) < num:
        num_gap = (num - len(xs))
        x = numpy.random.uniform(x_range[0], x_range[1], num_gap)
        y = numpy.random.uniform(y_range[0], y_range[1], num_gap)
        z = numpy.random.uniform(0, coeff, num_gap)
        resi = z - coeff*numpy.exp(-0.5*((x*sigy)**2 + 2*cxy*x*y + (sigx*y)**2)/A)
        idx = resi <= 0
        if len(x[idx]) > num_gap:
            xs.extend(x[idx][:num_gap].tolist())
            ys.extend(y[idx][:num_gap].tolist())
        else:
            xs.extend(x[idx].tolist())
            ys.extend(y[idx].tolist())
        if len(xs) == num:
            break
    return numpy.column_stack((numpy.array(xs),numpy.array(ys)))


def rand_gauss2n(num, means, cov, xy_range=None):
    r"""
    basing on numpy, to generate two sets of correlated data in (2,n) numpy array
    :param xy_range: list of the bound of the two sets [x_start, x_end, y_start, y_end]
    :param num: number
    :param means: list of means [mean1,mean2]
    :param cov: list, covariance matrix [[c11, c12],[c12,c11]]
    :return: (2,n) numpy array
    """
    finals = [[],[]]
    while True:
        nl = len(finals[0])
        gap = num - nl
        if gap == 0:
            break
        xy = numpy.random.multivariate_normal(means,cov,gap)
        if xy_range:
            idx1 = xy[:,0] <= xy_range[0]
            idx2 = xy[:,0] >= xy_range[1]
            idy1 = xy[:,1] <= xy_range[2]
            idy2 = xy[:,1] >= xy_range[3]
            target = xy[idx1&idx2&idy1&idy2]
        else:
            target = xy
        if target.shape[0] > 0:
            finals[0].extend(target[:,0].tolist())
            finals[1].extend(target[:,1].tolist())
    data = numpy.array(finals)
    return data

def ran_generator(pdf, num, seed, *args):
    """
    generate the n-dimension random according to the given pdf
    :param pdf: the probability distribution function
    :param num: size
    :param seed: initialize the random generator
    :param args: length = dimensions*2+2, the lower and upper bound for each dimension
                !!! and plus the last two values (also the lower and upper bound) for
                !!! the PDF function values to compare to.
                !!! the last pair should contain the region of pdf
    :return: (n,num) numpy array
    """
    dims = divmod(len(args), 2)[0] - 1
    final_vars = [numpy.array([]) for i in range(dims)]
    rng = numpy.random.RandomState(seed)
    while True:
        gap = num - len(final_vars[0])
        if gap == 0:
            break
        vars = [rng.uniform(args[i*2], args[(i*2+1)], int(gap*2)) for i in range(dims+1)]
        diff = vars[-1] - pdf(vars[:dims])
        idx = diff <= 0
        candi_num = len(vars[0][idx])
        if candi_num <= gap:
            for i in range(dims):
                final_vars[i] = numpy.append(final_vars[i], vars[i][idx])
        else:
            for i in range(dims):
                final_vars[i] = numpy.append(final_vars[i], vars[i][idx][:gap])
    return numpy.array(final_vars)

def disc_r_pdf(r, scale):
    """
    PDF of the semi-major axis scale-length of a specific scaled radius obtained from
    the relation: ln(R) = -1.145 - 0.269*(MAG - 23)
    P(r) ~ r*exp[-(r/scale)**(4/3)]
    See Miller et al, 2013, MNRAS

    the maximum of scaled radius is
    scale*0.75**0.75*exp[-0.75] ~ 0.381*scale

    :param r: a list of numpy array of radius, [ ]
    :param scale: the specific scaled radius, scale = rd/0.833
    :return: PDF(r), r*exp[-(r/scale)**(4/3)]
    """
    return r[0]*numpy.exp(-(r[0]/scale)**(4./3))

def disc_e_pdf(ellip):
    """
    the pdf for the ran_generator() method
    to generator the ellipticities of disc-dominated galaxies.
    P(e) = A*e*(1-exp((e-e_m)/a))/((1+e)*sqrt(e^2+e_0^2))
    A, e_m, e_0, a = 2.4318, 0.804, 0.0256, 0.2539
    See Miller et al, 2013, MNRAS

    :param ellip: list of numpy array
    :return: probability, maximum=2.0207
    """
    return 2.4318*ellip[0]*(1-numpy.exp((ellip[0] - 0.804)/0.2539))/((1+ellip[0])*numpy.sqrt(ellip[0]**2+0.0256**2))

def bulge_e_pdf(ellip):
    """
    the pdf for the ran_generator() method
    to generator the ellipticities of bulge-dominated galaxies.
    P(e) = 27.7478 * e * numpy.exp(-b * e - c * e * e) for e ~ (0, 1)
    b, c = 2.368, 6.691
    the factor "27.8366" for e ~ (0, 0.804)
    See Miller et al, 2013, MNRAS

    :param ellip: list of numpy array
    :return: probability, maximum=2.653
    """
    return 27.8366 * ellip[0] * numpy.exp(-2.368 * ellip[0] - 6.691*ellip[0]*ellip[0])

def mag_generator(num, mag_min, mag_max):
    """
    to generate the magnitudes by fitting the CFHTLenS i-band catalog
    :param num: total number
    :param mag_min, mag_max: bound of magnitude
    :return: 1-D numpy array
    """
    m = numpy.linspace(mag_min, mag_max, 1000000)
    # the parameters come from the fitting to the CFHTLenS i-band catalog
    pm = 10**(0.294895*m - 1.082894)
    pm = pm/numpy.sum(pm)
    new_pdf = numpy.random.choice(m, num, p=pm)
    return new_pdf


def mag_to_flux(mag, zpt=25.77, exp_time=600, area=8.0216, gain=1.5):
    """
    calculate the flux (ADU) relates to the magnitude,
    the default values are for CFHTLenS,
    flux = 10**(-0.4*(mag - zeropoint))*exposure time*effective area
    :param mag: numpy array or float
    :param zpt: zero point, default for CFHTLenS, 25.77 from the observational images
    :param exp_time: exposure time, second,
                    default for CFHTLenS, ~ 600 sec from the observational images
    :param area: effective area of the camera, m^2, default for CFHTLenS, 8.0216 m^2
    :param gain: default for the CFHTLenS, ~1.5 from the observational images
    :return: numpy array or float, depends on the type of the input,
            the total ADU number of a galaxy

    the default effective zero point: 34.5358
    """
    return 10**(-0.4*(mag - zpt))*exp_time*area/gain

def mag_to_radius(mag):
    """
    ln(R) = -1.145 - 0.269*(MAG - 23)
    See Miller et al, 2013, MNRAS, APPENDIX B1
    :param mag:
    :return: semi-major axis scale length, arcsec
    """
    return numpy.exp(-1.145-0.269*(mag-23))

def radii_from_mags(mag, ra_lb, ra_ub):
    """
    generate the radii from the magnitudes
    :param mag: numpy array, magnitude
    :param ra_lb, ra_ub: lower and upper bound of radius
    :return: numpy array, radii
    """
    num, mag_bin = numpy.histogram(mag, 500)
    mid_mags = mag_bin[:-1] + (mag_bin[1] - mag_bin[0])/2
    mag_bin[0] -= 0.0005
    # a = rd/0.833
    a = mag_to_radius(mid_mags)/0.833
    radii = numpy.zeros_like(mag)
    for i in range(len(num)):
        pr_ = lambda r: disc_r_pdf(r, a[i])
        if num[i] == 0:
            continue
        idx = (mag_bin[i] < mag) & (mag <= mag_bin[i+1])
        sub_num = idx.sum()
        radii[idx] = ran_generator(pr_, sub_num, i, ra_lb, ra_ub, 0, a[i]*0.39)[0]
    return radii


def bulge_frac_pdf(fraction):
    """
    PDF of Bulge-to-Total fraction for disc-dominated galaxies
    see Miller et al, 2013, MNRAS
    :param fraction: list of numpy array, bulge-to-total fraction
    :return: probability, maximum <= 1./numpy.sqrt(2*numpy.pi)/sig ~ 0.4/sig
    """
    sig, mean = 0.2, 0.
    return 1./numpy.sqrt(2*numpy.pi)/sig*numpy.exp(-(fraction[0]-mean)**2/2/sig/sig)

def ellip_bulge(num, seed=123400): # old-version
    """
    Generate random ellipticity for a given number
    following the distribution of the ellipticity
    of bulge-dominated galaxies

    See Miller et al, 2013, MNRAS

    The ellipticity is defined as: e = (a - b)/(a + b)exp[2i\theta]
    as what in Miller et al. 2013 MNRAS and CFHTLenS.
    And it has been convert to the definition used in Galsim
    e = (a^2 - b^2)/(a^2 + b^2)exp[2i\theta]
    """
    numpy.random.RandomState(seed)
    b, c = 2.368, 6.691
    # probability
    pe = lambda e: 27.7478 * e * numpy.exp(-b * e - c * e * e)

    emin, emax = 0.0, 0.803
    pmin, pmax = 0.0, 2.64456

    es = numpy.linspace(emin, emax, int((emax - emin) / 0.000001) + 1)
    pe_base = pe(es)
    # normalize
    pe_base = pe_base / numpy.sum(pe_base)
    rbe = numpy.random.choice(es, num, p=pe_base)
    # convert to the definition used in Galsim:
    # e = (a^2 - b^2)/(a^2 + b^2)exp[2i\theta]
    theta = numpy.random.uniform(0, numpy.pi, num)
    # q = (1 - rbe) / (1 + rbe)
    # es = (1 - q ** 2) / (1 + q ** 2)
    e1 = rbe * numpy.cos(2 * theta)
    e2 = rbe * numpy.sin(2 * theta)
    return e1, e2, rbe#, es

def CFHT_skysig(zpt=26.22, exp_time=600, pix_scale=0.187, sky_bright=20.3): # developing
    """
    the result seems to be wrong, the noise variance from fitting is about 60.
    the default values come from the CFHT MegaPrime/MegaCam telescope
    and the formula comes from the ETC of lsst
    :param zpt: zero point
    :param exp_time: exposure time, default = 300 sec
    :param pix_scale: arcsec/pixel
    :param sky_bright:
    :return: the sky background noise variance
    """
    return numpy.sqrt(zpt * 10**(-0.4*(sky_bright-24))*exp_time)*pix_scale


################################################################
# the methods for data analysis
################################################################
def find_block(scale, radius_s, radius_e, ny, nx, pts_y, pts_x, block_ny, block_nx, block_boundy, block_boundx):
    """
    the coordinate origin of the points should be set to the upper left corner of the big block.
    |(0,0),..., (0,x) ...|
    |....................|
    |(y,0),..............|
    for the correlation calculation
    find the target blocks with a distance which is between two given radius for the specific point
    :param scale: the side length of the square blocks
    :param radius_s, radius_e: the lower bound of radius, the upper bound of radius
    :param ny, nx: the block tag of the point
    :param pts_y, pts_x: y, x of the point
    :param block_ny, block_nx: number of the blocks along y-axis and x-axis for saving time
    :param block_boundy, block_boundx: (n, 4) numpy array, each row contains the y-coordinates (x-coordinates)
                        of the four point of each block
                        |y1, y2|    |x1, x2|
                        |y4, y3|,   |x4, x3|, actually, y1 = y2, y4 = y3, x1 = x4, x2 = x3
    :return: list of the target blocks
    """
    rs, re = radius_s ** 2, radius_e ** 2
    nx_left = int((radius_e - pts_x) / scale + nx) + 1
    nx_right = int((radius_e + pts_x) / scale - nx)
    ny_up = int((radius_e + pts_y) / scale - ny)
    nx_s, nx_e = max(nx - nx_left, 0), min(nx + nx_right + 1, block_nx)
    ny_e = min(ny + ny_up + 1, block_ny)

    needs = [iy * block_nx + ix for iy in range(ny, ny_e) for ix in range(nx_s, nx_e)]
    nxy = [(iy, ix) for iy in range(ny, ny_e) for ix in range(nx_s, nx_e)]
    dr_n = (block_boundy[needs] - pts_y) ** 2 + (block_boundx[needs] - pts_x) ** 2
    dr_n.sort()

    blocks_found = []
    for tag, xy in enumerate(nxy):
        iy, ix = nxy[tag]
        if dr_n[tag][3] < rs or (dr_n[tag][0] > re and iy != ny and ix != nx):
            continue
        else:
            if iy > ny or (iy == ny and ix >= nx):
                blocks_found.append((iy, ix))
    return blocks_found


def mcplot(x1_data, y1_data, x2_data, y2_data, e1mc, e2mc, cut_start, cut_end, xylim, path=None,show=False):
    # "x_data' is the 'x'
    # 'y_data' is an (3,n) array "[[y's],[dy's],[num's]]
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(121)
    ax.errorbar(x1_data, y1_data[0], y1_data[1], ecolor='black', elinewidth=1, fmt='none', capsize=2)
    ax.plot(x1_data, e1mc[0] * x1_data + e1mc[2], color='red')
    ax.plot([-0.1, 0.1], [-0.1, 0.1], label='y=x', color='blue')
    ax.scatter(x1_data, y1_data[0], c='black')

    for j in range(len(x1_data)):
        ax.text(x1_data[j], y1_data[0,j], str(round(y1_data[2,j] / 1000, 1)) + "K", color="red")
    ax.text(0.1, 0.85, 'm=' + str(round(e1mc[0] - 1, 6)) + '$\pm$' + str(round(e1mc[1], 6)), color='green', ha='left',
            va='center', transform=ax.transAxes, fontsize=20)
    ax.text(0.1, 0.8, 'c=' + str(round(e1mc[2], 6)) + '$\pm$' + str(round(e1mc[3], 6)), color='green', ha='left',
            va='center', transform=ax.transAxes, fontsize=20)
    ax.text(0.1, 0.75, "[ " + cut_start + ", " + cut_end + "]", color='green', ha='left', va='center', transform=ax.transAxes,
            fontsize=20)
    plt.xlabel('True  g1', fontsize=20)
    plt.ylabel('Est  g1', fontsize=20)
    plt.legend(fontsize=15)
    plt.ylim(xylim[0], xylim[1])
    plt.xlim(xylim[0], xylim[1])

    # plot g2 line
    ax = fig.add_subplot(122)
    ax.errorbar(x2_data, y2_data[0], y2_data[1], ecolor='black', elinewidth=1, fmt='none', capsize=2)
    ax.plot(x2_data, e2mc[0] * x2_data + e2mc[2], color='red')
    ax.plot([-0.1, 0.1], [-0.1, 0.1], label='y=x', color='blue')
    ax.scatter(x2_data, y2_data[0], c='black')
    for j in range(len(x2_data)):
        ax.text(x2_data[j], y2_data[0, j], str(round(y2_data[2, j] / 1000, 1)) + "K", color="red")
    ax.text(0.1, 0.85, 'm=' + str(round(e2mc[0] - 1, 6)) + '$\pm$' + str(round(e2mc[1], 6)), color='green', ha='left',
            va='center', transform=ax.transAxes, fontsize=20)
    ax.text(0.1, 0.8, 'c=' + str(round(e2mc[2], 6)) + '$\pm$' + str(round(e2mc[3], 6)), color='green', ha='left',
            va='center', transform=ax.transAxes, fontsize=20)
    ax.text(0.1, 0.75, "[ " + cut_start + ", " + cut_end + "]", color='green', ha='left', va='center', transform=ax.transAxes,
            fontsize=20)
    plt.xlabel('True  g2', fontsize=20)
    plt.ylabel('Est  g2', fontsize=20)
    plt.legend(fontsize=15)
    plt.ylim(xylim[2], xylim[3])
    plt.xlim(xylim[2], xylim[3])

    if path is not None:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()

def mc_compare(x, mc1_list, mc2_list, labels, cap=4, ms=20, linewidth=2, margin=0.1,
               pic_path=None, multi_fig=True, size=(10,10),show=False):
    r"""
    it is designed to show the variation of m's and c's with respect to the cutoff
    it can show the figures one by one or just a big one contains four
    'pic_path' must be a list contains the directories that each figure will be stored
    'x' must be a 1-D numpy array
    'mc1/2_list' must be a list of (n,4) numpy array "m,dm,c,dc"
    'labels' is the label of each mc array
    :param x:
    :param mc1_list:
    :param mc2_list:
    :param labels:
    :param cap:
    :param ms:
    :param linewidth:
    :param margin:
    :param pic_path:
    :param multi_fig:
    :param size:
    :param show:
    :return:
    """
    num = len(mc1_list)
    if num > 5:
        print("Two many lines")
        exit()
    colors = ["red", 'limegreen', "blue", "darkorange", "purple",'dodgerblue']
    mc_label = ["_$m_1$", "_$c_1$", "_$m_2$", "_$c_2$"]
    ylabels = ["multiplicative bias $m_1$", "additive bias $c_1$",
               "multiplicative bias $m_2$", "additive bias $c_2$"]
    x1 = x.min() - (x.max() - x.min())*margin
    x2 = x.max() + (x.max() - x.min())*margin
    if multi_fig:
        fig = plt.figure(figsize=size)
    for i in range(4):
        if multi_fig:
            ax = fig.add_subplot(221 + i)
        else:
            fig = plt.figure(figsize=size)
            ax = fig.add_subplot(111)
        a, b = divmod(i, 2)
        if i < 2:
            mc = mc1_list
        else:
            mc = mc2_list
        for j in range(num):
            lb = labels[j] + mc_label[i]
            ax.errorbar(x+j*0.05, mc[j][:, 2*b]-1, mc[j][:, 2*b+1], ecolor=colors[j],
                        linewidth=linewidth,color=colors[j], capsize=cap, label=lb)
            ax.scatter(x+j*0.05, mc[j][:, 2*b]-1, c=colors[j], s=ms)
            if j == 0:
                ax.plot([x1, x2], [0, 0], c='k')
            ax.set_xlim(x1, x2)
            ax.set_xlabel("cutoff")
            ax.set_ylabel(ylabels[i], fontsize=12)
            plt.legend(ncol=2)
        if not multi_fig and pic_path:
            plt.savefig(pic_path[i])
        if not multi_fig and show:
            plt.show()
    if multi_fig:
        if pic_path:
            plt.savefig(pic_path[0])
        if show:
            plt.show()

def check_in(interval):
    if interval[0] <= 0 <= interval[1]:
        return True
    else:
        return False

def set_bin(data, bin_num):
    temp_data = numpy.sort(data[data>0])[:int(len(data[data>0])*0.99)]
    bin_size = len(temp_data)/bin_num*2
    bins = numpy.array([temp_data[int(i*bin_size)] for i in range(1, int(bin_num / 2))])
    bins = numpy.sort(numpy.append(numpy.append(-bins, [0.]), bins))
    bound = numpy.max(numpy.abs(data)) * 10000.
    bins = numpy.append(-bound, numpy.append(bins, bound))
    return bins

def back_to_block(data, num, cols, size_y, size_x, yi, xi, area_i, distance_thresh):
    """
    for the analysis of the data measured by SExtractor
    the sequence of measured parameters are reserved.
    :param data: (m, n) numpy array
    :param num: the total number of the true blocks
    :param cols: the column number of the true blocks
    :param size_y, size_x: size of block
    :param yi, xi: the column number of y and x
    :param area_i: the column number of area
    :param distance_thresh: the maximum of the peak be away from the center of the image
    :return: (num, data.shape[1]) numpy array
    """

    # for the case that only one source detected, the data will be a (para_num, ) numpy array
    if len(data.shape) != 2:
        data.shape = (1, len(data))
    # the coordinates in SExtractor start from 1 not 0!
    sex_ori = 1
    block_arr = numpy.zeros((num, data.shape[1]))
    my, dy = numpy.divmod(data[:, yi] - sex_ori, size_y)
    mx, dx = numpy.divmod(data[:, xi] - sex_ori, size_x)
    block_n = (my*cols + mx).astype(int)
    away = numpy.sqrt((dy-size_y/2)**2 + (dx-size_x/2)**2)
    for i in range(data.shape[0]):
        block_id = block_n[i]
        # print(block_id)
        if data[i, area_i] >= block_arr[block_id, area_i] and away[i] <= distance_thresh:
            block_arr[block_id] = data[i]
    block_arr[:, yi] -= sex_ori
    block_arr[:, xi] -= sex_ori
    return block_arr


def field_dict(file_list_path):
    # to build a dictionary that contains the exposures as {"field": {"exposure":[expo1, expo2..]}....}
    with open(file_list_path, "r") as f:
        contents = f.readlines()
    file_dict = {}
    fields = [] # to avoid the disorder of the field sequence
    for c in contents:
        c = c.split("\n")[0]
        if "w" in c:
            field = c
            file_dict.setdefault(field, {})
            fields.append(field)
        else:
            chip = c.split(".fits")[0]
            expo = c.split("_")[0]
            if expo in file_dict[field].keys():
                file_dict[field][expo].append(chip)
            else:
                file_dict[field].setdefault(expo, [chip])
    return file_dict, fields

def cfht_label(field_name):
    # the location of each galaxy is labeled by the field_label and exposure_label
    # counting from the left, the first, third and fifth figure denotes "w_m(p)_(m)p_"
    # the second and the fourth denotes "m" or "p" (1=m,0=p)
    # the last two figure is zero and will denote the chip NO.
    # the exposure label will be stored in the other place
    mp1, mp2 = 0, 0
    if field_name[2] == "m":
        mp1 = 10 ** 3
    if field_name[4] == "m":
        mp2 = 10
    return int(field_name[1])*10**4 + int(field_name[3])*10**2 + int(field_name[5]) + mp1 + mp2


def ellip_plot(ellip, coordi, lent, width, title, mode=1,path=None,show=True):
    e1 = ellip[:, 0]
    e2 = ellip[:, 1]
    e = numpy.sqrt(e1 ** 2 + e2 ** 2)
    scale = numpy.mean(1 / e)
    x = coordi[:, 0]
    y = coordi[:, 1]
    if mode== 1:
        dx = lent * e1 / e / 2
        dy = lent * e2 / e / 2
    else:
        dx = scale * lent * e1 / e / 2
        dy = scale * lent * e2 / e / 2
    x1 = x + dx
    x2 = x - dx
    y1 = y + dy
    y2 = y - dy

    norm = plt.Normalize(vmin=numpy.min(e), vmax=numpy.max(e))
    cmap = plt.get_cmap('YlOrRd')
    fig = plt.figure(figsize=(20,10))
    plt.axes().set_aspect('equal', 'datalim')
    for i in range(len(x)):
        cl = cmap(norm(e[i]))
        plt.plot([y1[i], y2[i]], [x1[i], x2[i]], color=cl, linewidth=width)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    plt.colorbar(sm)
    plt.title(title,fontsize = 18)
    if path is not None:
        plt.savefig(path)
    if show is True:
        plt.show()


################################################################
# the general methods
################################################################
def allot(allot_list, fractions):
    """
    allot the "target mission list" to CPUs
    :param allot_list: list of any thing
    :param fractions: INT, the number of sub-list
    :return: list of some sub-list
    """
    num = len(allot_list)
    m, n = divmod(num, fractions)
    pool = []
    for i in range(fractions):
        temp = []
        for j in range(m*i, m*(i+1)):
            temp.append(allot_list[j])
        if n > 0:
            temp.append(allot_list[m*fractions+n-1])
            n -= 1
        pool.append(temp)
    return pool

def file_name(path):
    """
    check the file, if it exists, the new file will be rename with a number suffix
    :param path: absolute path to the file, include the filename
    :return: absolute path of file
    """
    ex = os.path.exists(path)
    ori_name = os.path.basename(path)
    ab_path = path.split(ori_name)[0]
    i = 1
    name = ori_name.split(".")
    n = len(name)
    if n == 1:
        a = name[0]
        b = ""
    else:
        a = ".".join(name[:n-1])
        b = ".".join(("",name[-1]))
    while ex:
        path = "".join((ab_path, "".join((a, "_%s"%str(i), b))))
        ex = os.path.exists(path)
        i += 1
    return path


def get_logger(log_path, log_name="RECORD", mode="a"):
    """
    It is the same as the method "write_log()"
    Call it before any loops!!!
    If more than one log files will be created,
    then different log_name should be provided for each log file!!!
    return the logger object for write logs,
    !!! then call logger.info("...") anywhere in the program!!!

    :param log_path: absolute path to log file
    :param mode: "w": truncate the file and write, "a": append new messages
    :return: the logger object
    """
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    lf = logging.FileHandler(log_path, mode)
    form = logging.Formatter('%(asctime)s -- %(message)s')
    lf.setFormatter(form)
    logger.addHandler(lf)
    return logger

def write_log(log_path, content, mode="a"):
    """
    write log
    :param log_path: absolute path to log file
    :param content: to write into the log file
    :param way: write with logger or directly
    :param mode: "w": truncate the file and write, "a": append new messages
    :return: no return
    """
    content = time.strftime("%Y-%m-%d %H:%M:%S -- ") + content
    if content[-1] != "\n":
        content += "\n"
    with open(log_path, mode) as f:
        f.write(content)


def config(path, cmd, contents, write=False):
    """
    operation to the configuration file
    :param path: string, directory to file
    :param cmd: a list of operations , "add, get, sect_del, opt_del"
                can be a mixture of "add, get, sect_del, opt_del"
                if the config file doesn't exist, it will be ignored
                "add" : add a section or an option in a section if the
                        option exists, it will be raplaced by new value
                "get" : get the value of a option in a section
                "sect_del" : delete the section in the configure file
                "opt_del" : delete the option in a section
    :param contents: [["section","para","value"],[...],...]
    :return: list of the values of the target options if "get" is in cmd, else a empty list
    """
    cobj = configparser.ConfigParser()

    # if the config file doesn't exist
    if not os.path.exists(path):
        for ii, subcon in enumerate(contents):
            if len(subcon) == 3:
                sect, para, value = subcon
                if not cobj.has_section(sect):
                    cobj.add_section(sect)
                cobj.set(sect, para, value)
            else:
                raise ValueError("Each sublist in contents must have 3 components!")

        with open(path,"w") as confile:
            cobj.write(confile)
        return None
    else:
        cobj.read(path)
        opt_vals = []
        if len(cmd) != len(contents):
            raise ValueError("The lengths of cmd and contents don't match!")
        else:
            for ii in range(len(contents)):
                subcon = contents[ii]
                if len(subcon) == 3:
                    sect, para, value = subcon
                    op = cmd[ii]
                    if op == "add":
                        if not cobj.has_section(sect):
                            cobj.add_section(sect)
                        # if the option exists, the value will be replaced by new one!
                        cobj.set(sect, para, value)

                    elif op == "get":
                        if cobj.has_section(sect):
                            if cobj.has_option(sect,para):
                                opt_vals.append(cobj.get(sect,para))
                                # print(cobj.get(sect,para))
                            else:
                                raise ValueError("The %s doesn't exist!" % sect)
                        else:
                            raise ValueError("The section doesn't exist!")

                    elif op == "sect_del":
                        if cobj.has_section(sect):
                            cobj.remove_section(sect)
                        else:
                            raise ValueError("The section doesn't exist!")

                    elif op == "opt_del":
                        if cobj.has_option(sect, para):
                            cobj.remove_option(sect,para)
                        else:
                            raise ValueError("The %s in %s doesn't exist!"%(para,sect))

                    else:
                        raise ValueError("The 'cmd' must be one of [add, get, sect_del, opt_del] in %s doesn't exist!")

                else:
                    raise ValueError("Each sublist in contents must have 3 components!")
            if write:
                with open(path, "w") as confile:
                    cobj.write(confile)
            return opt_vals
