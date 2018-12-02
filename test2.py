import os
from sys import path
path.append("E:/Github/astrophy-research/my_lib")
import time
import numpy
import tool_box
from Fourier_Quad import Fourier_Quad
import matplotlib.pyplot as plt
obj = Fourier_Quad(48,101)

size = 52
siz = 48
m = 48
psf_scale = 4
flux = 100
num_po,radius = 50,5
g1 ,g2 = 0.01,0.01
she_pos = obj.ran_pos(num_po, radius, (g1, g2))[1]
gal = obj.draw_noise(0,1)+obj.convolve_psf(she_pos,psf_scale,flux)

gal_p = obj.pow_spec(gal)       #48*48

a = gal_p
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


gal_log = numpy.log10(data)     #we need 52*52,now it's 52*52

gal_log_fit = tool_box.smooth(gal_log, size)
gal_fit = 10**gal_log_fit
alpha = numpy.zeros((m*m,6,6))
new = numpy.zeros_like(gal_log)
for row in range(2, size-2):
    for col in range(2, size-2):
        tag = 0
        x = []
        y = []
        z = []
        pk = 0
        for i in range(-2, 3):
            for j in range(-2, 3):
                if tag not in [0, 4, 20, 24]:
                    if (i+row)*size + j+col != (size+1)*size/2:
                        x.append(j)
                        y.append(i)
                        z.append(gal_log[i+row, j+col])
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
        fx = numpy.sum(z*x)
        fy = numpy.sum(z*y)
        fx2 = numpy.sum(z*x*x)
        fxy = numpy.sum(z*x*y)
        fy2 = numpy.sum(z*y*y)

        n = len(x)
        x1 = x.sum()
        x2 = numpy.sum(x*x)
        x3 = numpy.sum(x*x*x)
        x4 = numpy.sum(x*x*x*x)

        y1 = y.sum()
        y2 = numpy.sum(y*y)
        y3 = numpy.sum(y*y*y)
        y4 = numpy.sum(y*y*y*y)

        xy = numpy.sum(x*y)
        x2y = numpy.sum(x*x*y)
        xy2 = numpy.sum(x*y*y)
        x3y = numpy.sum(x*x*x*y)
        xy3 = numpy.sum(x*y*y*y)
        x2y2 = numpy.sum(x*x*y*y)

        cov = numpy.array([[n, x1, y1, x2, xy, y2],
                          [x1, x2, xy, x3, x2y, xy2],
                          [y1, xy, y2, x2y, xy2, y3],
                          [x2, x3, x2y, x4, x3y, x2y2],
                          [xy, x2y, xy2, x3y, x2y2, xy3],
                          [y2, xy2, y3, x2y2, xy3, y4]])
        inv_cov = numpy.linalg.inv(cov)
        alpha[(row-2)*48+col-2] = inv_cov
        f_z = numpy.array([[f],[fx],[fy],[fx2],[fxy],[fy2]])
        para = numpy.dot(inv_cov, f_z)
        new[row, col] = para[0,0]
        # print(para, gal_log[row, col])
numpy.savez('alpha.npz',alpha)
print alpha[48*47]
plt.subplot(221)
plt.imshow(gal_log[2:size-2, 2:size-2])

plt.subplot(222)
plt.imshow(gal_log_fit[2:size-2, 2:size-2])
plt.subplot(223)
plt.imshow(new[2:size-2, 2:size-2])
plt.subplot(224)
plt.imshow(new[2:size-2, 2:size-2] - gal_log_fit[2:size-2, 2:size-2])
plt.colorbar()
plt.show()