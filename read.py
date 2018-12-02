# import numpy as np
# import pylab as plt
import matplotlib.pyplot as plt
import numpy as  np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# from scipy import optimize
# from scipy import asarray as ar, exp
# from scipy.optimize import curve_fit
# from tool_box import *
# from F_Q_P_S import *
# obj = F_Q_P_S(48,123)
# e_g = np.zeros((2,10))
#
# gal_num = 100000
#
# data = np.zeros((10,4,7*gal_num))
#
#     #change every sample
#
# for k in range(10):
#     a0 = np.load("/home/shenzhi/data4/3.00_%d.npz"%k)
#     b0= a0['arr_0']
#     c=b0
#     for i in range(6):
#         psf = [3.2,3.41,3.59,3.8,4.,4.2]
#         a = np.load("/home/shenzhi/data4/%.2f_%d.npz"%( psf[i],k))
#         b=a['arr_0']
#         c =np.concatenate([c,b],1)
#     data[k]=c
#     # print c.shape
# # c= np.concatenate([b0,b1,b2,b3,b4,b5],1)
#     e_g[0,k ], e_g[1,k] = obj.fmin_g(c[0], c[2], c[3], 1, 4)
# # print e_g
# np.savez("/home/shenzhi/data4/res.npz",data)
#
# x = np.linspace(-0.02,0.02,10)
# a,b,c,d=data_fit(x,e_g[0],e_g[1])
# print 1-a,b,c,d
# for i in range(10):
#     g_pdf ,g_ave = obj.bootstr(data[i],300)
#     # np.save("g_pdf_boot.npz",g_pdf)
#     # np.savez("g_ave_boot.npz",g_ave)
#     # num1 = np.histogram(g_pdf[0],15)
#     # num2 = np.histogram(g_ave[0],15)
#     a = np.std(g_pdf)
#     b = np.std(g_ave)
#
#     print ("g_pdf std:",a)
#     print ("g_ave_std:",b)

import numpy as np

from sys import path

path.append('/home/hklee/work/fourier_quad/')

from F_Q_P_S import F_Q_P_S

import tool_box

from mpi4py import MPI

import time

# from sys import argv
#
# psf_scale = float(argv[1])

start = time.clock()

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

cpus = comm.Get_size()


est_g1 = np.zeros((10,200))
gal_num = 10000
for i in range(100):
    a = np.load('/home/shenzhi/data/tatal_1.6.%d_%d.npz'%(i,gal_num))['arr_0']
    est_g1[:,i] = a[0,:]

# x = g1
# y = est_g[0]
# y_err = est_g[1]
# y2 = est_g[2]
# y2_err = est_g[3]
# m,m_sig,c,c_sig = data_fit(x, y, y_err)
# m1,m1_sig,c1,c1_sig =data_fit(x,y2,y2_err)
# msig = np.mean(est_g[1])
# nsig =  np.mean(est_g[3])
#
# plt.plot(x, x, 'k')
# plt.errorbar(x, y, y_err, label='PDF_SYM', fmt = ',',color='r', ecolor='r',capsize=5)
# plt.errorbar(x, y2, y2_err, label='AVE_PDF', fmt = ',',color='b', capsize=3)
# plt.text(-0.028, .020, r'$m={%.5f}({%.7f})$' % (m, m_sig))
# plt.text(-0.028, .016, r'$c={%.6f}({%.7f})$' % (c, c_sig))
# plt.text(-0.028, .012, r'$m={%.5f}({%.5f})$' % (m1, m1_sig))
# plt.text(-0.028, .008, r'$c={%.6f}({%.7f})$' %(c1, c1_sig))
# plt.text(-0.028, .004, r'errorbar=${%.6f},{%.7f}$' %(msig,nsig))
#
#
# plt.xticks(np.arange(-0.04, 0.04, 0.01))
# plt.title('Est_g1')
# plt.legend()
# plt.savefig('/home/shenzhi/data3/%d_sn%d.png'%(gal_num,flux),dpi=600)
#
# np.savez('/home/shenzhi/data3/est_g1_0.npz',est_g1)



#same point in different noise and psf scale

# if rank ==0:
#     print b.shape
# t_g[0], t_g[1] = obj.fmin_g(b[0], b[2], b[3], 1, 8)
# t_g[2] = np.mean(b[0]) / np.mean(b[2])
# t_g[3] = np.sqrt(np.mean(np.mean(b[0] ** 2)) / np.mean(b[2]) ** 2 / len(b[0]))
#
# t_g[4], t_g[5] = obj.fmin_g(b[1], b[2], b[3], 1, 8)
# t_g[6] = np.mean(b[1]) / np.mean(b[2])
# t_g[7] = np.sqrt(np.mean(np.mean(b[1] ** 2)) / np.mean(b[2]) ** 2 / len(b[1]))

# t_g = np.reshape(t_g,(1,8))

# different point in same noise and psf scale
# for i in range(6):
#     res2[i] = np.concatenate((res[0+i],res[6+i],res[12+i],res[18+i],res[24+i]),1)
#
#     b = res2[i]
#
#     t_g2[0,i], t_g2[1,i] = obj.fmin_g(b[0], b[2], b[3], 1, 8)
#     t_g2[2,i] = np.mean(b[0]) / np.mean(b[2])
#     t_g2[3,i] = np.sqrt(np.mean(np.mean(b[0] ** 2)) / np.mean(b[2]) ** 2 / len(b[0]))
#
#     t_g2[4,i], t_g2[5,i] = obj.fmin_g(b[1], b[2], b[3], 1, 8)
#     t_g2[6,i] = np.mean(b[1]) / np.mean(b[2])
#     t_g2[7,i] = np.sqrt(np.mean(np.mean(b[1] ** 2)) / np.mean(b[2]) ** 2 / len(b[1]))

# t_g = np.reshape(t_g,(1,40))
#
# if rank > 0:
#
#     comm.Send(t_g, dest=0, tag=rank)
#
# else:
#
#     for i in range(1, cpus):
#         recv = np.empty((8,1), dtype=np.float64)
#
#         comm.Recv(recv, source=i, tag=i)
#
#         t_g = np.column_stack((t_g, recv))
#
#
# if rank == 0:
#     print(t_g)
#     np.savez("/home/shenzhi/data6/t_g.npz",t_g)
#
#     # np.savez("/home/shenzhi/data6/pdf_ave_g.npz",est_g)
#     y = t_g[0]
#     y1 = t_g[2]
#     y_err = t_g[1]
#     y_err1=t_g[3]
#     a, b, c, d = tool_box.data_fit(g1, y, y_err)
#     a1, b1, c1, d1 = tool_box.data_fit(g1, y1, y_err1)
#
#     print("m: %.4f(%.4f), c: %.6f(%.6f)" % (1 - a, b, c, d))
#     print("m: %.4f(%.4f), c: %.6f(%.6f)" % (1 - a1, b1, c1, d1))

end = time.clock()
print("Whole time = %.2f" % (end - start))
#
# m=(np.array([ 2, 14, 19, 13, 62, 38, 38, 33, 63, 12,  2,  3,  0,  0,  1]),
#  np.array([-0.0206543 , -0.02052734, -0.02040039, -0.02027344, -0.02014648,
#        -0.02001953, -0.01989258, -0.01976563, -0.01963867, -0.01951172,
#        -0.01938477, -0.01925781, -0.01913086, -0.01900391, -0.01887695,
#        -0.01875   ]))
# y = np.array([ 3,  9,  4, 14, 18, 38, 30, 48, 41, 29, 28, 19, 13,  3,  3])
# x = np.array([-0.02053407, -0.02043146, -0.02032886, -0.02022625, -0.02012365,
#        -0.02002104, -0.01991844, -0.01981583, -0.01971323, -0.01961062,
#        -0.01950802, -0.01940541, -0.01930281, -0.0192002 , -0.0190976 ])
# def func(x, a, b, c):
#     return a * np.exp((x-b)**2/c**2)
# b0=np.mean(y)
# c0 = sum(y-b0)**2/len(y)
# me,sig = curve_fit(func,x,y,[1,b0,c0])
# print me,sig
#
# plt.plot(x,y,'b+:',label='data')
# plt.plot(x,func(x,*me),'ro:',label='fit')
# plt.legend()
# plt.show()

# a0 = np.load("/home/shenzhi/data/3_0.npz")
# a1 = np.load("/home/shenzhi/data/3.2_0.npz")
# a2 = np.load("/home/shenzhi/data/3.41_0.npz")
# a3 = np.load("/home/shenzhi/data/3.59_0.npz")
# a4 = np.load("/home/shenzhi/data/3.800000_0.npz")
# a5 = np.load("/home/shenzhi/data/4_0.npz")
# b0=a0['arr_0']
# b1=a1['arr_0']
# b2=a2['arr_0']
# b3=a3['arr_0']
# b4=a4['arr_0']
# b5=a5['arr_0']