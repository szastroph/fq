# encoding: utf-8
from tool_box import *
import numpy as np
from  tool_box import *
import matplotlib.pyplot as plt
# a = np.load('t_g_SN100.npz')['arr_0']
# x = np.linspace(-0.005,0.004,10)
# y = a[0]
# y_arr = a[1]
# print (data_fit(x,y,y_arr))

a = np.zeros(20)
print a[19]
print a.shape
# from scipy import optimize
# from scipy import asarray as ar, exp
# from scipy.optimize import curve_fit
# import lsstetc
# # from tool_box import *
# # from mpi4py import MPI
# import os.path
# import h5py
# # m: -0.0418(0.0545), c: 0.000480(0.000318)
# # m: 0.0235(0.1259), c: -0.000014(0.000729)
# import csv
# #
# # comm = MPI.COMM_WORLD
# # rank = comm.Get_rank()
# # cpus = comm.Get_size()
# from F_Q_P_S import F_Q_P_S
# obj = F_Q_P_S(54,13234354)
# #get result of std_g1-g2
# # g1 = [-0.02,-0.012,-0.004,0.004]
# # g1_std = np.zeros((2,4))
# # g2_std = np.zeros((2,4))
# # g1_rea = np.zeros((2,4))
# # g2_rea = np.zeros((2,4))
# # g = np.zeros((8,4))
# # for i in range(4):
# #     a = np.load('t_g_%.3f.npz'%g1[i])['arr_0']
# #     g[0,i] = np.std(a[0])
# #     g[1,i] = np.std(a[2])
# #     g[2,i] = np.std(a[6])
# #     g[3,i] = np.std(a[8])
# #     g[4,i] = sum(a[4]**2)/10
# #     g[5,i] = sum(a[5]**2)/10
# #     g[6,i] = sum((a[6]-g1[i])**2)/10
# #     g[7,i] = sum((a[8]-g1[i])**2)/10
# # print g
# # np.savetxt('g_std.txt',g)
# x = np.linspace(0,2*np.pi,300)
# y = np.sin(x)**2
# plt.plot(x,y)
# plt.show()
#
# # print c
# # zf =str('±')
# # print c.shape
# # # print c
# # for row in range(10):
# #     for column in range(4):
# #         m = c[row,2*column]
# #         n = c[row,2*column+1]
# #
# #         num =str( '%.6f%s%.6f'%(m,zf,n))
# #         print ('%s'%num)
#
#
#
#
#
#
# # for i in range(3):
# #     c = np.loadtxt('%.3f_PDF_PDF-AVE.txt'%g1[i])
# #     print c.shape
# #     for j in range(10):
# #         print ''
#
# # a = '0.00409'+'±'+'0.000987'
# # print a
# # flux = np.load('flux.npz')['arr_0']
# # a,b = np.histogram(flux,15)
# # lobin = np.log10(a)
# # bin =  np.delete(b,-1)
# # flbin = lobin/sum(lobin)
# # k = np.polyfit(bin,flbin,1)
# # # plt.plot(bin,flbin)
# # # plt.show()
# # print np.poly1d(k)
# # est_g = np.load('t_g.npz')['arr_0']
# # a = np.array([0.00125,-0.001383,-1.2E-05,-0.000594,0.000484,0,-0.001047,-0.000578,0.000305,1.6E-05])
# # b=np.array([0.000338,-0.000515,0.000815,-0.000109,0.000579,-0.001031,-0.001928,-0.000756,-0.000258,-0.000936])
# # print np.sum(a**2),a.shape
# # print np.sum(b**2),b.shape
# # est_g = est_g.T
# # np.savetxt('10w_PDFAVE.txt',est_g,fmt ='%10.6f')
# # a = np.loadtxt('10w_PDFAVE.txt')
# # for i in range(10):
#
#
#
# # print est_g
# # g1 = np.linspace(-0.02,0.016,10)
# # print est_g.shape
# # print est_g[4] < est_g[5]
# # print sum(est_g[4] ** 2) < sum(est_g[5] ** 2)
# # print np.var(est_g[0] - g1) < np.var(est_g[2] - g1)
# # -4.866e-06 x + 4.878
# # -1.439e-07 x + 0.1443
# # def flux_mock(num, flux_min, flux_max):#flux_min = 150,flux_max = 1.2e+06
# #     m = np.linspace(flux_min, flux_max, 1000000)
# #     pm = 10**(-4.189e-5 *m )
# #     # pm = 10 ** (-1.439e-07*m + 0.1443)
# #     pm = pm/np.sum(pm)
# #     new_pdf = np.random.choice(m, num, p=pm)
# #     return new_pdf
# #
# # a = flux_mock(1000,170,1000000)
# # a = np.sort(a)
# # b = np.linspace(170,1e06,1000)
# # plt.plot(b,a)
# # plt.show()
# # print np.histogram(a,10)
#
# # data_path ='D:/Python1/pydocument/seniorproject_quenching2/practice/data/M16'
# # fname = os.path.join(data_path,'hmf.dat')
# # lnMh_arr, dndlnMh_arr = np.genfromtxt(fname, unpack=True)
# # b = a[:,2:6]
# #
# # res =b.T
# # # G1 = f[2]
# # # N = f[4]
# # # U = f[5]
# # f.close()
# # est_g[0], est_g[1] = obj.fmin_g(res[0], res[2], res[3], 1, 8)
# # est_g[2] =np.mean(res[0])/np.mean(res[2])
# # est_g[3] =np.sqrt(np.mean(np.mean(res[0]**2))/np.mean(res[2])**2/len(res[0]))
# # print est_g
#
#
# # ['id\t                  pos\t          PSF_e1\t                             PSF_e2\t            e1\t     e2\tweight\tfitclass\tZ_B\tm\tc2\tLP_Mi\tMAG_i']
# # ['W2m0m0_7981\tPosition ICRS 133.9514396000002 -4.72322487300001\t0.0\t0.0\t0.0\t0.0\t0.0\t             -2\t0.45\t-0.233851\t5.97624\t-18.25\t24.5079']
# # [[-0.01523437 -0.009375   -0.00664063 -0.00546875 -0.009375   -0.01132813
# #   -0.00839844 -0.00410156 -0.015625   -0.00898438]
# #  [ 0.00106349  0.00108811  0.00109564  0.00108707  0.00107612  0.0010791
# #    0.00112894  0.00107211  0.00108515  0.00107799]
# #  [-0.01508597 -0.00986552 -0.0065053  -0.00581469 -0.01063918 -0.01099969
# #   -0.00740022 -0.00379034 -0.01677846 -0.00846295]
# #  [ 0.00105135  0.00105641  0.00105113  0.00105736  0.0010494   0.00106297
# #    0.00105928  0.00104613  0.00105519  0.00106333]
# #  [ 0.00723437  0.001375    0.00135937  0.00253125  0.001375    0.00332813
# #    0.00039844  0.00389844  0.007625    0.00098438]
# #  [ 0.00708597  0.00186552  0.0014947   0.00218531  0.00263918  0.00299969
# #    0.00059978  0.00420966  0.00877846  0.00046295]]
# # [False  True  True False  True False  True  True  True False]
# # True
# # True
# # [[-0.01054688 -0.01230469 -0.0125     -0.00703125 -0.0078125  -0.00859375
# #   -0.00625    -0.01171875 -0.00703125 -0.0078125 ]
# #  [ 0.00255115  0.00245595  0.00239346  0.0025186   0.00243683  0.00244788
# #    0.00254024  0.00251199  0.00244321  0.0025257 ]
# #  [-0.01071897 -0.01143806 -0.01300275 -0.00586846 -0.00826216 -0.00726589
# #   -0.00877552 -0.01126268 -0.00547042 -0.00717752]
# #  [ 0.00238359  0.00239528  0.00241112  0.00239844  0.00241138  0.00239946
# #    0.00241057  0.00240657  0.00238635  0.00240741]
# #  [ 0.00254688  0.00430469  0.0045      0.00096875  0.0001875   0.00059375
# #    0.00175     0.00371875  0.00096875  0.0001875 ]
# #  [ 0.00271897  0.00343806  0.00500275  0.00213154  0.00026216  0.00073411
# #    0.00077552  0.00326268  0.00252958  0.00082248]]
# # [ True False  True  True  True  True False False  True  True]
# # True
# # True
# # [[-1.25000000e-02 -1.25000000e-02  3.12500000e-03 -7.81250000e-03
# #   -7.81250000e-03  6.25000000e-03  6.25000000e-03 -6.25000000e-03
# #    9.37500000e-03 -1.87500000e-02]
# #  [ 7.76004194e-03  7.71303751e-03  8.26634500e-03  7.69434983e-03
# #    7.58180590e-03  7.29486809e-03  7.80653714e-03  7.64925281e-03
# #    7.97154017e-03  7.49856452e-03]
# #  [-7.77420092e-03 -4.61089997e-03 -3.97510535e-03 -2.58317765e-02
# #   -1.42153394e-02  7.63243571e-03  8.83309973e-03 -5.79390473e-04
# #    3.68135202e-03 -2.90328584e-02]
# #  [ 1.26559164e-02  1.17590549e-02  1.30137858e-02  1.29480118e-02
# #    1.15932171e-02  1.17668931e-02  1.10104793e-02  1.24032413e-02
# #    1.30528332e-02  1.19347587e-02]
# #  [ 8.50000000e-03  8.50000000e-03  7.12500000e-03  3.81250000e-03
# #    3.81250000e-03  1.02500000e-02  1.02500000e-02  2.25000000e-03
# #    1.33750000e-02  1.47500000e-02]
# #  [ 3.77420092e-03  6.10899975e-04  2.48946457e-05  2.18317765e-02
# #    1.02153394e-02  1.16324357e-02  1.28330997e-02  3.42060953e-03
# #    7.68135202e-03  2.50328584e-02]]
# # [False False False  True  True  True  True  True False  True]
# # True
# # True
# # ax.text(0.1, 0.85, 'm=' + str(round(e1mc[0] - 1, 6)) + '$\pm$' + str(round(e1mc[1], 6)), color='green', ha='left',
# #         va='center', transform=ax.transAxes, fontsize=20)
# # ax.text(0.1, 0.8, 'c=' + str(round(e1mc[2], 6)) + '$\pm$' + str(round(e1mc[3], 6)), color='green', ha='left',
# #         va='center', transform=ax.transAxes, fontsize=20)
# # ax.text(0.1, 0.75, "[ " + cut_start + ", " + cut_end + "]", color='green', ha='left', va='center',
# #         transform=ax.transAxes,
# #         fontsize=20)
#
#
#
#
# #10w
# # [[-0.01699219 -0.01523438 -0.0125     -0.0171875  -0.01640625 -0.015625
# #   -0.0125     -0.021875   -0.015625   -0.01289063]
# #  [ 0.00228711  0.00231589  0.00249022  0.00238915  0.00234296  0.00247885
# #    0.00224653  0.0026928   0.00233081  0.00232863]
# #  [-0.01662626 -0.01232423 -0.01638236 -0.01984534 -0.01833696 -0.02041305
# #   -0.01518549 -0.02778712 -0.01333747 -0.01637132]
# #  [ 0.00379599  0.00364419  0.00374744  0.00371317  0.00367458  0.00374001
# #    0.00375887  0.00373711  0.00358682  0.00389296]
# #  [ 0.00099219  0.00076562  0.0035      0.0011875   0.00040625  0.000375
# #    0.0035      0.005875    0.000375    0.00310937]
# #  [ 0.00062626  0.00367577  0.00038236  0.00384534  0.00233696  0.00441305
# #    0.00081451  0.01178712  0.00266253  0.00037132]]
# # [False  True False  True  True  True False  True  True False]
# # True
# # True
