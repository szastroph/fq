
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tool_box import *
# from scipy import optimize
# from scipy import asarray as ar, exp
# from scipy.optimize import curve_fit
# import PIL
# from PIL import ImageFont
# from PIL import Image
# from PIL import ImageDraw
# # from tool_box import *
from F_Q_P_S import F_Q_P_S
gal_num = 100000
flux = 100
g = np.zeros((2,10))
g[0] = np.linspace(-0.03,0.03,10)
g[1] = np.linspace(-0.03,0.03,10)
est_g = np.load("/home/shenzhi/test/fit_%d.npz" %gal_num)['arr_0']
x = g[0]
y = est_g[0]
y_err = est_g[1]

y2 = est_g[4]
y2_err = est_g[5]
m, m_sig, c, c_sig = data_fit(x, y, y_err)
m1, m1_sig, c1, c1_sig = data_fit(x, y2, y2_err)
msig = np.mean(est_g[1])
nsig = np.mean(est_g[3])
# plt.subplot(121)
plt.plot(x, x, 'k')
plt.errorbar(x, y, y_err, label='PDF', fmt=',', color='r', ecolor='r', capsize=5)
plt.errorbar(x, y2, y2_err, label='PDF_fit', fmt=',', color='b', capsize=3)
plt.text(-0.028, .020, r'$m={%.5f}({%.7f})$' % (m, m_sig))
plt.text(-0.028, .016, r'$c={%.6f}({%.7f})$' % (c, c_sig))
plt.text(-0.028, .012, r'$m={%.5f}({%.5f})$' % (m1, m1_sig))
plt.text(-0.028, .008, r'$c={%.6f}({%.7f})$' % (c1, c1_sig))
plt.text(-0.028, .004, r'errorbar=${%.6f},{%.7f}$' % (msig, nsig))

# plt.xticks(np.arange(-0.03, 0.03, 0.01))
# plt.yticks(np.arange(-0.03, 0.03, 0.01))

plt.title('Est_g1')
plt.xlim()
plt.ylim()

plt.legend()
plt.savefig('/home/shenzhi/data2/%d_sn%d.png' % (gal_num, flux), dpi=600)



# # a = np.load('flux.npz')['arr_0']
# # print a.shape
# # b,c  = np.histogram(a,15)
# # plt.hist(b,c)
# # plt.show()
#
#
# # plot CFHT_simulation
# # plt.subplot(121)
#
# # g1 = np.linspace(-0.005,0.004,10)
# # plt.figure()
# # # x = np.linspace(1,9,9)
# # # y = np.zeros(10)
# # # y_err = np.zeros(10)
# # # y2 = np.zeros(9)
# # # y2_err = np.zeros(9)
# # x0 = g1
# #
# # d = np.load('t_g_sn50.npz')['arr_0']
# # print d
# #
# # y = d[0]
# # y_err = d[1]
# # y2 = d[2]
# # y2_err = d[3]
# #
# # plt.plot(x0,x0,'k')
# # plt.errorbar(x0,    y, y_err,      label = 'PDF_SYM',color = 'r',capsize=5)
# # plt.errorbar(x0,    y2, y2_err,  label = 'AVE_PDF',color = 'b',capsize=3)
# # plt.errorbar(x0,    y, y_err,    color = 'r',capsize=5)
# # plt.errorbar(x0,    y2, y2_err,  color = 'b',capsize=3)
# # plt.xticks(g1)
# # plt.title('Est_g1')
# # plt.legend()
#
# # plt.subplot(122)
# # x0 = g1
# # for i in range(9):
# #     d = np.load('t_g_%.3f.npz'%g1[i])['arr_0']
# #
# #
# #     y[i] = d[6,0]
# #     y_err[i] = d[7,0]
# #     y2[i] = d[8,0]
# #     y2_err[i] = d[9,0]
# #
# # plt.plot(x0,x0,'k')
# # plt.errorbar(x0,    y,y_err,      label = 'PDF_SYM',color = 'r',capsize=5)
# # plt.errorbar(x0+0,    y2,y2_err,  label = 'AVE_PDF',color = 'b',capsize=3)
# # plt.errorbar(x0,    y,y_err,    color = 'r',capsize=5)
# # plt.errorbar(x0+0,    y2,y2_err,  color = 'b',capsize=3)
# # plt.title('Est_g2')
# # plt.legend()
# #
# # plt.show()
# #
# # plt.savefig('SN5.png',dpi=600)
# #plot pdf_ave
# # obj = F_Q_P_S(54,1233444)
# # #plot pdf - ave result
# # a = np.load('t_g.npz')['arr_0']
# # print a.shape
# # g1 = np.linspace(-0.02,0.016,10)
# # g2 = np.linspace(-0.02,0.016,10)
# # plt.figure()
# # plt.subplot(121)
# # x = np.arange(1,11,1)
# # y1 = a[0]
# # y1_err = a[1]
# # y2 = a[2]
# # y2_err = a[3]
# # plt.plot(g1,g1,color = 'g')
# # plt.errorbar(g1,y1,y1_err,color= 'r',capsize=3,label = 'PDF')
# # plt.errorbar(g1,y2,y2_err,color = 'b',capsize=5,label = 'AVE')
# # plt.text(-0.02,0.008," m: 0.0186,m_sig:0.0243"+'\n'+" c: -0.000403,c_sig:0.000284",size=12,bbox = dict(facecolor = "r", alpha = 0.2))
# #
# # plt.title('Est_g1')
# # plt.legend()
# # # m: 0.0186(0.0243), c: -0.000403(0.000284)
# # # m: 0.(0.0222), c: -0.(0.)
# #
# # plt.subplot(122)
# # y3 = a[6]
# # y3_err = a[7]
# # y4 = a[8]
# # y4_err = a[9]
# # plt.plot(g2,g2,color = 'k')
# # plt.errorbar(g2,y3,y3_err,color= 'r',capsize=3,label = 'PDF')
# # plt.errorbar(g2,y4,y4_err,color = 'b',capsize=5,label = 'AVE')
# # plt.title('Est_g2')
# # plt.text(-0.02,0.01," m: 0.0227,m_sig:0.0222"+'\n'+" c: -0.000270,c_sig:0.000259",size=12,bbox = dict(facecolor = "r", alpha = 0.2))#bbox = dict(facecolor = "r", alpha = 0.2)
# # plt.legend()
# # plt.show()
#
#
#
# #plot sample_SNR & check the image
# #
# from F_Q_P_S import F_Q_P_S
# # import Fourier_Quad
#
# g1 =np.array([-0.028,-0.023,-0.007,-0.003,0.003,0.008,0.016,0.021,-0.016,0.028])
#
# snr = [5,10,20,50]
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     est_g = np.load('sn%d.npz' % snr[i])['arr_0']
#     x = g1
#     x0 = g1
#     y = est_g[0]
#     y_err = est_g[1]
#     y2 = est_g[2]
#     y2_err = est_g[3]
#     m, m_sig, c, c_sig = data_fit(x, y, y_err)
#     m1, m1_sig, c1, c1_sig = data_fit(x, y2, y2_err)
#     msig = np.mean(est_g[1])
#     nsig = np.mean(est_g[3])
#
#     plt.plot(x0, x0, 'k')
#     plt.errorbar(x0, y, y_err, label='PDF_SYM', fmt=',', color='r', ecolor='r', capsize=5)
#     plt.errorbar(x0, y2, y2_err, label='AVE_PDF', fmt=',', color='b', capsize=3)
#     # plt.text(-0.028, .020, r'$m={%.5f}({%.7f})$' % (m, m_sig))
#     # plt.text(-0.028, .016, r'$c={%.6f}({%.7f})$' % (c, c_sig))
#     # plt.text(-0.028, .012, r'$m={%.5f}({%.5f})$' % (m1, m1_sig))
#     # plt.text(-0.028, .008, r'$c={%.6f}({%.7f})$' % (c1, c1_sig))
#     plt.text(-0.028, .020, r'errorbar=${%.6f},{%.7f}$' % (msig, nsig))
#     plt.text(-0.028, .010, r'$sn{%d}$' % (snr[i]))
#
# plt.savefig('10000_diffsn.png',dpi=600)
# plt.show()


from subprocess import Popen
import numpy as  np
import matplotlib.pyplot as plt
# plt.switch_backend('agg')
#
# # for i in range(6):
# #
# #     cmd = 'mpiexec -n 10 python Testing2.py %d %d %d'%(i,snr[i],gal_num)
# #
# #     a = Popen(cmd, shell=True)
# #
# #     a.wait()
#
# gal_num = 100000
# # a = np.load('/home/shenzhi/data3/est_g1_0.npz' )['arr_0']
# # b = np.load('/home/shenzhi/data3/est_g1_1.npz' )['arr_0']
# # est_g1 = np.concatenate((a,b),1)
# snr1 = np.load('snr_1.5_1.2.npz')['arr_0']
# snr = np.zeros(5)
# for i in range(5):
#     snr[i] = snr1[2,i+4]
#
#
# est_g1 = np.zeros((5,400))
# for i in range(400):
#     a = np.load('/home/shenzhi/data3/%d_%d.npz'%(gal_num,i))['arr_0']
#     est_g1[:,i] = a[0,4:9]
# plt.figure()
#
#
# a = np.load('/home/shenzhi/data2/tatal1.5_1.2_%d.npz'%gal_num)['arr_0']
# errorbar1 = a[1,4:9]*np.sqrt(gal_num)
# errorbar2 = a[3,4:9]*np.sqrt(gal_num)
#
#
# print est_g1.shape
# errorbar3 = np.zeros(5)
# for i in range(5):
#     errorbar3[i] = np.std(est_g1[i])*np.sqrt(gal_num)
# plt.plot(snr,errorbar3,label = 'True errorbar')
#
# plt.rcParams['xtick.direction']='in'
# plt.rcParams['ytick.direction']='in'
# plt.tick_params(top = 'on',right = 'on',which = 'both')
# plt.plot(snr,errorbar1,label = 'Seperated')
# plt.plot(snr,errorbar2,label = 'Gathered')
#
# plt.xticks(snr)
# plt.xscale('log')
# plt.title('Errorbar change law')
# plt.xlabel('SNR')
# plt.ylabel('Shape Noise')
# plt.legend()
# plt.savefig('change_law_%d-1.5-1.2-h.png'%gal_num,dpi = 600)


# import numpy as np
#
# from sys import path
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
# # path.append('/home/hklee/work/fourier_quad/')
#
# from Fourier_Quad import Fourier_Quad
#
# import time
#
# from sys  import argv
#
# start = time.clock()
# # from mpi4py import MPI
# #
# # comm = MPI.COMM_WORLD
# #
# # rank = comm.Get_rank()
# #
# # cpus = comm.Get_size()
#
# flux = 5
#
# size = 64
# num_po = 50
# radius = 6
# gal_num = 10
#
# g1 = np.linspace(-0.03,0.03,10)
# # g1 = np.sort(g1)
# g2 = np.linspace(-0.03, 0.03, 10)
#
# est_g = np.zeros((10, 1))
# # seed = np.random.randint(0,100000000,10)
# obj = Fourier_Quad(size, 10+111)
# # obj = F_Q_P_S(size, rank*10+111)
#
# psf_scale = 3
#
# noise = 1
# psf= obj.cre_psf(psf_scale,flux)
# p_ps = obj.pow_spec(psf)
# hlr = obj.get_radius_new(p_ps,2.)[0]
# # flux = flux_cfhtls(gal_num,200,1.2e+06)#120W
# # noise = 43
#
# res = np.zeros((4, gal_num*7 ))
# res1 = np.zeros((4,gal_num))
# res2 = np.zeros((4,gal_num))
# # d_img = np.zeros((7,size,size))
#
# for i in range(gal_num):
#     she_pos = obj.ran_pos(num_po, radius, (g1[0], g2[0]))[1]        # psf = obj.cre_psf(psf_scale=psf_scale,flux=flux)
#
#     psf_image = obj.convolve_psf(pos=she_pos, psf_scale=psf_scale, flux=flux)
#
#     noise1 = obj.draw_noise(0,noise)
#     noise2 = obj.draw_noise(0,noise)
#
#     gal_image=psf_image+noise1
#
#     area = obj.get_radius_new(gal_image,2.)[1]
#     total_flux = obj.get_radius_new(gal_image,2.)[2]
#     print area#,total_flux,total_flux/area