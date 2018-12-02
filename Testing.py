import numpy as np

from sys import path
import matplotlib.pyplot as plt
plt.switch_backend('agg')
# path.append('/home/hklee/work/fourier_quad/')

from F_Q_P_S import F_Q_P_S
from Fourier_Quad import Fourier_Quad

from tool_box import *

import time

from sys  import argv


start = time.clock()
from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()

cpus = comm.Get_size()

flux =5

size = 48
num_po = 50
radius = 5
gal_num =int(argv[1])
ind = int(argv[2])

# g = np.load('g.npz')['arr_0']
g = np.zeros((2,10))
g[0] = np.linspace(-0.03,0.03,10)
g[1] = np.linspace(-0.03,0.03,10)

est_g = np.zeros((8, 1))
# seed = np.random.randint(0,100000000,10)
obj = Fourier_Quad(size, rank*10+111+ind)
# obj = F_Q_P_S(size, rank*10+111)

psf_scale = 4

noise = 1
psf= obj.cre_psf(psf_scale,flux)#+obj.draw_noise(0,1)
p_ps = obj.pow_spec(psf)
# p_ps2 = fit_2o(obj.pow_spec(psf+obj.draw_noise(0,1)))
# p_ps = fit_2o(p_ps)
hlr = obj.get_radius_new(p_ps,2.)[0]

res = np.zeros((4, gal_num ))
res1 = np.zeros((4, gal_num ))
read = time.clock()
# image = np.load('/mnt/perc/shenzhi/image100000_%.3f.npz'%g[0,rank])['arr_0']
step1 = time.clock()
# print "time to read image = %.6f"%(step1-read)
l,ll = 0,0
for i in range(gal_num):

    she_pos = obj.ran_pos(num_po, radius, (g[0,rank],g[1,rank]))[1]

    cov_image = obj.convolve_psf(pos=she_pos, psf_scale=psf_scale, flux=flux)

    noise1 = obj.draw_noise(0,noise)
    gal_image = cov_image +noise1
    f1 = time.clock()
    # gal_image=image[i,:,:]
    f2 = time.clock()
    tt = f2-f1
    l = l+tt
    f3 = time.clock()

    gal_p =fit_2o2(obj.pow_spec(gal_image))
    gal_ps = fit_2o2(gal_p)
    noise2 = obj.draw_noise(0,noise)
    noi_p = fit_2o2(obj.pow_spec(noise2))
    noi_ps = fit_2o2(noi_p)

    f4 = time.clock()
    ttt = f4-f3
    ll = ll+ttt
    res[0,i],res[1,i], res[2,i],res[3,i] = obj.shear_est(gal_image,psf,noise2)[0:4]
    res1[0,i],res1[1,i], res1[2,i],res1[3,i] = obj.shear_est(gal_ps,p_ps,noi_ps,F=True,G=True)[0:4]
# np.savez('res_%d.npz'%ind,res)
# np.savez('res1_%d.npz'%ind,res1)
# np.savez('/mnt/perc/shenzhi/image%d_%.3f'%(gal_num,g[0,rank]),image)
if rank ==0:
    print "time to fit =%.6f"%ll

est_g[0], est_g[1] = obj.fmin_g(res[0],res[2] + res[3], 8)
est_g[2], est_g[3] = obj.fmin_g(res[1],res[2] - res[3], 8)
est_g[4], est_g[5] = obj.fmin_g(res1[0],res1[2] + res1[3], 8)
est_g[6], est_g[7] = obj.fmin_g(res1[1],res1[2] - res1[3], 8)

if rank > 0:

    comm.Send(est_g, dest=0, tag=rank)

else:

    for i in range(1, cpus):
        recv = np.empty((8, 1), dtype=np.float64)

        comm.Recv(recv, source=i, tag=i)

        est_g = np.column_stack((est_g, recv))

if rank == 0:
    np.savez("/home/shenzhi/test/fit_%d.npz"%(gal_num),est_g)
    print(est_g)
    x = g[0]
    y = est_g[0]
    y_err = est_g[1]

    y2 = est_g[4]
    y2_err = est_g[5]
    m,m_sig,c,c_sig =data_fit(x, y, y_err)
    m1,m1_sig,c1,c1_sig =data_fit(x,y2,y2_err)
    msig = np.mean(est_g[1])
    nsig =  np.mean(est_g[3])
    # plt.subplot(121)
    plt.plot(x, x, 'k')
    plt.errorbar(x, y, y_err, label='PDF', fmt = ',',color='r', ecolor='r',capsize=5)
    plt.errorbar(x, y2, y2_err, label='PDF_fit', fmt = ',',color='b', capsize=3)
    plt.text(-0.028, .020, r'$m={%.5f}({%.7f})$' % (m, m_sig))
    plt.text(-0.028, .016, r'$c={%.6f}({%.7f})$' % (c, c_sig))
    plt.text(-0.028, .012, r'$m={%.5f}({%.5f})$' % (m1, m1_sig))
    plt.text(-0.028, .008, r'$c={%.6f}({%.7f})$' %(c1, c1_sig))
    plt.text(-0.028, .004, r'errorbar=${%.6f},{%.7f}$' %(msig,nsig))

    # plt.xticks(np.arange(-0.03, 0.03, 0.01))
    # plt.yticks(np.arange(-0.03, 0.03, 0.01))

    plt.title('Est_g1')
    plt.xlim()
    plt.ylim()

    plt.legend()
    plt.savefig('/home/shenzhi/data2/%d_sn%d_%d.png'%(gal_num,flux,ind),dpi=600)
end = time.clock()
print("Whole time = %.2f" % (end - start))