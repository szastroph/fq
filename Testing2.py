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

flux =10

size = 48
num_po = 50
radius = 5
gal_num =int(argv[1])
g = np.zeros((2,10))
g[0] = np.linspace(-0.03,0.03,10)
g[1] = np.linspace(-0.03,0.03,10)


est_g = np.zeros((4, 1))
# seed = np.random.randint(0,100000000,10)
obj = Fourier_Quad(size, rank*10+111)
# obj = F_Q_P_S(size, rank*10+111)

psf_scale = 3

noise = 1
psf= obj.cre_psf(psf_scale,flux)
p_ps = obj.pow_spec(psf)
hlr = obj.get_radius_new(p_ps,2.)[0]
# flux = flux_cfhtls(gal_num,200,1.2e+06)#120W
# noise = 43
res = np.zeros((4, gal_num ))

step1 = time.clock()
for i in range(gal_num):
    she_pos = obj.ran_pos(num_po, radius, (g[0,rank],g[1,rank]))[1]

    cov_image = obj.convolve_psf(pos=she_pos, psf_scale=psf_scale, flux=flux)

    noise1 = obj.draw_noise(0,noise)

    gal_image=cov_image+noise1
    ps_gal = obj.pow_spec(gal_image)
    gal_image = smooth(ps_gal,size)

    noise2 = obj.draw_noise(0,noise)
    noise2 = obj.pow_spec(noise2)
    noise2 = smooth(noise2,size)

    res[0,i], res[1,i], res[2, i], res[3,i] = obj.shear_est(gal_image, p_ps,noise2,F=True, G=True)[0:4]


est_g[0], est_g[1] = obj.fmin_g(res[0],res[2]+ res[3], 8)
est_g[2], est_g[3] = obj.fmin_g(res[1],res[2]- res[3], 8)




# est_g[2,0] =np.mean(res[0])/np.mean(res[2])
# est_g[3,0] =np.sqrt(np.mean(np.mean(res[0]**2))/np.mean(res[2])**2/len(res[0]))
# est_g[4, 0], est_g[5, 0] = obj.fmin_g(res[1], res[2], res[3], 2, 8)
# est_g[6,0] =np.mean(res[1])/np.mean(res[2])
# est_g[7,0] =np.sqrt(np.mean(np.mean(res[1]**2))/np.mean(res[2])**2/len(res[0]))
#
if rank > 0:

    comm.Send(est_g, dest=0, tag=rank)

else:

    for i in range(1, cpus):
        recv = np.empty((4, 1), dtype=np.float64)

        comm.Recv(recv, source=i, tag=i)

        est_g = np.column_stack((est_g, recv))

if rank == 0:
    np.savez("/home/shenzhi/test/fit_%d.npz"%(gal_num),est_g)
    print(est_g)
    x = g[0]
    y = est_g[0]
    y_err = est_g[1]
    # y2 = est_g[2]
    # y2_err = est_g[3]
    m,m_sig,c,c_sig = data_fit(x, y, y_err)
    # m1,m1_sig,c1,c1_sig =data_fit(x,y2,y2_err)
    # msig = np.mean(est_g[1])
    # nsig =  np.mean(est_g[3])

    plt.plot(x, x, 'k')
    plt.errorbar(x, y, y_err, label='PDF_SYM', fmt = ',',color='r', ecolor='r',capsize=5)
    # plt.errorbar(x, y2, y2_err, label='AVE_PDF', fmt = ',',color='b', capsize=3)
    plt.text(-0.028, .020, r'$m={%.5f}({%.7f})$' % (m, m_sig))
    plt.text(-0.028, .016, r'$c={%.6f}({%.7f})$' % (c, c_sig))
    # plt.text(-0.028, .012, r'$m={%.5f}({%.5f})$' % (m1, m1_sig))
    # plt.text(-0.028, .008, r'$c={%.6f}({%.7f})$' %(c1, c1_sig))
    # plt.text(-0.028, .004, r'errorbar=${%.6f},{%.7f}$' %(msig,nsig))

    plt.xticks(np.arange(-0.03, 0.03, 0.01))
    plt.title('Est_g1')
    plt.xlim()
    plt.ylim()
    plt.legend()
    plt.savefig('/home/shenzhi/data2/%d_sn%d.png'%(gal_num,flux),dpi=600)
end = time.clock()
print("Whole time = %.2f" % (end - start))