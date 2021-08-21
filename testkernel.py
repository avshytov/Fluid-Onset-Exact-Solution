import numpy as np
import pylab as pl
from xkernel_new import WHKernels
from contours import make_arc
import path
import time


def make_contours(k, gamma, gamma1):
    kappa = np.sqrt(k**2 + gamma**2)
    path_ul = make_arc(k, kappa)
    path_ru = path.transform(path_ul, lambda z: complex(-z.real, z.imag))
    path_ur = path.reverse(path_ru)
    path_up = path.append_paths(path_ul, path_ur)
    path_dn = path.transform(path_up, lambda z: complex(z.real, -z.imag))
    K = WHKernels(gamma, gamma1)
    now = time.time()
    K_up_rho = np.vectorize(lambda z: K.rho(k, z))(path_up.points())
    K_up_omega = np.vectorize(lambda z: K.omega(k, z))(path_up.points())
    K_dn_rho = np.vectorize(lambda z: K.rho(k, z))(path_dn.points())
    K_dn_omega = np.vectorize(lambda z: K.omega(k, z))(path_dn.points())
    print ("K tabulated, time ", time.time() - now); now = time.time()
    Krho_p_up = np.vectorize(lambda z: K.rho_plus(k, z))(path_up.points())
    Komega_p_up = np.vectorize(lambda z: K.omega_plus(k, z))(path_up.points())
    print ("K+ tabulated, time", time.time() - now); now = time.time()
    z_up = path_up.points()
    z_dn = path_dn.points()
    z_z1  = np.outer(np.ones(np.shape(z_up)), z_dn)
    z_z1 -= np.outer(z_up,                    np.ones(np.shape(z_dn)))
    f_rho_up = np.log(K_dn_rho) / z_z1 / (2.0 * np.pi * 1j)
    f_omega_up = np.log(K_dn_omega) / z_z1 / (2.0 * np.pi * 1j)
    log_Krho_p_new = path_dn.integrate_array(f_rho_up)
    log_Komega_p_new = path_dn.integrate_array(f_omega_up)
    #z_fact = 1j * gamma / np.pi / np.sqrt(z_up**2 + kappa**2)

    z_b = path_dn.begins_at()
    z_e = path_dn.ends_at()
    z_be = 0.5 * (z_b + z_e)
    #u_b = z_b + np.sqrt(z_b**2 ) 
    #log_Krho_p += z_fact * np.log(z_b + z_up )
    # integral from the endpoints to the infinity can be approximated:
    log_corr = 1j / np.pi / z_up * np.log(z_be/(z_be - z_up))
    log_Krho_p_new += gamma * log_corr
    log_Komega_p_new += 2 * gamma1 * log_corr
    
    Krho_p_new = np.exp(-log_Krho_p_new)
    Komega_p_new = np.exp(-log_Komega_p_new)
    print ("contour integrals done, time", time.time() - now)
    pl.figure()
    pl.plot(path_up.points().real, path_up.points().imag)
    pl.plot(path_dn.points().real, path_dn.points().imag)
    pl.legend()
    pl.figure()
    x_arc = path_up.arc_lengths()
    pl.plot(x_arc, Krho_p_up.real, label='Re Krho_old')
    pl.plot(x_arc, Krho_p_up.imag, label='Im Krho_old')
    pl.plot(x_arc, Krho_p_new.real, '--', label='Re Krho_new')
    pl.plot(x_arc, Krho_p_new.imag, '--', label='Im Krho_new')
    pl.legend()
    pl.plot(x_arc, Komega_p_up.real, label='Re Komega_old')
    pl.plot(x_arc, Komega_p_up.imag, label='Im Komega_old')
    pl.plot(x_arc, Komega_p_new.real, '--', label='Re Komega_new')
    pl.plot(x_arc, Komega_p_new.imag, '--', label='Im Komega_new')
    pl.legend()
    pl.figure()
    pl.plot(x_arc, Krho_p_up.real - Krho_p_new.real, label='diff Re Krho')
    pl.plot(x_arc, Krho_p_up.imag - Krho_p_new.imag, label='diff Im Krho')
    pl.plot(x_arc, Komega_p_up.real - Komega_p_new.real,label='diff Re Komega')
    pl.plot(x_arc, Komega_p_up.imag - Komega_p_new.imag,label='diff Im Komega')
    pl.legend()

    pl.show()

k = 0.5
gamma = 1.0
gamma1 = 0.9
make_contours(k, gamma, gamma1)
