import numpy as np
import pylab as pl
from scipy import integrate
import scipy
import random
from scipy import linalg
#import _makequad as mq
from scipy import special
import time
import frhs_old as frhs
#from whsolve4 import Fourier as Fourier_old
#from whsolve5old import im_plus as im_plus_old

gamma = 1.0

def quad_grid(qmax, N):
    xi = np.linspace(-1.0, 1.0, N)
    return qmax * xi * np.abs(xi)

def exp_grid(qmin, qmax, N):
    xi  = np.log(qmax/qmin) * np.linspace(0.0, 1.0, int(N/2))
    q_u = qmin * np.exp(xi)
    q_d = list(-q_u)
    q_d.reverse()
    q_d.extend(q_u)
    return np.array(q_d)

from sqrtint import sqrt_integrator, invsqrt_integrator
from contourint import SinglePathIntegrator, JointPathIntegrator
from xkernel_old import Xrho, Xrho_z, Xrho_plus, Xrho_minus, Xrho_plus_z, Xrho_minus_z

from xkernel import WHKernels

from xkernel_old import Xomega, Xomega_z, Xomega_plus, Xomega_minus, Xomega_plus_z,Xomega_minus_z

from xkernel_old import Xrho_star                                                   
from frhs_old import Frho, Fomega, Fm 
    
class WHData:
    def __init__ (self):
        self.q = None
        self.flux_s = None
        self.flux_j = None
        self.drho_j = None
        self.rho_j  = None
        self.rho_s  = None
        self.omega_s = None
        self.omega_j = None
        self.f_s = None
        self.phi_s = None
        self.phi_j = None
        self.k = None
        self.D = None
        self.bc_omega_s = None
        self.bc_omega_j = None
        self.bc_drho_j = None
        self.bc_rho_s = None
        self.jx = None
        self.jy = None
        self.bc_phi_j = None
        self.bc_phi_s = None
        self.bc_phi_sin = None
        self.bc_phi_cos = None
        self.rho_sin = None
        self.rho_cos = None
        self.jx_cos = None
        self.jy_cos = None
        self.jx_sin = None
        self.jy_sin = None


def wh_solve(h, delta, k, yv, src='isotropic'):
    absk = abs(k)
    kappa = np.sqrt(k**2 + gamma**2)

    #t_up = np.linspace(-50.0, 50.0, 10001)
    #t_up = exp_grid(1e-2, 30.0, 5001)
    #z_up = 2.0/np.pi * kappa * np.arctan(t_up) \
    #       + 1j * (absk/2.0 + kappa * (np.sqrt(t_up**2 + 1.0) - 1.0))
    t_circle = quad_grid(np.pi/2.0, 501) + 1.5 * np.pi
    a = np.sqrt(k * kappa)
    z_circle = a * np.cos(t_circle) + 1j * (kappa + (kappa - absk/2)*np.sin(t_circle))
    t_vert1 = np.linspace(100.0, 1, 4000)
    z_vert1= -a + 1j * kappa * t_vert1
    t_vert2 = np.linspace(1, 100, 4000)
    z_vert2 = a + 1j * kappa * t_vert2
    z_up = list(z_vert1)
    z_up.extend(list(z_circle[1:-1]))
    z_up.extend(list(z_vert2))
    z_up = np.array(z_up)
    print("len of the contour: ", len(z_up))
    if True:
        import pylab as pl
        pl.figure()
        pl.plot(z_up.real, z_up.imag)
        #pl.show()
    
    #z_vert2 = kappa - 1j * kappa * t_vert
    #t_pole = np.linspace(0.0, 2.0*np.pi, 101)
    #R_pole = 0.2 * (kappa - absk)
    #z_pole_up = 1j * absk + R_pole * np.exp(1j * t_pole)
    #z_pole_down = - 1j * absk + R_pole * np.exp(-1j * t_pole)
    #t_cut = np.linspace(-50.0, 50.0, 10001)
    #l_cut = 0.5 * (kappa - absk); 
    #z_cut_up  = 1j * kappa + 2.0/np.pi * np.arctan(t_cut)
    #z_cut_up += 1j * (-l_cut + kappa * (np.sqrt(t_cut**2 + 1.0) - 1.0)) 

    # Choice of conjugated contours allows one to save on calculations
    # of Xrho, seel below. Please keep the contours symmetric 
    z_down = z_up.conj()
    # Xo_m_up = np.vectorize(lambda z: Xomega_minus_z(k, z))(z_up)
    #Xomega_up = np.vectorize(lambda z: Xomega_z(k, z))(z_up)
    #Xomega_down = np.vectorize(lambda z: Xomega_z(k, z))(z_down)
    #Xo_p_down = np.vectorize(lambda z: Xomega_plus_z(k, z))(z_down)
    #Xo_p_up = Xo_m_up / Xomega_up
    #Xo_m_down = Xo_p_down * Xomega_down
    
    #Xo_p_up   = Xomega_plus_z(k, z_up)
    #Xo_p_down = Xomega_plus_z(k, z_down)
    #Xo_m_up = Xomega
    c_up_int = SinglePathIntegrator(z_up)
    c_down_int = SinglePathIntegrator(z_down)
    exp_zy_up = np.exp(1j*np.outer(np.abs(yv), z_up)) 
    exp_zyh_up = np.exp(1j*np.outer(np.abs(yv - h), z_up))
    exp_zy_down = np.exp(-1j*np.outer(np.abs(yv), z_down)) 

    
    now = time.time()
    print("**** Calculate X(s)")
    #Xrho_m_up = np.vectorize(lambda z: Xrho_minus_z(k, z))(z_up)

    K = WHKernels(1.0, 1.0)
    # Economise on left-right conjugation symmetry
    
    z_down_left = z_down[0:int(len(z_up)/2)+1]
    # for testing:
    #z_down_left = 0.001j + np.linspace(-10.0, 10.0, 1001)
    Xrho_p_down_left = np.vectorize(lambda z: Xrho_plus_z(k, z))(z_down_left)
    Xrho_p_down_left_new = np.vectorize(lambda z: K.rho_plus(k, z))(z_down_left)
    if True:
        import pylab as pl
        pl.figure()
        #x_l = z_down_left.real
        x_l = np.array(range(0, len(z_down_left)))
        pl.plot(x_l,
                np.log(Xrho_p_down_left).real,
                label='Re Xrho')
        pl.plot(x_l,
                np.log(Xrho_p_down_left).imag,
                label='Im Xrho')
        pl.plot(x_l,
                np.log(Xrho_p_down_left_new).real,
                '--', label='Re Xrho new')
        pl.plot(x_l,
                np.log(Xrho_p_down_left_new).imag,
                '--', label='Im Xrho new')
        pl.legend()
        pl.show()
    Xrho_p_down = 0.0 * z_down + 0.0j
    Xrho_p_down[0:len(z_down_left)] = Xrho_p_down_left
    Xrho_p_down[-1:-len(z_down_left)-1:-1] = Xrho_p_down_left.conj()

    # We also economize on the symmetry between Xrho+ and Xrho-
    #Xrho_p_down = 1.0/Xrho_m_up.conj()
    Xrho_m_up = 1.0/Xrho_p_down.conj()
    #Xrho_p_down = np.vectorize(lambda z: Xrho_plus_z(k, z))(z_down)
    #Xrho_m_up = np.vectorize(lambda z: Xrho_minus_z(k, z))(z_up)
    #Xrho_p_down = np.vectorize(lambda z: Xrho_plus_z(k, z))(z_down)
    if False:
        pl.figure()
        pl.plot(t_up, Xrho_m_up.real, label='Re Xrho_m_up')
        pl.plot(t_up, Xrho_m_up.imag, label='Im Xrho_m_up')
        pl.plot(t_up, Xrho_p_down.real, label='Re Xrho_m_down')
        pl.plot(t_up, Xrho_p_down.imag, label='Im Xrho_m_down')
        pl.legend()
        print("X_p - 1/X_m*: ", linalg.norm(Xrho_p_down - 1.0/Xrho_m_up.conj()))
        pl.figure()
        pl.plot(t_up, (Xrho_p_down - 1.0/Xrho_m_up.conj()).real, label='Re X+ - 1/X-*')
        pl.plot(t_up, (Xrho_p_down - 1.0/Xrho_m_up.conj()).imag, label='Re X+ - 1/X-*')
        pl.plot(t_up, (1/Xrho_p_down - Xrho_m_up.conj()).real, label='Re 1/X+ - X-*')
        pl.plot(t_up, (1/Xrho_p_down - Xrho_m_up.conj()).imag, label='Re 1/X+ - X-*')
        pl.legend()
        pl.figure()
        pl.plot(t_up, Xrho_m_up.real, label='Re Xrho_m_up')
        pl.plot(t_up, Xrho_m_up.imag, label='Im Xrho_m_up')
        pl.plot(t_up, Xrho_m_up[-1::-1].real, label='Re Xrho_m_up rev')
        pl.plot(t_up, Xrho_m_up[-1::-1].imag, label='Im Xrho_m_up rev')
        pl.legend()
        print("norm (X - X*rev)", linalg.norm(Xrho_m_up - Xrho_m_up[-1::-1].conj()))
        pl.figure()
        pl.plot(t_up, (Xrho_m_up - Xrho_m_up[-1::-1].conj()).real, label='Re X - X*rev')
        pl.plot(t_up, (Xrho_m_up - Xrho_m_up[-1::-1].conj()).imag, label='Im X - X*rev')
        pl.legend()
        pl.show()
    print("done, ", time.time () - now)
    Xrho_up = np.vectorize(lambda z: Xrho_z(k, z))(z_up)
    Xrho_down = np.vectorize(lambda z: Xrho_z(k,z))(z_down)
    Xrho_up_new = np.vectorize(lambda z: K.rho_plus(k, z))(z_up)
    Xrho_down_new = np.vectorize(lambda z: K.rho_plus(k, z))(z_down)
    Xrho_m_down = Xrho_p_down * Xrho_down
    Xrho_p_up = Xrho_m_up / Xrho_up

    Xomega_up     = Xrho_up**2   * (z_up**2 + kappa**2)   / (z_up**2 + k**2)
    Xomega_down   = Xrho_down**2 * (z_down**2 + kappa**2) / (z_down**2 + k**2)
    Xomega_p_up_new = np.vectorize(lambda z: K.omega_plus(k, z))(z_up)
    #Xomega_p_down_new = np.vectorize(lambda z: K.omega_plus(k, z))(z_down)
    Xomega_p_up   = Xrho_p_up**2 * (-1j * absk + z_up) / (-1j * kappa + z_up)
    Xomega_p_down = Xrho_p_down**2 * (-1j * absk + z_down) / (-1j * kappa + z_down)
    Xomega_m_up   = Xrho_m_up**2 * ((1j * kappa + z_up) / (1j * absk + z_up))
    Xomega_m_down = Xrho_m_down**2 * ((1j * kappa + z_down) / (1j * absk + z_down))

    if True:
        import pylab as pl
        pl.figure()
        x_l = range(len(z_up))
        pl.plot(x_l, np.log(Xomega_p_up).real, label='Re log Xo')
        pl.plot(x_l, np.log(Xomega_p_up).imag, label='Im log Xo')
        pl.plot(x_l, np.log(Xomega_p_up_new).real, '--', label='Re log Xo new')
        pl.plot(x_l, np.log(Xomega_p_up_new).imag, '--', label='Im log Xo new')
        pl.legend()
        pl.show()
    print("done", time.time() - now)

    Xrho_0 = Xrho_star(k)
    Xo_0 = Xrho_0**2 * (2 * absk / (absk + kappa))

    print("Xrho_0 = ", Xrho_0)
    #Xrho_star0 = Xrho_star(k)
    print("Xo0 = ", Xo_0)

    print("new values: ", K.rho_star(k), K.omega_star(k))

    sgn_k = 0.0
    if k < 0:
       sgn_k = -1.0
    if k > 0:
       sgn_k = 1.0


    J_0 = np.exp(-absk * h)
    i_y_0 = np.argmin(np.abs(yv))
    i_y_h = np.argmin(np.abs(yv - h))
    if yv[i_y_h] < h:
       i_y_h += 1

    sgn_yh = 1.0 + 0.0 * yv
    sgn_yh[i_y_h] = 0.0
    sgn_yh[0:i_y_h] = -1.0;

    absk2_up   = z_up**2 + k**2
    absk2_down = z_down**2 + k**2

    Frho_0 = frhs.Frho_star(k)
    Fomega_0 = frhs.Fomega_star(k)
     
    # flux due to particles on the wall
    phi_f =  1.0/np.pi - 0.25 * k/gamma  - 0.125 * gamma/k/Xrho_0**2 
    phi_f += 0.25 * k/gamma * 1/Xo_0**2
    print("phi_f = ", phi_f)
    print("phi_f - 1/pi = ", phi_f - 1.0/np.pi)

    # Bulk and surface current sources require separate treatment
    if (h > delta):
        phi_j  =  - gamma**2 / 4.0 / absk**2 / Xrho_0**2 * J_0
        phi_j +=  - 0 * gamma / np.pi / k * J_0
        phi_j += 0 * gamma**2 / k * Frho_0 * J_0
        phi_j += ( 0.5 / Xo_0**2  + 0 * gamma/k * Fomega_0) * J_0
        phi_chi  = c_down_int(np.exp(-1j*z_down*h)/(z_down - 1j * absk) \
                              *(1.0/Xrho_m_down))
        phi_chi /=  2.0 * np.pi * 1j * Xrho_0 * 2.0 
        phi_j += phi_chi
        print("phi_j = ", phi_j)
        print("phi_chi = ", phi_chi)
    else:
        phi_j =  0.5 -  1.0/Xrho_0 - 0.25 * gamma**2 / k**2 / Xrho_0**2
        phi_j += 0.5/Xo_0**2
    print("flux phi_j", phi_j)

    exp_kh = np.exp(-absk * h)

    flux_sin_cos_comm   = -0.125 * gamma / absk / Xrho_0**2 * exp_kh
    flux_sin_cos_comm += 0.25 * k / gamma / Xo_0**2 * exp_kh
    dflux_sin = -0.25/np.pi/gamma/Xo_0 * k \
            *  c_down_int(k/(z_down**2 + k**2)/Xomega_m_down \
                          * np.exp(-1j*z_down*h))
    dflux_cos = -0.25/np.pi/gamma/Xo_0 * k \
                    *  c_down_int(-z_down/(z_down**2 + k**2) \
                                  /Xomega_m_down*np.exp(-1j*z_down*h))

    flux_sin = flux_sin_cos_comm + dflux_sin
    flux_cos = flux_sin_cos_comm * 1j * sgn_k + dflux_cos
    print("new flux values for sin and cos:", flux_sin, flux_cos)

    abs_yh = np.abs(yv - h)
    th_plus  = (1.0 + sgn_yh) * 0.5
    th_minus = (1.0 - sgn_yh) * 0.5

    exp_zy_up = np.exp(1j*np.outer(np.abs(yv), z_up))
    exp_zyh_up = np.exp(1j*np.outer(np.abs(yv - h), z_up))

    exp_kh = np.exp(-absk * h)
    if (h > delta):
        # rho for a bulk density source
        def chi_m(z):
            return c_down_int(1.0/(z_down - z) \
                            *np.exp(-1j*z_down*h)\
                            *(1.0/Xrho_m_down - 1.0))/2.0/np.pi/1j
        chi_m_up = np.vectorize(chi_m)(z_up)
        rho_j_up   =  1.0/gamma * (1.0/Xrho_up - 1.0)*exp_zyh_up
        rho_j_up  +=  - 1.0/gamma * Xrho_p_up * chi_m_up * exp_zy_up
        rho_j_up  += -2.0 * gamma * exp_zyh_up/(k**2 + z_up**2)
        rho_j_up  +=  gamma * exp_kh / absk / (absk - 1j * z_up)\
                      * Xrho_p_up / Xrho_0 * exp_zy_up
        drho_j_up = rho_j_up + 1.0/gamma * (Xrho_up - 1.0)*exp_zyh_up
        rho_j   = c_up_int(rho_j_up) / 2.0 / np.pi
        drho_j  = c_up_int(drho_j_up) / 2.0 / np.pi
        rho_f_up  = (-1j * z_up / (z_up**2 + k**2)) * exp_zy_up
        rho_f_up += 0.5 / (absk - 1j * z_up) * Xrho_p_up / Xrho_0 * exp_zy_up
        rho_f = c_up_int(rho_f_up) / 2.0 / np.pi
    else:
        rho_j_up   =  2.0/gamma * (Xrho_p_up - 1.0)*exp_zy_up
        rho_j_up  += -2.0 * gamma * exp_zy_up/(k**2 + z_up**2)
        rho_j_up  +=  gamma / absk / (absk - 1j * z_up) * Xrho_p_up \
                      / Xrho_0 * exp_zy_up
        drho_j_up = rho_j_up + 2.0/gamma * (Xrho_up - 1.0)*exp_zy_up
        rho_j   = c_up_int(rho_j_up) / 2.0 / np.pi
        drho_j  = c_up_int(drho_j_up) / 2.0 / np.pi
        
        rho_f_up = (-1j * z_up / (z_up**2 + k**2) \
                    + 0.5 / (absk - 1j * z_up) * Xrho_p_up / Xrho_0) \
                  * exp_zy_up
        rho_f = c_up_int(rho_f_up) / 2.0 / np.pi
    rho_j[0:i_y_0] = 0
    drho_j[0:i_y_0] = 0
    rho_f[0:i_y_0] = 0
    if False:
        pl.figure()
        pl.plot(yv, rho_f.real, label='Re rho_f')
        pl.plot(yv, rho_f.imag, label='Im rho_f')
        #pl.plot(yv, rho_f_new.real, '--', label='Re rho_f new')
        #pl.plot(yv, rho_f_new.imag, '--', label='Im rho_f new')
        pl.legend()
        pl.figure()
        pl.plot(yv, rho_j.real, label='Re rho_j')
        pl.plot(yv, rho_j.imag, label='Im rho_j')
        #pl.plot(yv, rho_j_new.real, '--', label='Re rho_j new')
        #pl.plot(yv, rho_j_new.imag, '--', label='Im rho_j new')
        pl.legend()
        pl.figure()
        pl.plot(yv, drho_j.real, label='Re drho_j')
        pl.plot(yv, drho_j.imag, label='Im drho_j')
        #pl.plot(yv, drho_j_new.real, '--', label='Re drho_j new')
        #pl.plot(yv, drho_j_new.imag, '--', label='Im drho_j new')
        pl.legend()
        pl.show()
            
    yv_lo = yv[0:i_y_h]
    yv_hi = yv[i_y_h:]

    if False:
        pl.figure()
        #pl.plot(yv, rho_j_y.real, label='rho_j')
        #pl.plot(yv, (rho_ja_exact + rho_jb_exact + rho_jc_exact + rho_jd_exact).real, '--', label='exact')
        pl.plot(yv, rho_j, label='rho_j');
        pl.plot(yv, drho_j, label='drho_j')
        pl.plot(yv, rho_j - drho_j, label='rho_0')
        pl.plot(yv, 1.0/np.pi * special.kn(0, kappa * np.sqrt(np.abs(yv - h))**2 + 1e-12), label='k0')
        #rho_exact_diff = (rho_ja_exact + rho_jb_exact + rho_jc_exact + rho_jd_exact).real - rho_j_exact.real
        #pl.plot(yv, rho_exact_diff, label='diff ex-ex')
        #pl.plot(yv, rho_j_y.real - rho_j_exact.real, label='fourier - laplace')
        pl.legend()
        pl.show()

    jx_omega_up = -z_up * Xomega_p_up / (z_up**2 + k**2) / Xo_0 * exp_zy_up
    jy_omega_up =     k * Xomega_p_up / (z_up**2 + k**2) / Xo_0 * exp_zy_up

    jx_omega = c_up_int(jx_omega_up) / 2.0 / np.pi
    jy_omega = c_up_int(jy_omega_up) / 2.0 / np.pi

    if False:
       pl.figure()
       pl.plot(yv, jx_omega.real, label='Re jx O')
       pl.plot(yv, jx_omega.imag, label='Im jx O')
       pl.plot(yv, jy_omega.real, label='Re jy O')
       pl.plot(yv, jy_omega.imag, label='Im jy O')
       #pl.plot(yv, jx_omega_new.real,'--',  label='Re jx O new')
       #pl.plot(yv, jx_omega_new.imag, '--', label='Im jx O new')
       #pl.plot(yv, jy_omega_new.real, '--', label='Re jy O new')
       #pl.plot(yv, jy_omega_new.imag, '--', label='Im jy O new')
       pl.legend()
       pl.show()
    
    if h < delta:
       jx_D = -0.5 * 1j * sgn_k * np.exp(-absk * np.abs(yv))
       jy_D = -0.5 * np.exp(-absk * np.abs(yv)) * (-1.0) + 0.0j
    else:
       jx_D =  np.exp(-absk * abs_yh)/(1 - k**2 * delta**2) + 0.0j
       ###jx_D += - absk * delta * np.exp(-abs_yh/delta)/(1 - k**2 * delta**2)
       ###jx_D += -0.5 * np.exp(-h/delta) / (1.0 + absk * delta) * np.exp(-absk * np.abs(yv))# * th_plus
       #jx_D += -0.5 * np.exp(-h/delta) / (1.0 + absk * delta) * np.exp(-absk * abs_yh)# * th_minus
       #jx_D +=  0.5 * absk * delta * np.exp(-h/delta) * np.exp(-absk * abs_yh)  / (1.0 - k**2 * delta**2) * th_minus
       jx_D *= -0.5 * 1j * sgn_k 
       jy_D  =   np.exp(-absk * abs_yh) * (-sgn_yh)/(1 - k**2 * delta**2) + 0.0j
       ###jy_D  += -  k * k * delta * delta * np.exp(-abs_yh/delta) * (-sgn_yh)/(1 - k**2 * delta**2) + 0.0j
       ###jy_D  += -0.5 * np.exp(-h/delta) / (1.0 + absk * delta) * np.exp(-absk * np.abs(yv)) * (-1)
       jy_D *= -0.5 

    jx_f = 0.25 * 1j * k * np.exp(-absk * np.abs(yv))
    jy_f = -0.25 * absk  * np.exp(-absk * np.abs(yv)) + 0.0j
    jx_f += jx_omega * 0.5 * k
    jy_f += jy_omega * 0.5 * k

    jx_j = jx_D  + 0.0j
    jy_j = jy_D + 0.0j
    jx_j += J_0 * jx_omega * sgn_k
    jy_j += J_0 * jy_omega * sgn_k

    if False:
        pl.figure()
        pl.plot(yv, jx_D.imag, label='jx_D')
        pl.plot(yv, jx_D_exact.imag, '--', label='exact')
        pl.plot(yv, jy_D.real, label='jy_D')
        pl.plot(yv, jy_D_exact.real, '--', label='exact')
        pl.legend()
        pl.figure()
        pl.plot(yv, jx_f.imag, label='jx_f')
        pl.plot(yv, jx_f_exact.imag, '--', label='exact')
        pl.plot(yv, jy_f.real, label='jy_f')
        pl.plot(yv, jy_f_exact.real, '--', label='exact')
        pl.legend()

        pl.show()


    rho_f[0:i_y_0] = 0.0
    rho_j[0:i_y_0] = 0.0
    jx_f[0:i_y_0] = 0.0
    jx_j[0:i_y_0] = 0.0
    jy_f[0:i_y_0] = 0.0
    jy_j[0:i_y_0] = 0.0

    exp_zyh_up_sgn = exp_zyh_up * sgn_yh[:, None]
    rho_sin_up  = -1j * z_up/(z_up**2 + k**2) * exp_zyh_up_sgn
    rho_sin_up += 0.5 / (absk - 1j * z_up)*exp_kh/Xrho_0 * Xrho_p_up * exp_zy_up
    rho_cos_up  = -1j * k/(z_up**2 + k**2) * exp_zyh_up
    rho_cos_up += 0.5  * 1j * sgn_k / (absk - 1j * z_up)*exp_kh/Xrho_0 * Xrho_p_up * exp_zy_up

    rho_sin = c_up_int(rho_sin_up) / 2.0 / np.pi
    rho_cos = c_up_int(rho_cos_up) / 2.0 / np.pi
    
    rho_sin[0:i_y_0-1] = 0.0
    rho_cos[0:i_y_0-1] = 0.0

    th_y = 1.0 + 0.0 * yv
    th_y[0:i_y_0] = 0.0
    abs_yh = np.abs(yv - h)
    #exp_syh = np.exp(-np.outer(abs_yh, s))
    #exp_sy = np.exp(-np.outer(np.abs(yv), s))
    
    if False:
       pl.figure()
       pl.plot(yv, rho_sin.real, label='Re rho sin O')
       pl.plot(yv, rho_sin.imag, label='Im ')
       pl.plot(yv, rho_cos.real, label='Re rho cos O')
       pl.plot(yv, rho_cos.imag, label='Im O')
       pl.legend()
       pl.show()

    Cj = 0.5 / gamma / 2.0 / np.pi
    jx_cos_a = c_up_int((1.0/Xomega_up - 1.0)*exp_zyh_up/(z_up**2 + k**2)*(-z_up)*(-z_up)) * Cj
    jy_cos_a = c_up_int((1.0/Xomega_up - 1.0)*exp_zyh_up_sgn/(z_up**2 + k**2)*(-z_up)*k) * Cj
    jx_sin_a = c_up_int((1.0/Xomega_up - 1.0)*exp_zyh_up_sgn/(z_up**2 + k**2)*k*(-z_up)) * Cj
    jy_sin_a = c_up_int((1.0/Xomega_up - 1.0)*exp_zyh_up/(z_up**2 + k**2)*k*k) * Cj
    # cos jx:  q^2
    # cos jy:  -q * k
    # sin jx: - k * q
    # sin jy: k^2
    
    if False:
        pl.figure()
        pl.plot(yv, jx_cos_a.real, '-', label='Re jx cos a')
        pl.plot(yv, jx_cos_a.imag, '-', label='Im jx cos a')
        pl.plot(yv, jy_cos_a.real, '-', label='Re jy cos a')
        pl.plot(yv, jy_cos_a.imag, '-', label='Im jy cos a')
        pl.legend()
        pl.figure()
        pl.plot(yv, jx_sin_a.real, '-', label='Re jx sin a')
        pl.plot(yv, jx_sin_a.imag, '-', label='Im jx sin a')
        pl.plot(yv, jy_sin_a.real, '-', label='Re jy sin a')
        pl.plot(yv, jy_sin_a.imag, '-', label='Im jy sin a')
        pl.legend()
        pl.show()

    jx_cos_b = 0.5/gamma * c_up_int((-z_up) * k/(z_up**2 + k**2)*(1.0 - 1.0/Xo_0) * Xomega_p_up * exp_zy_up) * exp_kh / 2.0 / np.pi * 1j * sgn_k
    jx_sin_b = 0.5/gamma * c_up_int(k * (-z_up)/(z_up**2 + k**2)*(1.0 - 1.0/Xo_0) * Xomega_p_up*exp_zy_up) * exp_kh / 2.0 / np.pi 
    jy_cos_b = 0.5/gamma * c_up_int(k * k/(z_up**2 + k**2)*(1.0 - 1.0/Xo_0) * Xomega_p_up*exp_zy_up) * exp_kh / 2.0 / np.pi * 1j * sgn_k
    jy_sin_b = 0.5/gamma * c_up_int(k * k/(z_up**2 + k**2)*(1.0 - 1.0/Xo_0) * Xomega_p_up * exp_zy_up) * exp_kh  / 2.0 / np.pi 
    

    if False:
       pl.figure()
       pl.plot(yv, jx_sin_b.real, label='Re jx sin b')
       pl.plot(yv, jx_sin_b.imag, label='Im')
       pl.plot(yv, jy_sin_b.real, label='Re jy sin b')
       pl.plot(yv, jy_sin_b.imag, label='Im')
       pl.legend()
       pl.figure()
       pl.plot(yv, jx_cos_b.real, label='Re jx cos b')
       pl.plot(yv, jx_cos_b.imag, label='Im')
       pl.plot(yv, jy_cos_b.real, label='Re jy cos b')
       pl.plot(yv, jy_cos_b.imag, label='Im')
       pl.legend()
       pl.show()

    
    def Jpsi_cos(z):
        return c_down_int(1.0/(z - z_down)/(z_down + 1j * absk)\
                         *(1.0/Xomega_m_down - 1.0)*np.exp(-1j*h*z_down)\
                         * (-z_down)) / 2.0/np.pi/1j
    def Jpsi_sin(z):
        return c_down_int(1.0/(z - z_down)/(z_down + 1j * absk) \
                         *(1.0/Xomega_m_down - 1.0)*np.exp(-1j*h*z_down) \
                         * (k) ) /2.0/np.pi/1j
    Jpsi_cos_up = np.vectorize(lambda z: Jpsi_cos(z))(z_up)
    Jpsi_sin_up = np.vectorize(lambda z: Jpsi_sin(z))(z_up)

    jx_cos_d = c_up_int(1.0/(z_up - 1j * k) * (-z_up) * Jpsi_cos_up * Xomega_p_up * exp_zy_up) / 2.0 / np.pi * 0.5 / gamma
    jx_sin_d = c_up_int(1.0/(z_up - 1j * k) * (-z_up) * Jpsi_sin_up * Xomega_p_up * exp_zy_up) / 2.0 / np.pi * 0.5 / gamma
    jy_cos_d = c_up_int(1.0/(z_up - 1j * k) * (k) * Jpsi_cos_up * Xomega_p_up * exp_zy_up) / 2.0 / np.pi * 0.5 / gamma
    jy_sin_d = c_up_int(1.0/(z_up - 1j * k) * (k) * Jpsi_sin_up * Xomega_p_up * exp_zy_up) / 2.0 / np.pi * 0.5 / gamma

   
    if False:
       pl.figure()
       pl.plot(yv, jx_sin_d.real, label='Re jx sin d')
       pl.plot(yv, jx_sin_d.imag, label='Im')
       pl.plot(yv, jy_sin_d.real, label='Re jy sin d')
       pl.plot(yv, jy_sin_d.imag, label='Im')
       pl.legend()
       pl.figure()
       pl.plot(yv, jx_cos_d.real, label='Re jx cos d')
       pl.plot(yv, jx_cos_d.imag, label='Im')
       pl.plot(yv, jy_cos_d.real, label='Re jy cos d')
       pl.plot(yv, jy_cos_d.imag, label='Im')
       pl.legend()
       pl.show()

    jx_cos = jx_cos_a - jx_cos_b + jx_cos_d
    jy_cos = jy_cos_a - jy_cos_b + jy_cos_d
    jx_sin = jx_sin_a - jx_sin_b + jx_sin_d
    jy_sin = jy_sin_a - jy_sin_b + jy_sin_d

    if False:
        pl.figure()
        pl.plot(yv, jx_cos.real, label='Re jx new cos')
        pl.plot(yv, jx_cos.imag, label='Im jx new cos')
        pl.plot(yv, jy_cos.real, label='Re jy new cos')
        pl.plot(yv, jy_cos.imag, label='Im jy new cos')
        #pl.plot(yv, jx_cos.real, '--', label='Re jx cos a')
        #pl.plot(yv, jx_cos.imag, '--', label='Im jx cos a')
        #pl.plot(yv, jy_cos.real, '--', label='Re jy cos a')
        #pl.plot(yv, jy_cos.imag, '--', label='Im jy cos a')
        pl.legend()
        pl.figure()
        pl.plot(yv, jx_sin.real, label='Re jx new sin')
        pl.plot(yv, jx_sin.imag, label='Im jx new sin')
        pl.plot(yv, jy_sin.real, label='Re jy new sin')
        pl.plot(yv, jy_sin.imag, label='Im jy new sin')
        #pl.plot(yv, jx_sin.real, '--', label='Re jx sin a')
        #pl.plot(yv, jx_sin.imag, '--', label='Im jx sin a')
        #pl.plot(yv, jy_sin.real, '--', label='Re jy sin a')
        #pl.plot(yv, jy_sin.imag, '--', label='Im jy sin a')
        pl.legend()
        pl.show()
            
    jx_sin[0:i_y_0-1] = 0.0
    jx_cos[0:i_y_0-1] = 0.0
    jy_sin[0:i_y_0-1] = 0.0
    jy_cos[0:i_y_0-1] = 0.0

    
    result = WHData()
    result.h = h
    result.delta = delta
    result.k = k
    result.src = np.exp(-absk * np.abs(yv - h))
    #result.rho_s = rho_f
    result.bc_phi_s = phi_f
    result.bc_phi_j = phi_j
    result.bc_phi_sin = flux_sin
    result.bc_phi_cos = flux_cos
    result.rho_j = rho_j
    result.drho_j = drho_j
    result.rho_s = rho_f
    #result.omega_s = omega_f
    #result.omega_j = omega_j
    result.rho_sin = rho_sin
    result.rho_cos = rho_cos
    result.jx_s = jx_f
    result.jy_s = jy_f
    result.jx_j = jx_j
    result.jy_j = jy_j
    result.jx_sin = jx_sin
    result.jy_sin = jy_sin
    result.jx_cos = jx_cos
    result.jy_cos = jy_cos
    return result

if __name__ == '__main__':
    #kvals = np.linspace(0.03, 30.0, 2998)
    #kvals = np.linspace(0.01, 30.0, 3000)
    #kvals = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    #kvals = np.linspace(0.01, 1.0, 100)

    if True:
       #kvals = quad_grid()
       #kvals = exp_grid(0.001, 30.0, 500)[250:]
       kvals = np.array([0.1])
       #list(np.linspace(0.001, 0.009, 5))
       #kvals.extend(np.linspace(0.01, 0.99, 99))
       #kvals.extend(np.linspace(1.0, 10.0, 361))
       #kvals.extend(np.linspace(10.1, 30.0, 200))
    else:
       kvals = list(np.linspace(0.001, 0.099, 99))
       kvals.extend(np.linspace(0.100, 0.995, 180))
       kvals.extend(np.linspace(1.0, 9.98, 450))
       kvals.extend(np.linspace(10.0, 30.0, 401))

    kvals = np.array(kvals)

    h = 0.0
    delta = 0.005
    #h = 0.0
    delta = 0.0001
    #kvals = np.array(kvals)
    #kvals = np.array([0.1, 0.5, 1.0, 3.0, 10.0, 20.0, 30.0])
    #kvals = np.array([1.0, 2.0, 20.0, 10.0, 3.0, 1.0, 0.5, 0.1])
    #kvals = np.array([0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0])
    #print "solve for", kvals
    yv = np.linspace(-1.0, 10.0, 1101)
    jx_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    jy_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    jx_sin_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    jy_sin_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    jx_cos_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    jy_cos_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    rho_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    rho_sin_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    rho_cos_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    drho_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    src_ky = np.zeros((len(kvals), len(yv)), dtype=complex)
    fs_k = 0.0 * kvals + 0.0 * 1j
    f_sin_k = 0.0 * kvals + 0.0 * 1j
    f_cos_k = 0.0 * kvals + 0.0 * 1j
    
    phi_sk = 0.0 * fs_k
    phi_sin_k = 0.0 * fs_k
    phi_cos_k = 0.0 * fs_k
    phi_jk = 0.0 * fs_k
    
    def examine_quantity(name, q, X, re_fit_func, im_fit_func):
        from scipy import optimize
        fit_qmax = 2.0 * np.abs(k)
        i_fit = [t for t in range(len(q)) if np.abs(q[t]) < fit_qmax]
        if len(i_fit) == 0: return
        q_fit = np.array([q[t] for t in i_fit])
        Xre_fit = np.array([X[t].real for t in i_fit])
        Xim_fit = np.array([X[t].imag for t in i_fit])

        re_fit, re_cov = optimize.curve_fit(re_fit_func, q_fit, Xre_fit)
        re_fit_vals = re_fit_func(q_fit, *re_fit)
        print("*** fit for Re ", name, re_fit_func)
        print("   results: ", re_fit, re_cov)
        im_fit, im_cov = optimize.curve_fit(im_fit_func, q_fit, Xim_fit)
        im_fit_vals = im_fit_func(q_fit, *im_fit)
        print("*** fit for Im ", name, im_fit_func)
        print("   results: ", im_fit, im_cov)
        if False:
            pl.figure()
            pl.title("final fits for %s" % name)
            pl.plot(q, X.real, label='Re %s' % name)
            pl.plot(q, X.imag, label='Im %s' % name)
            pl.plot(q_fit, re_fit_vals, '--', label='Re fit %s' % name)
            pl.plot(q_fit, im_fit_vals, '--', label='im fit %s' % name)
            pl.xlim(-1.5*fit_qmax, 1.5*fit_qmax)
            pl.legend()
        print()
        return re_fit, im_fit

    def interim_save():
        np.savez("wh-data16-h=%g-delta=%g-kmin=%g-kmax=%g" % (h, delta,
                                        np.min(kvals), np.max(kvals)),
                 h=h, delta=delta, k=kvals, y=yv,
                 jx=jx_ky, jy=jy_ky, src=src_ky,
                 f_s = fs_k, f_sin = f_sin_k, f_cos = f_cos_k,
                 phi_s=phi_sk, phi_j=phi_jk,
                 phi_sin = phi_sin_k, phi_cos = phi_cos_k,
                 rho=rho_ky, rho_sin = rho_sin_ky, rho_cos = rho_cos_ky,
                 jx_sin = jx_sin_ky, jy_sin = jy_sin_ky,
                 jx_cos = jx_cos_ky, jy_cos = jy_cos_ky,
                 drho=drho_ky)
    #f_save = open("lor11a-h=%g-delta=%g-qmax=%g.dat" % (h, delta, q_max), "w")
    #f_save.write("# k\tphi_s\tphi_j\tf_s\tCrho_s\tCrho_j\tComega_s\tComega_j\tComega\tCrho\tDomega\tDrho\n")

    for i_k, k in enumerate(kvals):
        print("k = ", k, "h = ", h, "delta = ", delta)
        result = wh_solve(h, delta, k, yv)
        dphi = (1.0/np.pi - result.bc_phi_s)
        f_s = result.bc_phi_j / dphi
        f_cos = result.bc_phi_cos / dphi
        f_sin = result.bc_phi_sin / dphi
        rho_y = result.rho_j + f_s * result.rho_s
        rho_sin_y = result.rho_sin + f_sin * result.rho_s
        rho_cos_y = result.rho_cos + f_cos * result.rho_s
        drho_y = result.drho_j + f_s * result.rho_s
        jx_y = result.jx_j + f_s * result.jx_s
        jy_y = result.jy_j + f_s * result.jy_s
        jx_sin_y = result.jx_sin + f_sin * result.jx_s
        jy_sin_y = result.jy_sin + f_sin * result.jy_s
        jx_cos_y = result.jx_cos + f_cos * result.jx_s
        jy_cos_y = result.jy_cos + f_cos * result.jy_s

        rho_ky[i_k, :]   = rho_y
        drho_ky[i_k, :]  = drho_y
        jx_ky[i_k, :] = jx_y
        jy_ky[i_k, :] = jy_y
        fs_k[i_k]     = f_s
        f_sin_k[i_k] = f_sin
        f_cos_k[i_k] = f_cos
        phi_jk[i_k] = result.bc_phi_j
        phi_sk[i_k] = result.bc_phi_s
        phi_sin_k[i_k] = result.bc_phi_sin
        phi_cos_k[i_k] = result.bc_phi_cos
        rho_sin_ky[i_k, :] = rho_sin_y
        rho_cos_ky[i_k, :] = rho_cos_y
        jx_sin_ky[i_k, :] = jx_sin_y
        jy_sin_ky[i_k, :] = jy_sin_y
        jx_cos_ky[i_k, :] = jx_cos_y
        jy_cos_ky[i_k, :] = jy_cos_y
        src_ky[i_k] = result.src
        print("f_s = ", f_s, "f_sin = ", f_sin, "f_cos = ", f_cos)

        if False:
            pl.figure()
            pl.plot(yv, jy_y.real, label='j_y')
            pl.plot(yv, result.jy_j.real, '--', label='jy_j')
            pl.plot(yv, result.jy_s.real, '--', label='jy_s')
            pl.show()
        #examine_quantity("drho", result.q, drho_q, lorentzian_re, lorentzian_im)
        #fit_rho = examine_quantity("rho", result.q,  drho_q + Fm(k, result.q)*2.0, lorentzian_re, lorentzian_im)
        #examine_quantity("rho", result.q,  drho_q + Fm(k, result.q)*2.0, lorentzian2_re, lorentzian2_im)
        #examine_quantity("j_x",  result.q, jx_q, lorentzian2_im, lorentzian2_re)
        #examine_quantity("j_y",  result.q, jy_q, lorentzian2_re, lorentzian2_im)
        #fit_omega = examine_quantity("omega", result.q, omega_q, lorentzian_re, lorentzian_im);
        #fit_rho_s = examine_quantity("rho_s", result.q, result.rho_s, lorentzian_re, lorentzian_im)
        #fit_rho_j = examine_quantity("rho_j", result.q, result.rho_j + 2*Fm(k, result.q),
        #                             lorentzian_re, lorentzian_im)
        #fit_omega_s = examine_quantity("omega_s", result.q, result.omega_s, lorentzian_re, lorentzian_im)
        #fit_omega_j = examine_quantity("omega_j", result.q, result.omega_j, lorentzian_re, lorentzian_im)

        #pl.show()
        #f_save.write("# k phi_s phi_j f_s Crho_s Crho_j Comega_s Comega_j Comega Crho Domega Drho\n")
        #f_save.write("%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n" % (k,
        #                               result.bc_phi_s.real, result.bc_phi_j.real,
        #                               f_s.real, fit_rho_s[0][0],
        #                               fit_rho_j[0][0], fit_omega_s[0][0],
        #                               fit_omega_j[0][0], fit_omega[0][0],
        #                               fit_rho[0][0], fit_omega[0][1], fit_rho[0][1]))
        #f_save.flush()
        if (i_k > 0 and i_k % 10 == 0) or (i_k == 3):
            interim_save()

    interim_save()
    #f_save.close()
