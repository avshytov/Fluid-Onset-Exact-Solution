import numpy as np
import path
import xkernel_new as xkernel
from cauchy import cauchy_integral, cauchy_integral_array

def make_arc(k, kappa):
    abs_k = np.abs(k)
    a = np.sqrt(abs_k * kappa)
    b = kappa - abs_k / 2.0
    coarse = False
    if not coarse:
        #N_arc  = 101
        N_arc = 201
        N_vert = 500
    else:
        N_arc = 25
        N_vert = 300
    #N_arc = 51
    #N_vert = 401
    # Use quadractic scaling to add more points near the real axis,
    # i.e. near 3/2 pi
    path_arc  = path.ArcPath(1j * kappa, a, b, 1.5*np.pi, np.pi, N_arc,
                             lambda t: t * t)
    z_inf = 100.0j * kappa - 0.1 * a # the "infinity" point

    # Use t * sqrt(t) scaling to add more points at smaller q
    path_vert = path.StraightPath(path_arc.ends_at(), z_inf, N_vert,
                               lambda t: t*np.sqrt(np.abs(t)))
    path_up_left  = path.append_paths(path_arc, path_vert)
    path_up_left.reverse()
    return path_up_left

def make_contours(k, kappa):
    path_ul = make_arc(k, kappa)
    path_ru = path.transform(path_ul, lambda z: complex(-z.real, z.imag))
    path_ur = path.reverse(path_ru)
    path_up = path.append_paths(path_ul, path_ur)
    path_dn = path.transform(path_up, lambda z: complex(z.real, -z.imag))
    return path_up, path_dn

def make_paths_and_kernels(K, k):
    kappa = np.sqrt(k**2 + K.gamma**2)
    path_up, path_dn = make_contours(k, kappa)
    # Do Cauchy integrals:
    z_up = path_up.points()
    log_Krho_p = cauchy_integral(path_dn,
                                 lambda z: np.log(K.rho(k, z)), z_up)
    z_be = 0.5 * (path_dn.begins_at() + path_dn.ends_at())
    log_corr   = 1j / np.pi / z_up * np.log(z_be/(z_be - z_up))
    log_Krho_p   += K.gamma * log_corr
    log_Komega_p  = cauchy_integral(path_dn,
                                    lambda z: np.log(K.omega(k, z)), z_up)
    log_Komega_p += 2.0 * K.gamma1 * log_corr
    Krho_p_up     = np.exp(-log_Krho_p)
    Komega_p_up   = np.exp(-log_Komega_p)

    # Obtain the values at the lower contour by the
    # up-down symmetry
    z_dn = path_dn.points()
    Krho_dn     = K.rho(k,   z_dn)
    Komega_dn   = K.omega(k, z_dn)
    Krho_m_dn   = 1.0 / Krho_p_up.conj()
    Komega_m_dn = 1.0 / Komega_p_up.conj()
    Krho_p_dn   = Krho_m_dn / Krho_dn
    Komega_p_dn = Komega_m_dn / Komega_dn
    K_up = xkernel.TabulatedKernels(K, k, z_up, Krho_p_up, Komega_p_up)  
    K_dn = xkernel.TabulatedKernels(K, k, z_dn, Krho_p_dn, Komega_p_dn)
    return path_up, K_up, path_dn, K_dn
    
def append_paths_and_kernels(path_a, K_a, path_b, K_b):

    if abs(K_a.k - K_b.k) > 1e-6:
        raise Exception("k do not match")
    if abs(K_a.gamma - K_b.gamma) > 1e-6:
        raise Exception("gammas do not match")
    if abs(K_a.gamma1 - K_b.gamma1) > 1e-6:
        raise Exception("gammas do not match")
    
    path_joint = path.append_paths(path_a, path_b)
    q_joint = path_joint.points()
    Krho_p   = 0.0 * q_joint + 0.0j
    Komega_p = 0.0 * q_joint + 0.0j
    Krho_p[0:len(path_a.points())] = K_a.Krho_p
    Krho_p[-len(path_b.points()):] = K_b.Krho_p
    Komega_p[0:len(path_a.points())] = K_a.Komega_p
    Komega_p[-len(path_b.points()):] = K_b.Komega_p
    K_joint = xkernel.TabulatedKernels(K_a.K, K_a.k, q_joint,
                                       Krho_p, Komega_p)
    return path_joint, K_joint

def reverse_path_and_kernel(pth, K):
    path_new = path.reverse(pth)
    Krho_p = list(K.Krho_p)
    Komega_p = list(K.Komega_p)
    Krho_p.reverse()
    Komega_p.reverse()
    Krho_p = np.array(Krho_p)
    Komega_p = np.array(Komega_p)
    Knew = xkernel.TabulatedKernels(K.K, K.k,
                                  path_new.points(), Krho_p, Komega_p)
    return path_new, Knew

def conjugate_left_right(pth, K):
    path_c = path.transform(pth, lambda z: complex(-z.real, z.imag))
    Krho_p = K.Krho_p.conj()
    Komega_p = K.Komega_p.conj()
    Knew = xkernel.TabulatedKernels(K.K, K.k,
                                    path_c.points(), Krho_p, Komega_p)
    return path_c, Knew

def conjugate_up_down(pth, K):
    path_c     = path.transform(pth, lambda z: complex(z.real, -z.imag))

    # When z is changed to z_bar, X+ becomes X_- conjugate
    Krho_c       = K.K.rho(K.k,   path_c.points())
    Komega_c     = K.K.omega(K.k, path_c.points())

    Krho_p     = K.Krho_p
    Komega_p   = K.Komega_p

    Krho_m_new   =  1.0/Krho_p.conj()
    Komega_m_new = 1.0 / Komega_p.conj()
    Krho_p_new   = Krho_m_new / Krho_c
    Komega_p_new = Komega_m_new / Komega_c
    #Krho_m     = Krho * Krho_p
    #Komega_m   = Komega * Komega_p

    #Krho_new   = Krho_m.conj()
    #Komega_new = Komega_m.conj()

    Knew = xkernel.TabulatedKernels(K.K, K.k, path_c.points(),
                                    Krho_p_new, Komega_p_new)
    return path_c, Knew


def make_contours_and_kernels(k, gamma, gamma1):
    kappa = np.sqrt(gamma**2 + k**2)
    #path_up, path_dn = make_contours_im(abs(k), kappa)
    path_ul = make_arc(abs(k), kappa)
    # upper left segment, used to construct the rest 
    
    K = xkernel.WHKernels(gamma, gamma1)
    print ("tabulate kernels up")
    import time; now = time.time()
    K_ul = xkernel.tabulate_kernel(K, k, path_ul.points())
    print ("tabulation done, ", time.time() - now)
    #import sys; sys.exit(0)
    #K_ul = xkernel.load_kernel(K, k, path_ul.points(), "ul", False)
    # get upper-right segment
    path_ru, K_ru = conjugate_left_right(path_ul, K_ul) # backward
    path_ur, K_ur = reverse_path_and_kernel(path_ru, K_ru)
    
    path_up, K_up = append_paths_and_kernels(path_ul, K_ul, path_ur, K_ur)
    path_dn, K_dn = conjugate_up_down(path_up, K_up)

    #util.show_paths(path_up, path_dn,
    #           0.5*(path_up.points() + path_dn.points()))
    #import pylab as pl; pl.show()
    return path_up, K_up, path_dn, K_dn
