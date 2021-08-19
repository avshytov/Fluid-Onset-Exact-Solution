import numpy as np
import scipy
from scipy import integrate

def sqrt_z_new(gamma, k, q):
    kappa = np.sqrt(k * k + gamma * gamma)
    #return np.sqrt((kappa  + 1j * q))* np.sqrt((kappa - 1j * q))
    return np.sqrt((kappa  + 1j * q) * (kappa - 1j * q))


def sqrt_z_old(gamma, k, q):
    #sq = np.sqrt((gamma**2 + k**2 + q**2))
    kappa = np.sqrt(k**2 + gamma**2)
    sq_1 = np.abs(kappa - 1j * q)
    sq_2 = np.abs(kappa + 1j * q)
    sq = np.sqrt(sq_1 * sq_2) + 0.0j
    a1 = np.angle(kappa - 1j * q)
    a2 = np.angle(kappa + 1j * q)
    sq *= np.exp(0.5j * (a1 + a2))
    return sq

sqrt_z = sqrt_z_new

def K(gamma, k, q):
    return 1.0 - gamma / sqrt_z(gamma, k, q)

def Fm_analytic(gamma, k, q):
    sq = sqrt_z(gamma, k, q) #np.sqrt(gamma**2 + k**2 + q**2)
    sq0 = np.sqrt(k * k + gamma * gamma)
    return (np.pi + 2.0j * np.log((q + sq)/sq0))/sq / 2.0 / np.pi


def Frho_analytic(gamma, k, q):
    #sq = np.sqrt(q**2 + k**2 + gamma**2)
    sq = sqrt_z(gamma, k, q)
    f1 = (2.0 * k * np.arctan(k/gamma) + 1j * np.pi * q)/2.0/np.pi
    f2 = 1j * gamma * q * Fm(gamma, k, q)
    return 1.0/(k**2 + q**2) * (f1 - f2)
    #return 4.0 / (k**2 + q**2)  * (k * np.arctan(k/gamma) + q * gamma / 2.0 / sq * np.log((q + sq) / (sq - q)))

def Fomega_analytic(gamma, k, q):
    fm = Fm(gamma, k, q)
    absk2 = k**2 + q**2
    #sq = np.sqrt(k**2 + q**2 + gamma**2)
    f1 = k * (1 + gamma**2 / absk2) * fm
    f2 = - np.pi * gamma * k / absk2 / 2.0 / np.pi
    f3 = - 2.0 * 1j * gamma * q / absk2 * np.arctan(k/gamma) / 2.0 / np.pi
    return (f1 + f2 + f3)

Fomega = Fomega_analytic
Frho = Frho_analytic
Fm = Fm_analytic

def Frho_star(gamma, k):
    taninv = 1.0 / np.pi * np.arctan(np.abs(k)/gamma)
    return 0.5 * (taninv/np.abs(k) + 1.0/gamma/np.pi \
                  - abs(k)/gamma**2 * (0.5 - taninv))

def Fomega_star(gamma, k):
    sgn_k = 1.0
    if k < 0: sgn_k = -1.0
    taninv = 1.0 / np.pi * np.arctan(np.abs(k)/gamma)
    return 0.25 * abs(k) /gamma * (1.0 - 2.0 * taninv) \
        + 0.5 *sgn_k * (1/np.pi - gamma/np.abs(k)*taninv)
    
if __name__ =='__main__':
    q_re = np.linspace(-10.0, 10.0, 1001)
    q_im = np.linspace(-10.0, 10.0, 1001)
    gamma = 1.0
    k = 0.7
    X, Y = np.meshgrid(q_re, q_im)
    Q = X + 1j * Y
    K_Q = K(gamma, k, Q)
    Fm_Q = Fm(gamma, k, Q)
    Fm_Qm = Fm(gamma, k, -Q)
    Fm_sum = Fm_Q + Fm_Qm
    #Fm_Q = 0.01 * Q * Q
    Frjp_Q = Frho(gamma, k, Q)
    Fomega_Q = Fomega(gamma, k, Q)
    import util
    import pylab as pl
    util.display_complex(Q, K_Q, 1.0)
    pl.title("K(q)")
    util.display_complex(Q, Fm_Q, 1.0)
    pl.title("Fm(q)")
    util.display_complex(Q, Fm_Qm, 1.0)
    pl.title("Fm(-q)")
    util.display_complex(Q, Fm_sum, 1.0)
    pl.title("Fm +-")
    pl.show()
