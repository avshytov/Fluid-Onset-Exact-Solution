import numpy as np
import scipy
from scipy import integrate

gamma = 1.0

def Frho_direct(k, q):
    def f(theta):
        sn = np.sin(theta)
        cs = np.cos(theta)
        k_v = k * cs + q * sn
        return sn / (gamma + 1j * k_v)
    def f_re(theta):
        return f(theta).real
    def f_im(theta):
        return f(theta).imag
    I_re, eps_re = integrate.quad(f_re, 0.0, 1.0 * np.pi, limit=1000)
    I_im, eps_im = integrate.quad(f_im, 0.0, 1.0 * np.pi, limit=1000)
    return (I_re + 1j * I_im) / 2.0 / np.pi

def Fomega_direct(k, q):
    def f(theta):
        sn = np.sin(theta)
        cs = np.cos(theta)
        k_v = k * cs + q * sn
        return sn * (k * sn - q * cs) / (gamma + 1j * k_v)
    def f_re(theta):
        return f(theta).real
    def f_im(theta):
        return f(theta).imag
    I_re, eps_re = integrate.quad(f_re, 0.0, 1.0 * np.pi, limit=1000)
    I_im, eps_im = integrate.quad(f_im, 0.0, 1.0 * np.pi, limit=1000)
    return (I_re + 1j * I_im) / 2.0 / np.pi

def Fm_analytic(k, q):
    sq = np.sqrt(gamma**2 + k**2 + q**2)
    sq0 = np.sqrt(k*k + gamma * gamma)
    return (np.pi - 2.0j * np.log((q + sq)/sq0))/sq / 2.0 / np.pi


def Frho_analytic(k, q):
    sq = np.sqrt(q**2 + k**2 + gamma**2)
    return 1.0/(k**2 + q**2) * ((2.0 * k * np.arctan(k/gamma) - 1j * np.pi * q)/2.0/np.pi + 1j * gamma * q * Fm(k, q))
    #return 4.0 / (k**2 + q**2)  * (k * np.arctan(k/gamma) + q * gamma / 2.0 / sq * np.log((q + sq) / (sq - q)))

def Fomega_analytic(k, q):
    fm = Fm(k, q)
    absk2 = k**2 + q**2
    #sq = np.sqrt(k**2 + q**2 + gamma**2)
    f1 = k * (1 + gamma**2 / absk2) * fm
    f2 = - np.pi * gamma * k / absk2 / 2.0 / np.pi
    f3 = + 2.0 * 1j * gamma * q / absk2 * np.arctan(k/gamma) / 2.0 / np.pi
    return (f1 + f2 + f3)

Fomega = Fomega_analytic
Frho = Frho_analytic
Fm = Fm_analytic
#def phi_rho_minus(k, q):
if False:
  for k in [0.0, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
    pl.figure()
    pl.title("k = %g" % k)
    qvals = np.linspace(-10.0, 10.0, 1000)
    #yvals = np.vectorize(lambda k: Frho_direct(k, q))(kvals)
    #y0vals = np.vectorize(lambda k: Frho_analytic(k, q))(kvals)
    yvals = np.vectorize(lambda qt: Fomega_direct(k, qt))(qvals)
    y0vals = np.vectorize(lambda qt: Fomega_analytic(k, qt))(qvals)
    pl.plot(qvals, yvals.real, 'r-', label='Re I')
    pl.plot(qvals, y0vals.real, 'b--', label='Re F')
    pl.plot(qvals, yvals.imag, 'g-', label='Im I')
    pl.plot(qvals, y0vals.imag, 'k--', label='Im F')
    pl.legend()
  pl.show()


def Frho_star(k):
    taninv = 1.0 / np.pi * np.arctan(np.abs(k)/gamma)
    return 0.5 * (taninv/np.abs(k) + 1.0/gamma/np.pi \
                  - abs(k)/gamma**2 * (0.5 - taninv))

def Fomega_star(k):
    sgn_k = 1.0
    if k < 0: sgn_k = -1.0
    taninv = 1.0 / np.pi * np.arctan(np.abs(k)/gamma)
    return 0.25 * abs(k) /gamma * (1.0 - 2.0 * taninv) \
        + 0.5 *sgn_k * (1/np.pi - gamma/np.abs(k)*taninv)
    
