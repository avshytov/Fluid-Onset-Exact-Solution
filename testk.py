import xkernel_new as xkernel
import numpy as np
import pylab as pl
from scipy import linalg
from scipy import integrate

k = 0.2
gamma = 1.0
a = 1.0
da = 0.0001

kappa = np.sqrt(gamma**2 + k**2)
kappa1 = np.sqrt(kappa**2 - a**2)

K = xkernel.WHKernel(gamma, a)
Kp = xkernel.WHKernel(gamma, a + da)
Km = xkernel.WHKernel(gamma, a - da)

svals = np.linspace(kappa1, kappa, 101)[1:-1]
Kvals = np.vectorize(lambda s: K.plus(k, 1j * s))(svals)
log_Kvals = np.log(Kvals)

Kp_vals = np.vectorize(lambda s: Kp.plus(k, 1j * s))(svals)
Km_vals = np.vectorize(lambda s: Km.plus(k, 1j * s))(svals)

dlog_K = (np.log(Kp_vals) - np.log(Km_vals))/2.0/da


def arctanh1(x, a):
    lg =  np.log(np.abs(a + x) / np.abs(a - x)) * 0.5
    #if np.abs(x) < np.abs(a):
    #    return lg
    if x > np.abs(a):
        lg +=  np.pi/2 * 1j 
    if x < -np.abs(a):
        lg -= -np.pi/2 * 1j
    return lg
    #return np.log(np.abs(a + x)/np.abs(a - x)) * 0.5

def dlog_K_da(s, b, c1 = 1, c2 = 1, c3 = 1):
    kappa_b = np.sqrt(kappa**2 - b**2)
    K1 = b / 2.0 / kappa_b / (s + kappa_b)
    K2 = np.sqrt(kappa**2 - s**2) / np.pi / (kappa_b**2 - s**2) * np.arccos(s/kappa)
    K3 = -s * b / np.pi / (kappa_b**2 - s**2)/kappa_b * np.arcsin(b/kappa)
    return c1 * K1 + c2 * K2 + c3 * K3

dlog_Kvals_ex = np.vectorize(lambda s: dlog_K_da(s, a))(svals)
#dlog_Kvals_ex1 = np.vectorize(lambda s: dlog_K_da(s, a, 1, 0, 0))(svals)
#dlog_Kvals_ex2 = np.vectorize(lambda s: dlog_K_da(s, a, 0, 1, 0))(svals)
#dlog_Kvals_ex12 = np.vectorize(lambda s: dlog_K_da(s, a, 1, 1, 0))(svals)
#dlog_Kvals_ex2 = np.vectorize(lambda s: dlog_K_da(s, a, 0, 1, 0))(svals)
#dlog_Kvals_ex3 = np.vectorize(lambda s: dlog_K_da(s, a, 0, 0, 1))(svals)

def log_Kplus_exact(s, b):
    kappa_b = np.sqrt(kappa**2 - b**2)
    K1 = 0.5 * np.log((s + kappa) / (s + kappa_b))
    print ("a/sqrt(kappa**2 - s**2)", b/np.sqrt(kappa**2 - s**2))
    K2 = 1.0/np.pi * np.arccos(s/kappa) * arctanh1(b, np.sqrt(kappa**2 - s**2))
    K3 = -1.0/np.pi * arctanh1(s, kappa_b) * np.arcsin(b/kappa)
    def f_K(x):
        kappa_x = np.sqrt(kappa**2 - x**2)
        return arctanh1(s, kappa_x) / kappa_x #np.arctanh(kappa_x / s) / kappa_x
    def f_re(x):
        return f_K(x).real
    def f_im(x):
        return f_K(x).imag
    I_re, eps_re = integrate.quad(f_re, 0, b)
    I_im, eps_im = integrate.quad(f_im, 0, b)
    I = I_re + 1j * I_im
    K4 = I / np.pi
    print ("s = ", s, "K = ", K1, K2, K3, K4)
    return 1 * K1 + 1 * K2 + 1 * K3 + 1 * K4

log_Kvals_ex = np.vectorize(lambda s: log_Kplus_exact(s, a))(svals)
log_Kvals_ex_p = np.vectorize(lambda s: log_Kplus_exact(s, a + da))(svals)
log_Kvals_ex_m = np.vectorize(lambda s: log_Kplus_exact(s, a - da))(svals)
dlog_K2 = (log_Kvals_ex_p - log_Kvals_ex_m) / 2.0 / da

pl.figure()
pl.plot(svals, dlog_K.real, label='Re d log K(s)/da num')
pl.plot(svals, dlog_K.imag, label='Im')
pl.plot(svals, dlog_Kvals_ex.real, '--', label='Re dK exact')
pl.plot(svals, dlog_Kvals_ex.imag, '--', label='Im')
pl.plot(svals, dlog_K2.real, '--', label='Re num d(exact)')
pl.plot(svals, dlog_K2.imag, '--', label='Im')
pl.legend()

pl.figure()
pl.plot(svals, log_Kvals.real, label='Re K(s)')
pl.plot(svals, log_Kvals.imag, label='Im K(s)')
pl.plot(svals, log_Kvals_ex.real, '--', label='Re exact')
pl.plot(svals, log_Kvals_ex.imag, '--', label='Im exact')
pl.legend()

#pl.figure()
#pl.plot(svals, dlog_Kvals_ex12.real, label='Re d log K(s)/da 12')
#pl.plot(svals, dlog_Kvals_ex12.imag, label='Im d log K(s)/da 12')
#print ("dlog_K2 = ", dlog_K2)
#pl.plot(svals, dlog_K2.real, '--', label='Re d log K(s)/da from ex')
#pl.plot(svals, dlog_K2.imag, '--', label='Im d log K(s)/da from ex')
#pl.legend()

#pl.figure()
#pl.plot(svals, dlog_Kvals_ex3.real, label='Re d log K(s)/da 3')
#pl.plot(svals, dlog_Kvals_ex3.imag, label='Im d log K(s)/da 3')
#print ("dlog_K2 = ", dlog_K2)
#pl.plot(svals, dlog_K2.real, '--', label='Re d log K(s)/da from ex')
#pl.plot(svals, dlog_K2.imag, '--', label='Im d log K(s)/da from ex')
#pl.plot(svals, dlog_K2.real - dlog_Kvals_ex3.real, '--', label='Re diff')
#pl.legend()

pl.figure()
pl.plot(svals, dlog_Kvals_ex.real, label='Re d log K(s)/da eq')
pl.plot(svals, dlog_Kvals_ex.imag, label='Im d log K(s)/da eq')
print ("dlog_K2 = ", dlog_K2)
pl.plot(svals, dlog_K2.real, '--', label='Re d log K(s)/da num-eq')
pl.plot(svals, dlog_K2.imag, '--', label='Im d log K(s)/da num-eq')
pl.legend()


pl.show()
