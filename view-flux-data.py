import numpy as np
import pylab as pl
from scipy import special
import sys

pl.figure()
ax_all_sin = pl.gca()
pl.title("All sin data")

pl.figure()
ax_all_cos = pl.gca()
pl.title("All cos data")


first_run = True
#d = np.load("flux-data.npz")
for fname in sys.argv[1:]:
    d = np.load(fname)
    print(list(d.keys()))

    k = d['k']
    flux_sin = d['flux_sin']
    flux_cos = d['flux_cos']
    flux_sin_fourier = d['flux_sin_fourier']
    flux_cos_fourier = d['flux_cos_fourier']
    flux_sin_im = d['flux_sin_im']
    flux_cos_im = d['flux_cos_im']
    Xrho_0 = d['Xrho_0']
    Xo_0 = d['Xo_0']
    Xo_reg = d['Xo_reg_exact']
    Xo_prime = d['Xomega_prime_exact']
    h = d['h']
    delta_q = d['delta_q']
    #h = 0.5

    ax_all_sin.plot(k, flux_sin.real, label=fname)
    ax_all_cos.plot(k, flux_cos.imag, label=fname)
    pl.figure()
    pl.title(fname)
    pl.plot(k, flux_sin.real, label='Re F sin')
    #pl.plot(k, flux_sin.imag, label='Im F sin')
    #pl.plot(k, flux_cos.real, label='Re F cos')
    pl.plot(k, flux_cos.imag, label='Im F cos')
    k_large = np.linspace(3.0, 10.0, 35)
    pl.plot(k, flux_sin_im.real, '--', label='Re F sin im')
    #pl.plot(k, flux_sin_im.real*0.5, '-', label='Re F sin im')
    #pl.plot(k, flux_sin_im.imag, label='Im F sin')
    #pl.plot(k, flux_cos_im.real, label='Re F cos')
    pl.plot(k, flux_cos_im.imag, '--', label='Im F cos im')
    #pl.plot(k, flux_sin_im.real - flux_cos_im.imag, label='diff')
    pl.plot(k_large, -2.0 * k_large * h / np.pi * special.kn(1, k_large * h), label='k*K1')
    pl.plot(k_large, -2.0 * k_large * h / np.pi * special.kn(0, k_large * h), label='k*K0')

    pl.plot(k, flux_sin_fourier.real, 'o', label='Re F sin f')
    #pl.plot(k, flux_sin_fourier.imag, label='Im F sin')
    #pl.plot(k, flux_cos_fourier.real, label='Re F cos')
    pl.plot(k, flux_cos_fourier.imag, 'o', label='Im F cos f')
    pl.legend()

    pl.figure()
    pl.title(fname)
    pl.plot(k, flux_sin_im.real - flux_sin.real, label='diff sin')
    pl.plot(k, flux_cos_im.imag - flux_cos.imag, label='diff cos')
    #pl.plot(k, - 0.25 * k * (1.0/Xo_0**2) * np.exp(-k*h))
    #pl.figure()
    #pl.plot(k, flux_sin_im.real - flux_cos_im.imag, label='diff')
    pl.legend()

    if first_run:
        pl.figure()
        pl.title(fname)
        pl.plot(k, flux_sin.real / flux_sin_im.real, label='ratio sin')
        pl.plot(k, flux_cos.imag / flux_cos_im.imag, label='ratio cos')
        pl.legend()
        #pl.figure()
        #pl.plot(k, flux_sin_im.real - flux_cos_im.imag, label='diff')

    if first_run:
        pl.figure()
        pl.title(fname)
        pl.semilogy(k, np.abs(flux_sin.real), label='Re F sin')
        #pl.semilogy(k, flux_sin.imag, label='Im F sin')
        #pl.semilogy(k, flux_cos.real, label='Re F cos')
        pl.semilogy(k, np.abs(flux_cos.imag), label='Im F cos')
        pl.semilogy(k, np.abs(flux_sin_im.real), '--', label='Re F sin im')
        pl.semilogy(k, np.abs(flux_cos_im.imag), '--', label='Im F cos im')
        pl.legend()

    if first_run:
       pl.figure()
       pl.title(fname)
       pl.plot(k, Xrho_0, label='Xrho_0')
       pl.plot(k, Xo_0, label='Xomega_0')
       pl.legend()
    first_run = False

for ax in ax_all_sin, ax_all_cos:
    ax.legend()

    
pl.show()
