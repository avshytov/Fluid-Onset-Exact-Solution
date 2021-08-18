import numpy as np
import pylab as pl
import sys
from whsolve import Fourier

d = np.load(sys.argv[1])
print(list(d.keys()))
k = d['k']
Ck = 2.0
if (np.abs(k[0]) - np.abs(k[-1])) < 0.1:
    Ck = 1.0
print("Ck = ", Ck)
xi = np.linspace(0.0001, 1.0, 5000)
#x = np.linspace(-5.0, 50.0, 5000)
#x = 50.0 * xi * np.sqrt(np.abs(xi))
x = 50.0 * xi * np.abs(xi)
fourier = Fourier(k, x)
#flux_k = d['flux']
flux_k = d['f_s']
density_k = d['rho']
#r0 = d['r0']
src_k = d['src']


pl.figure()
corr =  src_k #np.exp(r0**2/2.0*k**2)
f_k = flux_k * corr
rho_k = density_k * corr
pl.plot(np.abs(k), f_k.real, label='flux_k')
pl.plot(np.abs(k), rho_k.real, label='rho_k')
pl.legend()

kmin = 10.0;
kmax = 20.0;
kmax2 = 0.2
k_min2 = 0.05
i_fit = np.array([t for t in range(len(k)) if k[t] > kmin and k[t] < kmax])
i_fit2 = np.array([t for t in range(len(k))
                   if abs(k[t]) < kmax2 and abs(k[t]) > k_min2])
k_fit = np.array([k[t].real for t in i_fit])
f_fit = np.array([f_k[t].real for t in i_fit])
rho_fit = np.array([rho_k[t].real for t in i_fit])
k_fit2 = np.array([k[t].real for t in i_fit2])
f_fit2 = np.array([f_k[t].real for t in i_fit2])
rho_fit2 = np.array([rho_k[t].real for t in i_fit2])

def linear_fitfunc(k, A, B, C):
    return A + B * abs(k) + C * k**2

def rho_fitfunc(k, A, B, C, D, E, F):
    lnk = np.log(k)
    return A * lnk + B + C / k + D * lnk / k + E / k**2 + F * lnk / k**2

def f_fitfunc(k, A, B, C, D):
    lnk = np.log(k)
    return A  / k + B * lnk / k + C / k**2 + D * lnk / k**2

from scipy import optimize
p_rho, cov_rho = optimize.curve_fit(rho_fitfunc, k_fit, rho_fit)
print("rho_fit: ", p_rho, cov_rho)
p_f, cov_f = optimize.curve_fit(f_fitfunc, k_fit, f_fit)
print("flux_fit: ", p_f, cov_f)
pl.plot(k_fit, rho_fitfunc(k_fit, *p_rho), '--', label='rho fit')
pl.plot(k_fit, f_fitfunc(k_fit, *p_f), '--', label='f fit')

p_rho_lin, cov_rho_lin = optimize.curve_fit(linear_fitfunc, k_fit2, rho_fit2)
print("rho_fit lin: ", p_rho_lin, cov_rho_lin)
p_f_lin, cov_f_lin = optimize.curve_fit(linear_fitfunc, k_fit2, f_fit2)
print("flux_fit lin: ", p_f_lin, cov_f_lin)

pl.plot(k, linear_fitfunc(k, *p_rho_lin), '--', label='rho fit lin')
pl.plot(k, linear_fitfunc(k, *p_f_lin), '--', label='flux fit lin')

filt_min = 0.1
filt_max = np.max(k)
i_filt = np.array([t for t in range(len(k)) if k[t] >= filt_min and k[t] <= filt_max])
rho_filt = np.array([rho_k[t] for t in i_filt])
f_filt = np.array([f_k[t] for t in i_filt])
k_filt = np.array([k[t] for t in i_filt])
fourier_filt = Fourier(k_filt, x)

#def f_k_extra(k):
#    return f_fit(k, *p_f)

#def rho_k_extra(k):
#    return rho_fit(k, *p_rho)


k1 = np.max(k)
k2 = 1500.0;
k_large = np.linspace(k_filt[-1], k2, 20000)
fourier_large = Fourier(k_large, x)
rho_large = rho_fitfunc(k_large, *p_rho)
f_large = f_fitfunc(k_large, *p_f)

print("k_filt: ", k_filt[0], k_filt[-1])
k_small = np.linspace(0.0, k_filt[0], 501)
fourier_small = Fourier(k_small, x)
f_small = linear_fitfunc(k_small, *p_f_lin)
rho_small = linear_fitfunc(k_small, *p_rho_lin)

fourier_data = [
    (k_filt,  fourier_filt,  f_filt,  rho_filt),
    (k_small, fourier_small, f_small, rho_small),
    (k_large, fourier_large, f_large, rho_large)
]


pl.figure()
ax_flux = pl.gca()
pl.figure()
ax_rho = pl.gca()

pl.figure()
ax_sq = pl.gca()

pl.figure()
ax_log = pl.gca()

#for r1 in [0.2, 0.1, 0.05, 0.02, 0.01]:
#for r1 in [0.15, 0.1, 0.05, 0.02, 0.01]:
#for r1 in [0.05, 0.02, 0.01]:
for r1 in [0.01, 0.005, 0.003]:
    def f_src(k):
        return np.exp(-(r1**2) * k**2 / 2.0)
    density_tot = 0.0 * x
    flux_tot = 0.0 * x; 
    for k_set, fourier_set, f_set, rho_set in fourier_data:
        src = f_src(k_set)
        flux_cur = np.dot(fourier_set, src * f_set).real * 2
        dens_cur = np.dot(fourier_set, src * rho_set).real * 2
        density_tot += dens_cur
        flux_tot += flux_cur
    #src = f_src(k)
    #src_extra = f_src(k_extra)
    #flux_x = np.dot(fourier, src * f_k).real * Ck
    #density_x = np.dot(fourier, src * rho_k).real * Ck
    #flux_x_extra = np.dot(fourier_extra, src_extra * f_extra).real * 2.0
    #density_x_extra = np.dot(fourier_extra, src_extra * rho_extra).real * 2.0
    
    #j_x = np.dot(fourier, src)


    #flux_tot    = flux_x + flux_x_extra
    #density_tot = density_x + density_x_extra
    #ax_flux.plot(x, flux_x, label='r0 = %g' % r1)
    ax_flux.plot(x, flux_tot, label='r0 = %g ex' % r1)
    #ax_rho.plot(x, density_x, label='r0 = %g' % r1)
    ax_rho.plot(x, density_tot, label='r0 = %g ex' % r1)

    x1_min = 0.05
    x1_max = 0.10
    
    x3_min = 0.1
    x3_max = 0.2
    
    x2_min = 6.0
    x2_max = 9.0

    i_x2 = np.array([t for t in range(len(x)) if x[t] > x2_min and x[t] < x2_max])
    i_x3 = np.array([t for t in range(len(x)) if x[t] > x3_min and x[t] < x3_max])
    i_x1 = np.array([t for t in range(len(x))
                     if x[t] < x1_max and x[t] > x1_min])
    x2 = np.array([x[t] for t in i_x2])
    x1 = np.array([x[t] for t in i_x1])
    x3 = np.array([x[t] for t in i_x3])
    flux2 = np.array([flux_tot[t] for t in i_x2])
    density2 = np.array([density_tot[t] for t in i_x2])
    flux3 = np.array([flux_tot[t] for t in i_x3])
    density1 = np.array([density_tot[t] for t in i_x1])
    def invsq_func(x, A, B, C):
        return A  + B/x + C/x * np.log(x) 

    def log_func(x, A, B, C, D):
        return B + A * np.log(x) + C * x + D * x * np.log(x)

    def log_inv_func(x, A, B, C, D, E):
        return A/x + B * np.log(x) + C + D * x + E * x * np.log(x) #+ D/x**2 + E/x**2 * np.log(x)
    
    p_flux2, cov_flux2 = optimize.curve_fit(invsq_func, x2, flux2 * x2**2)
    print("fit flux 1/x^2: ", p_flux2, cov_flux2)
    p_dens2, cov_dens2 = optimize.curve_fit(invsq_func, x2, density2 * x2**2)
    print("fit density 1/x^2: ", p_dens2, cov_dens2)
    p_flux3, cov_flux3 = optimize.curve_fit(log_func, x3, flux3)
    print("fit flux log(x) + const: ", p_flux3, cov_flux3)
    p_dens1, cov_dens1 = optimize.curve_fit(log_inv_func, x1, density1)
    print("fit density 1/x + log(x): ", p_dens1, cov_dens1)

    ax_flux.plot(x3, log_func(x3, *p_flux3), 'k--')
    ax_flux.plot(np.abs(x), log_func(np.abs(x), *p_flux3), 'k--')
    ax_flux.plot(x2, invsq_func(x2, *p_flux2)/x2**2, 'r--')
    ax_rho.plot(x1, log_inv_func(x1, *p_dens1), 'k--')
    ax_rho.plot(np.abs(x), log_inv_func(np.abs(x), *p_dens1), 'k--')
    ax_rho.plot(x2, invsq_func(x2, *p_dens2)/x2**2, 'r--')

    ax_sq.plot(x, flux_tot * x**2, label='flux')
    ax_sq.plot(x2, invsq_func(x2, *p_flux2), '--', label='flux fit')
    ax_sq.plot(x, density_tot * x**2, label='dens')
    ax_sq.plot(x2, invsq_func(x2, *p_dens2), '--', label='rho fit')
    ax_sq.legend()

    ax_log.loglog(np.abs(x), np.abs(flux_tot), label='flux')
    ax_log.loglog(np.abs(x), np.abs(density_tot), label='density')
    i_mid = np.argmin(np.abs(x - 7.0))
    ax_log.loglog(np.abs(x), np.abs(flux_tot[i_mid])*(x[i_mid]/np.abs(x)),
                  '--', label='1/x')
    ax_log.loglog(np.abs(x), np.abs(flux_tot[i_mid])*(x[i_mid]/np.abs(x))**2,
                  '--', label='1/x^2')
    ax_log.loglog(np.abs(x), np.abs(flux_tot[i_mid])*(x[i_mid]/np.abs(x))**3,
                  '--', label='1/x^3')
    ax_log.loglog(np.abs(x), np.abs(log_inv_func(np.abs(x), *p_dens1)),
                  '--', label='fit rho')
    ax_log.legend()

    
    

ax_flux.legend()
ax_rho.legend()
ax_rho.set_xlabel(r"Position $\gamma x$")
ax_rho.set_ylabel(r"Density $\rho(x)$")
ax_flux.set_xlabel(r"Position $\gamma x$")
ax_flux.set_ylabel(r"Flux to the wall $\Phi(x)$")

ax_flux.set_title("Flux to the wall")
ax_rho.set_title("Density")


pl.show()
