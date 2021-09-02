import contours
from edge import EdgeInjectedFlow, EdgeInjectedFlow_sym
from bulk import InjectedFlow
from diffuse import DiffuseFlow, DiffuseFlow_sym
from stokeslet import Stokeslet
from generic import GenericFlow
import pylab as pl
import numpy as np
from xkernel_new import WHKernels
from contours import make_paths_and_kernels
from scipy import integrate

k = 0.4
gamma = 1.0
gamma1 = 0.99999
y = np.linspace(0.0, 10.0, 1001)
print("k = k", "gamma1 = ", gamma1)

K = WHKernels(gamma, gamma1)
path_up, K_up, path_dn, K_dn = make_paths_and_kernels(K, k)
diffuse = DiffuseFlow(k, K_up, path_up, K_dn, path_dn)
inj = EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn)

diff_rho = diffuse.rho_y(y)
inj_drho = inj.drho_y(y)

kappa = np.sqrt(gamma**2 + k**2)
s = np.linspace(kappa*1.0005, 25*kappa, 5000)
Krho_s = np.vectorize(lambda t: K.rho_plus(k, 1j * t))(s)
Komega_s = np.vectorize(lambda t: K.omega_plus(k, 1j * t))(s)
Krho_star = K.rho_star(k)
Komega_star = K.omega_star(k)
Komega_star_star = K.omega_star_star(k)

dk = 1e-5
dK = K.omega_plus(k, 1j * (k + dk)) - K.omega_plus(k, 1j * (k - dk))
Kprime = dK / 2.0 / dk / Komega_star
atan = np.arctan(gamma / np.abs(k))
Kprime_exact = 1.0/np.pi/gamma - (k**2 + gamma**2)/np.pi/gamma**2 / k * atan
print ("Kprime = ", Kprime, "exact = ", Kprime_exact)

abs_k = np.abs(k)
exp_ky = np.exp(-abs_k * y)
gamma2 = gamma - gamma1
kappa1 = np.sqrt(k**2 + 4.0 * gamma1 * gamma2)
exp_kappay = np.exp(-kappa1 * y)

sgn_k = np.sign(k)


print ("Komega_star = ", Komega_star, K.omega_plus(k, 1j * np.abs(k)))
print ("Komega_star_star = ", Komega_star_star, K.omega_plus(k, 1j * kappa1))


diff_rho_pole = 0.5 * (1.0 + gamma**2 / 2.0 / k**2 / Krho_star**2)
#diff_rho_pole = gamma1 * gamma**2 / 2.0 / abs_k**3 / Krho_star**2
#diff_rho_pole += gamma / abs_k / Krho_star - gamma / abs_k
diff_rho_pole *= exp_ky
sq_s = np.sqrt(s**2 - kappa**2)
exp_sy = np.exp(-np.outer(s, y)) + 0.0j
diff_rho_cut = gamma / 2.0 / ( abs_k + s) / Krho_star + 0.0j
diff_rho_cut *= sq_s / (s**2 - k**2) / Krho_s
diff_rho_cut = diff_rho_cut[:, None] * exp_sy / np.pi
ds = s[1] - s[0]
diff_rho_cut = integrate.trapz(diff_rho_cut, s, axis=0)
#np.sum(diff_rho_cut, axis=0) * ds
repr_rho_diff = diff_rho_cut + diff_rho_pole


inj_rho_pole = gamma1 * gamma**2 / 2.0 / abs_k**3 / Krho_star**2
inj_rho_pole += 2.0 * gamma / abs_k / Krho_star - gamma1 / abs_k
inj_rho_pole *= exp_ky
inj_rho_cut =  gamma * gamma1 / abs_k / (abs_k + s) / Krho_star + 0.0j
inj_rho_cut += 2.0 
#inj_rho_cut +=  -2.0 * (s**2 - k**2) / sq_s**2 * Krho_s
inj_rho_cut += -2.0 / sq_s**2 *  (s**2 - k**2) * Krho_s 
inj_rho_cut *= sq_s / (s**2 - k**2) / Krho_s
inj_rho_cut = inj_rho_cut[:, None] * exp_sy / np.pi
ds = s[1] - s[0]
inj_rho_cut = integrate.trapz(inj_rho_cut, s, axis=0)
#np.sum(inj_rho_cut, axis=0) * ds
repr_rho_inj = inj_rho_cut + inj_rho_pole

diff_psi = -diffuse.jy_y(y) / 1j / k
diff_psi_pole1 =    gamma / abs_k / Komega_star**2 
diff_psi_pole1 += - gamma2 / abs_k
diff_psi_pole1 *= exp_ky
diff_psi_pole2  = - (gamma1 - gamma2) / kappa1 / Komega_star / Komega_star_star
diff_psi_pole2 *=  exp_kappay
diff_psi_pole = diff_psi_pole1 + diff_psi_pole2
diff_psi_pole *= 0.25 * 1j * k / gamma1 / gamma2
diff_psi_cut = 2.0 * gamma1 * sq_s / (s**2 - kappa1**2)/(s**2 - k**2) + 0.0j
diff_psi_cut *= 1.0 / (Komega_s * Komega_star)
diff_psi_cut = diff_psi_cut[:, None] * exp_sy / np.pi
diff_psi_cut *= 0.5 * k / gamma1 
ds = s[1] - s[0]
diff_psi_cut = integrate.trapz(diff_psi_cut, s, axis=0)
diff_psi_cut *= -1j
#np.sum(diff_psi_cut, axis=0) * ds
repr_psi_diff = diff_psi_cut + diff_psi_pole

psi_pole_visc = - 0.5 * Komega_star**2 + 1.0 + gamma**2/k**2 * (1 + abs_k * y)
psi_pole_visc +=  Kprime * gamma**2 / k
psi_pole_visc *= 0.5 / Komega_star**2 * 1j * sgn_k * exp_ky

inj_psi = -inj.jy_y(y) / 1j / k
inj_psi_pole1 =    gamma / abs_k / Komega_star**2 * exp_ky
inj_psi_pole1 += - gamma2 / abs_k * exp_ky
inj_psi_pole2  = - (gamma1 - gamma2) / kappa1 / Komega_star / Komega_star_star
inj_psi_pole2 *=  exp_kappay
inj_psi_pole = inj_psi_pole1 + inj_psi_pole2
inj_psi_pole *= 0.5 * 1j / gamma2 * sgn_k
inj_psi_cut = 2.0 * gamma1 * sq_s / (s**2 - kappa1**2)/(s**2 - k**2) + 0.0j
inj_psi_cut *= 1.0 / (Komega_s * Komega_star)
inj_psi_cut = inj_psi_cut[:, None] * exp_sy / np.pi
ds = s[1] - s[0]
inj_psi_cut = integrate.trapz(inj_psi_cut, s, axis=0)
inj_psi_cut *= -1j * sgn_k
inj_psi_0 = -1 * 2 *  0.5/1j/k * exp_ky
repr_psi_inj = inj_psi_cut + inj_psi_pole + inj_psi_0



pl.figure()
pl.plot(y, diff_rho.real, label='Diffuse.rho')
pl.plot(y, repr_rho_diff.real, '--', label='pole + cut')
pl.plot(y, diff_rho_pole.real, label='pole')
pl.plot(y, diff_rho_cut.real, label='cut')
pl.plot(y, repr_rho_diff.real - diff_rho.real, label='diff')
pl.legend()

pl.figure()
pl.plot(y, inj_drho.real, label='Injected.drho')
pl.plot(y, repr_rho_inj.real, '--', label='pole + cut')
pl.plot(y, inj_rho_pole.real, label='pole')
pl.plot(y, inj_rho_cut.real, label='cut')
pl.plot(y, inj_rho_pole.real - inj_drho.real, label='diff: pole - tot')
pl.plot(y, repr_rho_inj.real - inj_drho.real, label='diff: pole + cut - tot')
pl.legend()

pl.figure()
pl.plot(y, diff_psi.imag, label='Diffuse.psi')
pl.plot(y, repr_psi_diff.imag, '--', label='pole + cut')
pl.plot(y, diff_psi_pole.imag, label='pole')
pl.plot(y, diff_psi_cut.imag, label='cut')
pl.plot(y, diff_psi_pole.imag - diff_psi.imag, label='diff: pole - tot')
pl.plot(y, repr_psi_diff.imag - diff_psi.imag, label='diff: pole + cut - tot')
pl.plot(y, psi_pole_visc.imag, '--', label='visc expansion')
pl.plot(y, (psi_pole_visc.imag - diff_psi_pole.imag)/exp_ky, label='visc -pole')
pl.legend()

pl.figure()
pl.plot(y, inj_psi.imag, label='Injected.psi')
pl.plot(y, repr_psi_inj.imag, '--', label='pole + cut')
pl.plot(y, inj_psi_pole.imag, label='pole')
pl.plot(y, inj_psi_cut.imag, label='cut')
pl.plot(y, inj_psi_pole.imag - inj_psi.imag, label='inj: pole - tot')
pl.plot(y, repr_psi_inj.imag - inj_psi.imag, label='inj: pole + cut - tot')
pl.legend()

pl.show()


