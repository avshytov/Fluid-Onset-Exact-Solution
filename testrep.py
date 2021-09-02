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

k = 0.3
gamma = 1.0
gamma1 = 0.9
y = np.linspace(0.0, 10.0, 1001)

K = WHKernels(gamma, gamma1)
path_up, K_up, path_dn, K_dn = make_paths_and_kernels(K, k)
diffuse = DiffuseFlow(k, K_up, path_up, K_dn, path_dn)
inj = EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn)

diff_rho = diffuse.rho_y(y)
inj_drho = inj.drho_y(y)

kappa = np.sqrt(gamma**2 + k**2)
s = np.linspace(kappa*1.0005, 20*kappa, 5000)
Krho_s = np.vectorize(lambda t: K.rho_plus(k, 1j * t))(s)
#Komega_s = np.vectorize(lambda t: K.omega_plus(k, 1j * t))(s)
Krho_star = K.rho_star(k)
Komega_star = K.omega_star(k)

abs_k = np.abs(k)
diff_rho_pole = 0.5 * (1.0 + gamma**2 / 2.0 / k**2 / Krho_star**2)
#diff_rho_pole = gamma1 * gamma**2 / 2.0 / abs_k**3 / Krho_star**2
#diff_rho_pole += gamma / abs_k / Krho_star - gamma / abs_k
diff_rho_pole *= np.exp(- abs_k * y)
sq_s = np.sqrt(s**2 - kappa**2)
exp_sy = np.exp(-np.outer(s, y)) + 0.0j
diff_rho_cut = gamma / 2.0 / ( abs_k + s) / Krho_star + 0.0j
diff_rho_cut *= sq_s / (s**2 - k**2) / Krho_s
diff_rho_cut = diff_rho_cut[:, None] * exp_sy / np.pi
ds = s[1] - s[0]
diff_rho_cut = np.sum(diff_rho_cut, axis=0) * ds
repr_rho_diff = diff_rho_cut + diff_rho_pole


inj_rho_pole = gamma1 * gamma**2 / 2.0 / abs_k**3 / Krho_star**2
inj_rho_pole += 2.0 * gamma / abs_k / Krho_star - gamma1 / abs_k
inj_rho_pole *= np.exp(- abs_k * y)
inj_rho_cut =  gamma * gamma1 / abs_k / (abs_k + s) / Krho_star + 0.0j
inj_rho_cut += 2.0 
#inj_rho_cut +=  -2.0 * (s**2 - k**2) / sq_s**2 * Krho_s
inj_rho_cut += -2.0 / sq_s**2 *  (s**2 - k**2) * Krho_s 
inj_rho_cut *= sq_s / (s**2 - k**2) / Krho_s
inj_rho_cut = inj_rho_cut[:, None] * exp_sy / np.pi
ds = s[1] - s[0]
inj_rho_cut = np.sum(inj_rho_cut, axis=0) * ds
repr_rho_inj = inj_rho_cut + inj_rho_pole



pl.figure()
pl.plot(y, diff_rho, label='Diffuse.rho')
pl.plot(y, repr_rho_diff, '--', label='pole + cut')
pl.plot(y, diff_rho_pole, label='pole')
pl.plot(y, diff_rho_cut, label='cut')
pl.plot(y, repr_rho_diff - diff_rho, label='diff')
pl.legend()

pl.figure()
pl.plot(y, inj_drho, label='Injected.drho')
pl.plot(y, repr_rho_inj, '--', label='pole + cut')
pl.plot(y, inj_rho_pole, label='pole')
pl.plot(y, inj_rho_cut, label='cut')
pl.plot(y, inj_rho_pole - inj_drho, label='diff: pole - tot')
pl.plot(y, repr_rho_inj - inj_drho, label='diff: pole + cut - tot')
pl.legend()

pl.show()


