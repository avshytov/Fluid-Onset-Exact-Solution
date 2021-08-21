import contours
from edge import EdgeInjectedFlow
from bulk import InjectedFlow
from diffuse import DiffuseFlow
from stokeslet import Stokeslet
from generic import GenericFlow
import pylab as pl
import numpy as np
from xkernel_new import WHKernels
from contours import make_paths_and_kernels

def do_compare(x, y_new, y_old, quantity, label):
    pl.figure()
    pl.title("Compare: %s for generic vs %s" % (quantity, label))
    pl.plot(x, y_new.real, label='gen: Re %s' % quantity)
    pl.plot(x, y_new.imag, label='gen: Im %s' % quantity)
    pl.plot(x, y_old.real, '--', label='%s: Re %s' % (label, quantity))
    pl.plot(x, y_old.imag, '--', label='%s: Im %s' % (label, quantity))
    pl.legend()
    

def test(flow, label):
    gen_flow = GenericFlow(k, flow.K_up, flow.path_up,
                              flow.K_dn, flow.path_dn)
    gen_flow.solve(flow.rho_direct, flow.J, flow.Omega_direct)
    x_arc = flow.path_dn.arc_lengths()
    do_compare (x_arc, gen_flow.rho_plus_dn(), flow.rho_plus_dn(),
                "rho+", label)
    do_compare (x_arc, gen_flow.Omega_plus_dn(), flow.Omega_plus_dn(),
                "Omega+", label)
    print ("fluxes:", flow.wall_flux(), gen_flow.wall_flux())
    y = np.linspace(-1, 20, 2101)
    do_compare(y, gen_flow.rho_y(y), flow.rho_y(y), "rho(y)", label)
    do_compare(y, gen_flow.jx_y(y), flow.jx_y(y), "j_x(y)", label)
    do_compare(y, gen_flow.jy_y(y), flow.jy_y(y), "j_y(y)", label)

k = 0.5
h = 0.8
gamma  = 1.0
gamma1 = 1.0

K = WHKernels(gamma, gamma1)
path_up, K_up, path_dn, K_dn = make_paths_and_kernels(K, k)

custom_flows = {
    "diffuse"  : DiffuseFlow(k, K_up, path_up, K_dn, path_dn),
    "edge-src" : EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn),
    "bulk-src" : InjectedFlow(h, k, K_up, path_up, K_dn, path_dn),
    "stokes-x" : Stokeslet(1, 0, h, k, K_up, path_up, K_dn, path_dn),
    "stokes-y" : Stokeslet(0, 1, h, k, K_up, path_up, K_dn, path_dn),
}

for label, flow in custom_flows.items():
    test(flow, label)
    pl.show()

    
