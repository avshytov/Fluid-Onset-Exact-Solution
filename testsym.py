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

def do_compare(x, y_new, y_old, quantity, label):
    pl.figure()
    pl.title("Compare: %s for k>0 vs k < 0, %s" % (quantity, label))
    pl.plot(x, y_new.real, label='gen: Re %s' % quantity)
    pl.plot(x, y_new.imag, label='gen: Im %s' % quantity)
    pl.plot(x, y_old.real, '--', label='%s: Re %s' % (label, quantity))
    pl.plot(x, y_old.imag, '--', label='%s: Im %s' % (label, quantity))
    pl.legend()
    

def test(flow_pos, flow_neg, label):
    gen_flow_pos = GenericFlow(flow_pos.k, flow_pos.K_up, flow_pos.path_up,
                              flow_pos.K_dn, flow_pos.path_dn)
    gen_flow_pos.solve(flow_pos.rho_direct, flow_pos.J, flow_pos.Omega_direct,
                   flow_pos.flux_down())
    gen_flow_neg = GenericFlow(flow_neg.k, flow_neg.K_up, flow_neg.path_up,
                              flow_neg.K_dn, flow_neg.path_dn)
    gen_flow_neg.solve(flow_neg.rho_direct, flow_neg.J, flow_neg.Omega_direct,
                   flow_neg.flux_down())
    x_arc = flow_pos.path_dn.arc_lengths()
    do_compare (x_arc, flow_pos.rho_plus_dn(), flow_neg.rho_plus_dn(),
                "dn: rho+", label)
    do_compare (x_arc, flow_pos.Omega_plus_dn(), -flow_neg.Omega_plus_dn(),
                "dn: Omega+", label)
    do_compare (x_arc, flow_pos.rho_plus_up(), flow_neg.rho_plus_up(),
                "up: rho+", label)
    do_compare (x_arc, flow_pos.Omega_plus_up(), -flow_neg.Omega_plus_up(),
                "up: Omega+", label)
    
    do_compare (x_arc, gen_flow_pos.rho_plus_dn(), gen_flow_neg.rho_plus_dn(),
                "dn, gen: rho+", label)
    do_compare (x_arc, gen_flow_pos.Omega_plus_dn(),
                -gen_flow_neg.Omega_plus_dn(),
                "dn, gen: Omega+", label)
    do_compare (x_arc, gen_flow_pos.rho_plus_up(), gen_flow_neg.rho_plus_up(),
                "up, gen: rho+", label)
    do_compare (x_arc, gen_flow_pos.Omega_plus_up(), -gen_flow_neg.Omega_plus_up(),
                "up, gen: Omega+", label)
    print ("fluxes:", flow_pos.wall_flux(), flow_neg.wall_flux())
    print ("generic fluxes:",
           gen_flow_pos.wall_flux(), gen_flow_neg.wall_flux())
    y = np.linspace(-1, 20, 2101)
    do_compare(y, flow_pos.rho_y(y), flow_neg.rho_y(y), "rho(y)", label)
    do_compare(y, flow_pos.jx_y(y),  -flow_neg.jx_y(y), "j_x(y)", label)
    do_compare(y, flow_pos.jy_y(y),  flow_neg.jy_y(y), "j_y(y)", label)
    do_compare(y, gen_flow_pos.rho_y(y), gen_flow_neg.rho_y(y),
               "gen: rho(y)", label)
    do_compare(y, gen_flow_pos.jx_y(y),  -gen_flow_neg.jx_y(y),
               "gen: j_x(y)", label)
    do_compare(y, gen_flow_pos.jy_y(y),  gen_flow_neg.jy_y(y),
               "gen: j_y(y)", label)

k = 0.5
h = 0.8
gamma  = 1.0
gamma1 = 1.0

K = WHKernels(gamma, gamma1)
path_up, K_up, path_dn, K_dn = make_paths_and_kernels(K, k)

custom_flows_pos = {
    "edge-src-sym" : EdgeInjectedFlow_sym(k, K_up, path_up, K_dn, path_dn),
    "diffuse-sym"  : DiffuseFlow_sym(k, K_up, path_up, K_dn, path_dn),
    "diffuse"  : DiffuseFlow(k, K_up, path_up, K_dn, path_dn),
    "edge-src" : EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn),
    "bulk-src" : InjectedFlow(h, k, K_up, path_up, K_dn, path_dn),
    "stokes-x" : Stokeslet(1, 0, h, k, K_up, path_up, K_dn, path_dn),
    "stokes-y" : Stokeslet(0, 1, h, k, K_up, path_up, K_dn, path_dn),
}

custom_flows_neg = {
    "edge-src-sym" : EdgeInjectedFlow_sym(-k, K_up, path_up, K_dn, path_dn),
    "diffuse-sym"  : DiffuseFlow_sym(-k, K_up, path_up, K_dn, path_dn),
    "diffuse"  : DiffuseFlow(-k, K_up, path_up, K_dn, path_dn),
    "edge-src" : EdgeInjectedFlow(-k, K_up, path_up, K_dn, path_dn),
    "bulk-src" : InjectedFlow(h, -k, K_up, path_up, K_dn, path_dn),
    "stokes-x" : Stokeslet(1, 0, h, -k, K_up, path_up, K_dn, path_dn),
    "stokes-y" : Stokeslet(0, 1, h, -k, K_up, path_up, K_dn, path_dn),
}

for label in custom_flows_pos.keys():
    flow_pos = custom_flows_pos[label]
    flow_neg = custom_flows_neg[label]
    test(flow_pos, flow_neg, label)
    pl.show()

    
