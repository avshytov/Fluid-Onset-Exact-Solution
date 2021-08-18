import xkernel
import path
import flows
import numpy as np


def make_contours(k, kappa):
    a = np.sqrt(k * kappa)
    b = kappa - np.abs(k)/2
    smax = 100.0
    eps = 0.001
    z0 = 1j * kappa
    z_m = -a + 2j * kappa
    up_left = [
        #path.StraighPath(-eps + 1j * smax, z0 - a, 2000),
        path.StraighPath(-eps + 1j * smax, z_m,    2000),
        path.StraightPath(z_m,             z0 - a, 2000),
        path.ArcPath(z0, a, b, np.pi, 1.5 * np.pi, 200)
    ]
    path_up_left = path.append_paths (*up_left)
    path_up_right = path.transform(path.reverse(path_up_left),
                              lambda z: complex(-z.real, z.imag))
    path_up = path.append_paths(path_up_left, path_up_right)
    if True:
        import pylab as pl
        pl.figure()
        pl.plot(path_up.real, path_up.imag)
        pl.plot(path_up_left.real, path_up_left.imag)
        pl.plot(path_up_right.real, path_up_right.imag)

    path_dn = path.transform(path_up, lambda z: complex(z.real, -z.imag))
    return path_up, path_dn

def solve(h, k, gamma, gamma1, yv):
    kappa = np.sqrt(gamma**2 + k**2)
    path_up, path_dn = make_contour(abs(k), kappa)

    K = xkernel.WHKernels(gamma, gamma1)
    K_up = xkernel.TabulatedKernels(K, k, path_up)
    K_dn = xkernel.TabulatedKernels(K, k, path_dn)
    #Krho_sta = K_up.rho_star()
    #Xo_star  = K_up.omega_star()
    diff_up  = flows.DiffuseFlow(k, K_up)
    if np.abs(h) < 0.001:
        inj_up = flows.EdgeInjectedFlow(k, K_up)
    else:
        inj_up = flows.InjectedFlow(h, k, K_up, path_up, K_dn, path_dn)
    stokes_x  = flows.Stokeslet(1, 0, h, k, K_up, path_up, K_dn, path_dn)
    stokes_y  = flows.Stokeslet(0, 1, h, k, K_up, path_up, K_dn, path_dn)
    flows = {
          'diffuse'  : diff_up,
          'injection': inj_up,
          'stokes_x' : stokes_x,
          'stokes_y' : stokes_y
    }
    results = {}
    for label, flow in flows:
        res = dict()
        wall_flux = flow.wall_flux()
        rho = flow.rho(yv)
        jx, jy = flow.current(yv)
        res['jx'] = jx
        res['jy'] = jy
        res['flux'] = wall_flux
        res['rho'] = rho
        results[label] = res
    return results

    #src_flux  = inj_up.wall_flux()
    #diff_flux = diff_up.wall_flux()
    #sin_flux  = stokes_y.wall_flux()
    #cos_flux  = stokes_x.wall_flux()
    #rho_diff  = diff_up.rho(yv)
    #rho_inj   = inj_up.rho(yv)
    
