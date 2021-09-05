import xkernel_new as xkernel
import path
import flows
import numpy as np
from edgesin import EdgeSinFlow, EdgeSinFlow_sym
from diffuse import DiffuseFlow, DiffuseFlow_sym
from stokeslet import Stokeslet
from bulk import InjectedFlow
from edge import EdgeInjectedFlow, EdgeInjectedFlow_sym
from flows import CombinedFlow

import numpy as np
import contours
import pylab as pl

def make_contours_im(k, kappa):
    abs_k = np.abs(k)
    a = np.sqrt(abs_k * kappa)
    b = kappa - abs_k / 2.0
    path_vert = path.StraightPath(100.0j*kappa - a, -a + 1j * kappa, 4000)
    def scaling_func(t):
        xi  = 2 * t - 1.0      # (0, 1) -> (-1, 1)
        eta = xi * np.abs(xi)  # more points near the middle
        return (eta + 1.0) / 2.0 # (-1, 1) -> (0, 1)
    path_arc  = path.ArcPath(1j * kappa, a, b, np.pi, 1.5*np.pi, 501,
                               scaling_func)
    path_up_left  = path.append_paths(path_vert, path_arc)
    path_up_right = path.transform(path_up_left,
                                   lambda z: complex(-z.real, z.imag))
    path_up_right.reverse()
    path_up = path.append_paths(path_up_left, path_up_right)
    path_dn = path.transform(path_up,
                             lambda z: complex(z.real, -z.imag))
    return path_up, path_dn

def make_contours(k, kappa):
    if True:
       #path_up = path.StraightPath(-20.0 + 0.01j, 20.0 + 0.01j, 201)
       #path_up = path.StraightPath(-50.0 + 0.01j, 50.0 + 0.01j, 5001)
       path_up = path.append_paths(
           path.StraightPath(-30.0 + 3j, -20.0 + 0.1j, 101), 
           path.StraightPath(-20.0 + 0.1j, -10.0 + 0.005j, 101),
           path.StraightPath(-10.0 + 0.005j, 10.0 + 0.005j, 201),
           path.StraightPath( 10.0 + 0.005j, 20.0 + 0.1j, 101),
       #    #path.StraightPath(-20.0 + 0.005j, 20.0 + 0.005j, 801),
           path.StraightPath( 20.0 + 0.1j, 30.0 + 3j, 101)
       ) 
       path_dn = path.transform(path_up, lambda z: complex(z.real, -z.imag))
       return path_up, path_dn
    
    a = np.sqrt(k * kappa)
    b = kappa - np.abs(k)/2
    smax = 50 # 100.0
    eps = 0.01 # 0.001
    z0 = 1j * kappa
    z_m = -a + 2j * kappa
    up_left = [
        #path.StraighPath(-eps + 1j * smax, z0 - a, 2000),
        path.StraightPath(-eps + 1j * smax, z_m,    500),
        path.StraightPath(z_m,             z0 - a, 500),
        path.ArcPath(z0, a, b, np.pi, 1.5 * np.pi, 100)
    ]
    path_up_left = path.append_paths (*up_left)
    path_up_right = path.transform(path.reverse(path_up_left),
                              lambda z: complex(-z.real, z.imag))
    path_up = path.append_paths(path_up_left, path_up_right)
    if False:
        import pylab as pl
        pl.figure()
        pl.plot(path_up.real, path_up.imag)
        pl.plot(path_up_left.real, path_up_left.imag)
        pl.plot(path_up_right.real, path_up_right.imag)

    path_dn = path.transform(path_up, lambda z: complex(z.real, -z.imag))
    return path_up, path_dn

def show_rhs_and_diff(flow, label, lhs_rho, rhs_rho, lhs_omega, rhs_omega,
                      lhs_D, rhs_D, contour, contour_label):
    import pylab as pl

    err_rho = lhs_rho - rhs_rho
    err_D = lhs_D - rhs_D
    err_omega = lhs_omega - rhs_omega
    
    q = contour.arc_lengths() #path_up.points().real
    #q = contour.points().real

    pl.figure()
    pl.plot(q, lhs_rho.real, label='Re lhs.rho')
    pl.plot(q, lhs_rho.imag, label='Im lhs.rho')
    pl.plot(q, rhs_rho.real, '--', label='Re rhs rho')
    pl.plot(q, rhs_rho.imag, '--', label='Re rhs rho')
    pl.plot(q, np.abs(err_rho), label='|diff|')
    pl.legend()
    pl.title("%s rho %s" % (label, contour_label))

    pl.figure()
    pl.plot(q, lhs_omega.real, label='Re lhs.omega')
    pl.plot(q, lhs_omega.imag, label='Im lhs.omega')
   # print ("shape: ", q.shape, rhs_omega.shape)
    pl.plot(q, rhs_omega.real, '--', label='Re rhs omega')
    pl.plot(q, rhs_omega.imag, '--', label='Im rhs omega')
    pl.plot(q, np.abs(err_omega), label='|diff|')
    pl.legend()
    pl.title("%s omega %s" % (label, contour_label))

    pl.figure()
    pl.plot(q, lhs_D.real, label='Re lhs D')
    pl.plot(q, lhs_D.imag, label='Im lhs D')
    pl.plot(q, rhs_D.real, '--', label='Re rhs D')
    pl.plot(q, rhs_D.imag, '--', label='Re rhs D')
    pl.plot(q, np.abs(err_D), label='|diff|')
    pl.legend()
    pl.title("%s D %s" % (label, contour_label))

    pl.figure()
    pl.plot(q, err_rho.real, label='Re err rho')
    pl.plot(q, err_rho.imag, label='Im err rho')
    pl.plot(q, err_omega.real, label='Re err omega')
    pl.plot(q, err_omega.imag, label='Im err omega')
    pl.plot(q, err_D.real, label='Re err D')
    pl.plot(q, err_D.imag, label='Im err D')
    pl.title("errors: %s %s" % (label, contour_label))
    pl.legend()
    
    #pl.show()
    
def show_solution(flow, label, rho_plus, rho_minus, omega_plus, omega_minus,
                  contour, contour_label):
    import pylab as pl
    #q = contour.points().real
    q = contour.arc_lengths()
    pl.figure()
    pl.plot(q, rho_plus.real, label='Re rho+')
    pl.plot(q, rho_plus.imag, label='Im rho+')
    pl.plot(q, omega_plus.real, label='Re Omega+')
    pl.plot(q, omega_plus.imag, label='Im Omega+')
    pl.title("plus %s" % contour_label)
    #pl.legend()
    #pl.figure()
    pl.plot(q, rho_minus.real, '--', label='Re rho-')
    pl.plot(q, rho_minus.imag, '--', label='Im rho-')
    pl.plot(q, omega_minus.real, '--', label='Re Omega-')
    pl.plot(q, omega_minus.imag, '--', label='Im Omega-')
    pl.title("minus %s" % contour_label)
    pl.legend()
    #pl.show()

def test_up(flow, label, gamma, gamma1, k, h, K, path_up):
    
    q_up = path_up.points()
    rho_plus      = flow.rho_plus_up()
    rho_minus     = flow.rho_minus_up()
    rho_direct    = flow.rho_direct(q_up)
    omega_plus    = flow.Omega_plus_up()
    omega_minus   = flow.Omega_minus_up()
    omega_direct  = flow.Omega_direct(q_up)
    D_plus  = flow.D_plus_up()
    D_minus = flow.D_minus_up() 
    Krho   = K.rho(k, q_up)
    Komega = K.omega(k, q_up)
    lhs_rho = rho_plus * Krho + rho_minus
    k2 = k**2 + q_up**2
    rhs_rho = rho_direct + 2j * gamma1 / k2 * D_plus * Krho
    #err_rho = lhs_rho - rhs_rho
    lhs_D  = D_plus + D_minus
    rhs_D = -1j * gamma * rho_minus + 1j * flow.J(q_up)
    #err_D = lhs_D - rhs_D
    #print ("shape: omega-", np.shape(omega_direct), omega_direct)
    lhs_omega = Komega * omega_plus + omega_minus
    rhs_omega = omega_direct
    #err_omega = lhs_omega - rhs_omega
    show_rhs_and_diff(flow, label, lhs_rho, rhs_rho, lhs_omega, rhs_omega,
                      lhs_D, rhs_D, path_up, "upper contour")
    show_solution(flow, label, rho_plus, rho_minus, omega_plus, omega_minus,
                  path_up, "upper contour")

def test_dn(flow, label, gamma, gamma1, k, h, K, path_dn):
    q_dn = path_dn.points()
    rho_plus      = flow.rho_plus_dn()
    rho_minus     = flow.rho_minus_dn()
    rho_direct    = flow.rho_direct(q_dn)
    omega_plus    = flow.Omega_plus_dn()
    omega_minus   = flow.Omega_minus_dn()
    omega_direct  = flow.Omega_direct(q_dn)
    D_plus  = flow.D_plus_dn()
    D_minus = flow.D_minus_dn() 
    Krho   = K.rho(k, q_dn)
    Komega = K.omega(k, q_dn)
    lhs_rho = rho_plus * Krho + rho_minus
    k2 = k**2 + q_dn**2
    rhs_rho = rho_direct + 2j * gamma1 / k2 * D_plus * Krho
    #err_rho = lhs_rho - rhs_rho
    lhs_D  = D_plus + D_minus
    rhs_D = -1j * gamma * rho_minus + 1j * flow.J(q_dn)
    #err_D = lhs_D - rhs_D
    #print ("shape: omega-", np.shape(omega_direct), omega_direct)
    lhs_omega = Komega * omega_plus + omega_minus
    rhs_omega = omega_direct
    #err_omega = lhs_omega - rhs_omega
    show_rhs_and_diff(flow, label, lhs_rho, rhs_rho, lhs_omega, rhs_omega,
                      lhs_D, rhs_D, path_dn, "lower contour")
    show_solution(flow, label, rho_plus, rho_minus, omega_plus, omega_minus,
                  path_dn, "lower contour")
    
def test_rho_y_new(flow, label, yv):

    import pylab as pl
    rho_y   = flow.rho_y(yv)
    drho_y  = flow.drho_y(yv)
    rho_sing = flow.rho_sing_y(yv)
    rho_y2 = drho_y + rho_sing;
    pl.figure()
    pl.plot(yv, rho_y.real, label='Re rho(y)')
    pl.plot(yv, rho_y.imag, label='Im rho(y)')
    pl.plot(yv, drho_y.real, label='Re drho(y)')
    pl.plot(yv, drho_y.imag, label='Im drho(y)')
    pl.plot(yv, rho_y2.real, label='Re rho(y) + sing')
    pl.plot(yv, rho_y2.imag, label='Im drho(y) + sing')
    pl.plot(yv, rho_sing.real, label='Re sing')
    pl.plot(yv, rho_sing.imag, label='Im sing')
    pl.legend()
    ymin, ymax = pl.ylim()
    pl.ylim(max(ymin, -5.0), min(ymax, 5.0))
    pl.title("rho(y) for %s" % label)
    
def test_j_y(flow, label, yv):

    import pylab as pl
    jx_y  = flow.jx_y(yv)
    jy_y  = flow.jy_y(yv)
    pl.figure()
    pl.plot(yv, jx_y.real, label='Re j_x(y)')
    pl.plot(yv, jx_y.imag, label='Im j_x(y)')
    pl.plot(yv, jy_y.real, label='Re j_y(y)')
    pl.plot(yv, jy_y.imag, label='Im j_y(y)')
    pl.legend()
    pl.title("j(y) for %s" % label)
    ymin, ymax = pl.ylim()
    pl.ylim(max(ymin, -5.0), min(ymax, 5.0))
    
def test_rho_y(flow, label, yv, path_up, path_dn):
    import pylab as pl
    rho_plus_up  = flow.rho_plus_up()
    rho_plus_dn  = flow.rho_plus_dn()
    #drho_plus_up = flow.drho_plus_up()
    drho_plus_dn = flow.drho_plus_dn()
    rho_minus_up = flow.rho_minus_up()
    def rho_y(path, rho_q, y):
        q = path.points()
        return path.integrate_array(np.exp(-1j*y*q)*rho_q) / 2.0 / np.pi

    y_right = 5.0
    y_left = -5.0
    i_left  = np.argmin(np.abs(yv - y_left))
    i_right = np.argmin(np.abs(yv - y_right))
    #print (flow.__dict__)
    #if 'rho_plus_up' in flow.__dict__:
    #    print ("has rho_plus_up")
    #else:
    #    print ("no rho_plus_up")
    rho_yp  = np.vectorize(lambda y: rho_y(path_dn, rho_plus_dn,  y))(yv)
    rho_yp1 = np.vectorize(lambda y: rho_y(path_up, rho_plus_up,  y))(yv)
    drho_yp  = np.vectorize(lambda y: rho_y(path_dn, drho_plus_dn,  y))(yv)
    #drho_yp1 = np.vectorize(lambda y: rho_y(path_up, drho_plus_up,  y))(yv)
    rho_ym = np.vectorize(lambda y: rho_y(path_up,  rho_minus_up, y))(yv)
    rho_ys = rho_yp + rho_ym
    rho_yp[0:i_left] = 0
    rho_yp1[i_right:] = 0
    drho_yp[0:i_left] = 0
    #drho_yp1[i_right:] = 0
    rho_ym[i_right:] = 0
    pl.figure()
    pl.plot(yv, rho_yp.real, label='Re rho+(y) dn')
    pl.plot(yv, rho_yp.imag, label='Im rho+(y) dn')
    pl.plot(yv, rho_yp1.real, label='Re rho+(y) up')
    pl.plot(yv, rho_yp1.imag, label='Im rho+(y) up')
    pl.plot(yv, rho_ym.real, label='Re rho-(y) up')
    pl.plot(yv, rho_ym.imag, label='Im rho-(y) up')
    #pl.plot(yv, rho_ys.real, label='Re rho (y)')
    #pl.plot(yv, rho_ys.imag, label='Im rho (y)')
    pl.legend()
    #pl.xlim(-5, 20.0)
    #pl.ylim(-3.0, 3.0)
    pl.figure()
    pl.plot(yv, drho_yp.real, label='Re drho+(y) dn')
    pl.plot(yv, drho_yp.imag, label='Im drho+(y) dn')
    #pl.plot(yv, drho_yp1.real, label='Re drho+(y) up')
    #pl.plot(yv, drho_yp1.imag, label='Im drho+(y) up')
    pl.legend()
    pl.xlim(-5, 20.0)
    pl.ylim(-20.0, 20.0)
    #pl.show()

def show_paths(path_up, path_dn, q_m):
    import pylab as pl
    pl.figure()
    pl.plot(path_up.points().real, path_up.points().imag, '-o',
            ms=1.0, label='up')
    pl.plot(path_dn.points().real, path_dn.points().imag, '-o',
            ms=1.0, label='dn')
    pl.plot(q_m.real, q_m.imag, label='mid')
    pl.gca().set_aspect('equal', 'box')
    pl.legend()

def show_kernels(K_up, K_dn, path_up, path_dn):
    import pylab as pl
    pl.figure()
    x_up = path_up.points().real
    x_dn = path_dn.points().real
    pl.plot(x_up, K_up.rho_plus().real, label='Re Kup rho+')
    pl.plot(x_up, K_up.rho_plus().imag, label='Im Kup rho+')
    pl.plot(x_dn, K_dn.rho_plus().real, '--', label='Re Kdn rho+')
    pl.plot(x_dn, K_dn.rho_plus().imag, '--', label='Im Kdn rho+')
    pl.plot(x_up, K_up.rho_minus().real, label='Re Kup rho-')
    pl.plot(x_up, K_up.rho_minus().imag, label='Im Kup rho-')
    pl.plot(x_dn, K_dn.rho_minus().real, '--', label='Re Kdn rho-')
    pl.plot(x_dn, K_dn.rho_minus().imag, '--', label='Im Kdn rho-')
    pl.legend()

def test_Dstar(flow, label, path_up, path_dn):
    print ("testing flow", flow, type(flow))
    D_plus = flow.D_plus_up()
    omega_plus = flow.Omega_plus_up()
    D_minus = flow.D_minus_up()
    omega_minus = flow.Omega_minus_up()
    flux = flow.wall_flux()
    abs_k = np.abs(k)
    sgn_k = 1.0
    if k < 0:
        sgn_k = -1.0
    z_up = path_up.points()
    D_star = path_up.integrate_array(D_plus/(z_up - 1j * abs_k)) / 2.0 / np.pi / 1j
    Omega_star = path_up.integrate_array(omega_plus
                    /(z_up - 1j * abs_k)) / 2.0 / np.pi / 1j
    print ("Flow:", label,
           "D* = ", D_star,
           "O* = ", Omega_star, "diff = ",
           D_star - 1j * sgn_k * Omega_star)

    z_dn = path_dn.points()
    Dm_star = path_dn.integrate_array(D_minus/(z_dn + 1j * abs_k))/2.0/np.pi/1j
    Om_star = path_dn.integrate_array(omega_minus/(z_dn + 1j * abs_k))/2.0/np.pi/1j

    print ("wall flux: ", flux)
    print ("iD+O: ", 1j * Dm_star + Om_star * sgn_k)
    print ("iD-O: ", 1j * Dm_star + Om_star * sgn_k)

def test_wh(h, k, gamma, gamma1, yv):
    kappa = np.sqrt(gamma**2 + k**2)
    #path_up, path_dn = make_contours_im(abs(k), kappa)
    path_ul = contours.make_arc(abs(k), kappa)
    # upper left segment, used to construct the rest 
    

    #show_paths(path_up, path_dn, q_m)
    #import pylab as pl; pl.show()
    
    K = xkernel.WHKernels(gamma, gamma1)
    print ("tabulate kernels up")
    import time; now = time.time()
    K_ul = xkernel.tabulate_kernel(K, k, path_ul.points())
    #K_ul = xkernel.load_kernel(K, k, path_ul.points(), "ul", False)
    # get upper-right segment
    path_ru, K_ru = contours.conjugate_left_right(path_ul, K_ul) # backward
    path_ur, K_ur = contours.reverse_path_and_kernel(path_ru, K_ru)
    
    path_up, K_up = contours.append_paths_and_kernels(path_ul, K_ul,
                                                      path_ur, K_ur)
    path_dn, K_dn = contours.conjugate_up_down(path_up, K_up)
    print ("kernels ready: ", time.time() - now); now = time.time()
    #q_m = 0.5 * (path_up.points() + path_dn.points())
    #show_paths(path_up, path_dn, q_m)
    #K_up_old = xkernel.load_kernel(K, k, path_up.points(), "up", True)
    #K_dn_old = xkernel.load_kernel(K, k, path_dn.points(), "dn", True)

    if False:
        import pylab as pl
        arcl_up = path_up.arc_lengths()
        pl.figure()
        pl.plot(arcl_up, K_up.Krho_p.real, label="Re new Krho up")
        pl.plot(arcl_up, K_up.Krho_p.imag, label="Im new Krho up")
        pl.plot(arcl_up, K_up_old.Krho_p.real,'--',  label="Re old Krho up")
        pl.plot(arcl_up, K_up_old.Krho_p.imag, '--', label="Im old Krho up")
        pl.legend()
        arcl_dn = path_dn.arc_lengths()
        pl.figure()
        pl.plot(arcl_dn, K_dn.Krho_p.real, label="Re new Krho dn")
        pl.plot(arcl_dn, K_dn.Krho_p.imag, label="Im new Krho dn")
        pl.plot(arcl_dn, K_dn_old.Krho_p.real, '--',  label="Re old Krho dn")
        pl.plot(arcl_dn, K_dn_old.Krho_p.imag, '--', label="Im old Krho dn")
        pl.legend()
        pl.figure()
        pl.plot(arcl_up, K_up.Krho_p.real - K_up_old.Krho_p.real,
                label="Re doff Krho up")
        pl.plot(arcl_up, K_up.Krho_p.imag - K_up_old.Krho_p.imag,
                label="Im diff Krho up")
        pl.plot(arcl_dn, K_dn_old.Krho_p.real - K_dn.Krho_p.real,
                '--',  label="Re diff Krho dn")
        pl.plot(arcl_dn, K_dn_old.Krho_p.imag - K_dn.Krho_p.imag,
                '--', label="Im diff Krho dn")
        pl.legend()
        pl.show()
    
    
    #K_dn = xkernel.load_kernel(K, k, path_dn_left.points(), "dn")
    #K_up = xkernel.TabulatedKernels(K, k, path_up.points(), "up")
    print ("tabulate kernels dn")
    #K_dn = xkernel.TabulatedKernels(K, k, path_dn.points(), "dn")

    #show_kernels(K_up, K_dn, path_up, path_dn)
    
    print ("done")
    now = time.time()
    print ("make diffuse source")
    #Krho_sta = K_up.rho_star()
    #Xo_star  = K_up.omega_star()
    diff_flow  = DiffuseFlow(k, K_up, path_up, K_dn, path_dn)
    diff_flow_sym  = DiffuseFlow_sym(k, K_up, path_up, K_dn, path_dn)
    print ("create current source")
    if np.abs(h) < 0.001:
        inj_flow = EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn)
    else:
        inj_flow = InjectedFlow(h, k, K_up, path_up, K_dn, path_dn)
    inj_edge   = EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn)
    inj_edge_sym   = EdgeInjectedFlow_sym(k, K_up, path_up, K_dn, path_dn)
    stokes_x   = Stokeslet(1, 0, h, k, K_up, path_up, K_dn, path_dn)
    stokes_y   = Stokeslet(0, 1, h, k, K_up, path_up, K_dn, path_dn)
    inj_sin     = EdgeSinFlow(k, K_up, path_up, K_dn, path_dn)
    inj_sin_sym = EdgeSinFlow(k, K_up, path_up, K_dn, path_dn)
    print ("done", time.time() - now)
    all_flows  = {
          'inj-sin'      : inj_sin,
          'inj-sin-sym'  : inj_sin_sym,
          'inj-edge-sym' : inj_edge_sym, 
          'diffuse-sym'  : diff_flow_sym,
          'inj-edge' : inj_edge, 
          'diffuse'  : diff_flow,
          'inj-bulk' : inj_flow,
          'stokes_x' : stokes_x,
          'stokes_y' : stokes_y,
    }
    results = {}
    q_up = path_up.points()
    q_dn = path_dn.points()
    for label in all_flows.keys():
        flow = all_flows[label]
        #continue

        print ("show rho and j"); now = time.time()
        test_up(flow, label, gamma, gamma1, k, h, K, path_up)
        test_dn(flow, label, gamma, gamma1, k, h, K, path_dn)

        test_Dstar(flow, label, path_up, path_dn)

        #test_rho_y(flow, label, yv, path_up, path_dn)
        test_rho_y_new(flow, label, yv)
        test_j_y(flow, label, yv)
        print ("done", time.time() - now)
        import pylab as pl; pl.show()
        #res = dict()
        #wall_flux = flow.wall_flux()
        #rho = flow.rho(yv)
        #jx, jy = flow.current(yv)
        #res['jx'] = jx
        #res['jy'] = jy
        #res['flux'] = wall_flux
        #res['rho'] = rho
        #results[label] = res
    #return results

    for label in all_flows.keys():
        if label == 'diffuse': continue
        print ("make combined flow"); now = time.time()
        flow_bare = all_flows[label]
        flow_diff = diff_flow
        phi_bare = flow_bare.wall_flux()
        phi_diff = diff_flow.wall_flux()
        f_s = phi_bare / (1.0/np.pi - phi_diff)
        print ("bare flux: ", phi_bare, "diff: ", phi_diff)
        print ("flow: ", label, "f_s = ", f_s)
        full_flow = CombinedFlow(k, K_up, path_up, K_dn, path_dn)
        full_flow.add(flow_bare, 1.0)
        full_flow.add(flow_diff, f_s)

        full_label = "combined-%s" % label
        test_Dstar(full_flow, full_label, path_up, path_dn)
        test_rho_y_new(full_flow, full_label, yv)
        test_j_y(full_flow, full_label, yv)
        print ("done", time.time() - now)
        test_dn(full_flow, full_label, gamma, gamma1, k, h, K, path_dn)
        import pylab as pl; pl.show()
        test_dn(full_flow, full_label, gamma, gamma1, k, h, K, path_dn)

    #src_flux  = inj_up.wall_flux()
    #diff_flux = diff_up.wall_flux()
    #sin_flux  = stokes_y.wall_flux()
    #cos_flux  = stokes_x.wall_flux()
    #rho_diff  = diff_up.rho(yv)
    #rho_inj   = inj_up.rho(yv)

#h = 0.0
h = 1.2
k = 3.0
gamma = 1.0
gamma1 = 0.6
yv = np.linspace(-3.0, 30.0, 3301)
test_wh(h, k, gamma, gamma1, yv)
