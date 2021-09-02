import numpy as np
import pylab as pl
import sys
import fit
from scipy import linalg
from scipy import special
import matplotlib.patches as patches

global Fourier_F, Fourier_k, Fourier_x

Fourier_F = None
Fourier_k = None
Fourier_x = None

def _get_Fourier(k, x):
    global Fourier_F, Fourier_k, Fourier_x
    print ("compute the Fourier kernel", k, x)
    Fourier_F =  fit.Fourier(k, x)
    Fourier_k = k
    Fourier_x = x
    return Fourier_F

def get_Fourier(k, x):
    global Fourier_F, Fourier_k, Fourier_x
    if type(Fourier_F) != type(np.array([0.0])):
        print ("type mismatch", type(Fourier_F))
        return _get_Fourier(k, x)
    if len(Fourier_k) != len(k):
        print ("k len mismatch")
        return _get_Fourier(k, x)
    if linalg.norm(k - Fourier_k) > 1e-6:
        print ("k  mismatch")
        return _get_Fourier(k, x)
    if len(Fourier_x) != len(x):
        print ("x len mismatch")
        return _get_Fourier(k, x)
    if linalg.norm(x - Fourier_x) > 1e-6:
        print ("x  mismatch")
        return _get_Fourier(k, x)
    print ("use precomputed Fourier kernel")
    return Fourier_F

def even_odd(kvals, f):
    k_eo   = []
    f_even = []
    f_odd  = []
    for i_cur, k in enumerate(kvals):
        i_opp = np.argmin(np.abs(kvals + k))
        if abs(k + kvals[i_opp]) < 1e-6:
            f_even.append(0.5 * (f[i_cur] + f[i_opp]))
            f_odd.append(0.5 * (f[i_cur] - f[i_opp]))
            k_eo.append(k)
    return np.array(k_eo), np.array(f_even), np.array(f_odd)

def integrate_invk(k, f, x, F = None):
    def fit_inv_small(k, A, B, C):
        return A / k + B * np.sign(k) + C * k
    def fit_inv_large(k, A, B, C):
        return A/k + B * np.sign(k) / k**2 + C / k**3
    k_fit_small = np.linspace(0.001, 0.1, 501)
    k_fit_large = np.linspace(100.0, 300, 201)
    try:
       p_fit_small, p_cov_small, f_fit_sm = fit.do_restricted_fit(k, f.imag,
                                           0.001, 0.01, k_fit_small,
                                           fit_inv_small)
       A_small = p_fit_small[0]
       print ("fit small: ", p_fit_small, p_cov_small)
    except:
        import traceback
        traceback.print_exc()
        A_small = 0.0
        
    try:
       p_fit_large, p_cov_large, f_fit_lrg = fit.do_restricted_fit(k, f.imag,
                                                -200, -100.0, k_fit_large,
                                                fit_inv_large)
       A_large = p_fit_large[0]
       print ("fit_large: ", p_fit_large, p_cov_large)
    except:
        import traceback
        traceback.print_exc()
        A_large = 0.0
    #A_small = p_fit_small[0]
    #A_large = p_fit_large[0]
    eps = 0.1
    a   = 10.0
    # Subtract small-k behaviour which yields a step at large distances
    # Use exp to suppress its effect at large k
    df_small = A_small / k * np.exp(-eps * np.abs(k))
    # Subtract large-k behaviour which is responsible for a step
    # at short distances
    # Use a to make it regular and vanishing at small k
    df_large = A_large * k / (k**2 + a**2)
    # Fourier images of the functions above
    df_x_small = A_small / np.pi * np.arctan(x/eps)
    df_x_large = A_large / 2.0 * np.sign(x) * np.exp( - np.abs(a * x) )
    # df should not have 1/k singularities, neither at k = 0 or at k = infty
    df = f - 1j * df_small - 1j * df_large
    if type(F) != type(np.array([0.0])):
       print ("F = ", type(F), "recalculate the Fourier transform")
       #F = fit.Fourier(k, x)
       F = get_Fourier(k, x)
    df_x = np.dot(F, df)
    f_x = df_x + df_x_small + df_x_large
    return f_x

def get_Rvic(fname, x):
    print ("get Rvic from ", fname)
    d = np.load(fname)
    k = d['k']
    y = d['y']
    src_k = np.exp(-0.00005 * k**2)
    df_k = d['corr_tot:f_s']
    f_k = d['orig:f_s'] * src_k
    rho1_k = d['orig:drho']
    i_zero = np.argmin(np.abs(y))
    rho1_k = rho1_k[:, i_zero + 1] * src_k
    drho_k = d['corr_tot:rho'][:, i_zero]
    
    #kmin = 0.0003 * 0.999
    #kmax = 0.0003 * 1.001
    #i_incl = [t for t in range(len(k)) if np.abs(k[t]) < kmin or np.abs(k[t]) > kmax]
    i_incl = range(len(k)) #[t for t in range(len(k))]
    k_excl = np.array([k[t] for t in range(len(k)) if t not in i_incl])
    df_k_new = np.array([df_k[t] for t in i_incl])
    drho_k_new = np.array([drho_k[t] for t in i_incl])
    k_new = np.array([k[t] for t in i_incl])

    print ("excluded:", k_excl)
    
    #F = fit.Fourier(k_new, x)
    F     = get_Fourier(k_new, x)

    eps = 0.1
    #f_inv_k = 2.0/k_new * np.exp(-eps * np.abs(k_new))
    #f_inv_x = np.arctan(x/eps) / np.pi * 2.0
    #df_x   = np.dot(F, df_k_new - f_inv_k).real + f_inv_x
    #drho_x = np.dot(F, drho_k_new - f_inv_k).real + f_inv_x
    df_x   = integrate_invk(k_new, df_k_new,   x, F)
    drho_x = integrate_invk(k_new, drho_k_new, x, F)
    f_x    = np.dot(F, f_k)

    rho1_x  = np.dot(F, rho1_k)
    gamma = d['gamma']
    r = np.sqrt(x**2 + y[i_zero + 1]**2)
    rho_x = rho1_x + 1.0 / np.pi / r  * np.exp( - gamma * r)

    f_ref = 0.5 * (f_x[0] + f_x[-1])
    f_x -= f_ref
    rho_x -= f_ref

    return df_x, drho_x, f_x, rho_x, d['gamma'], d['gamma1']


def compareData(fnames):
    x = np.linspace(-10.0, 10.0, 5000)
    data = []
    for fname in fnames:
        df, drho, f, rho, gamma, gamma1 = get_Rvic(fname, x)
        data.append((fname, df, drho, f, rho, gamma, gamma1))
    data.sort(key = lambda x: -x[6])

    pl.figure()
    for fname, df, drho, f, rho, gamma, gamma1 in data:
        pl.plot(x, f.real, label=fname)
    pl.legend()
    pl.title("Edge flux (B = 0)")
    
    pl.figure()
    for fname, df, drho, f, rho, gamma, gamma1 in data:
        pl.plot(x, df.real, label=fname)
    pl.legend()
    pl.title("Edge flux(Hall)")

    pl.figure()
    for fname, df, drho, f, rho, gamma, gamma1 in data:
        pl.plot(x, rho.real, label=fname)
    pl.legend()
    pl.title("edge density")
    
    pl.figure()
    for fname, df, drho, f, rho, gamma, gamma1 in data:
        pl.plot(x, drho.real, label=fname)
    pl.legend()
    pl.title("edge density (Hall)")

    pl.figure()
    for x_p in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]:
        pass
        i_p = np.argmin(np.abs(x_p - x))
        V_p      = []
        phi_p    = []
        dV_p     = []
        dphi_p   = []
        gamma2_p = []
        for fname, df, drho, f, rho, gamma, gamma1 in data:
            gamma2_p.append(1.0 - gamma1)
            dV_p.append(df[i_p].real)
            dphi_p.append(drho[i_p].real)
            V_p.append(f[i_p].real)
            phi_p.append(rho[i_p].real)
        pl.plot(np.array(gamma2_p), np.array(V_p), label=r'$x_p = %g$' % x_p)
    pl.title("V(x) vs ohmicity, B=0")
    pl.xlabel(r"$\gamma''$")
    pl.legend()
    
    pl.figure()
    for x_p in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]:
        pass
        i_p = np.argmin(np.abs(x_p - x))
        V_p      = []
        phi_p    = []
        dV_p     = []
        dphi_p   = []
        gamma2_p = []
        for fname, df, drho, f, rho, gamma, gamma1 in data:
            gamma2_p.append(1.0 - gamma1)
            V_p.append(f[i_p].real)
            phi_p.append(rho[i_p].real)
            dV_p.append(df[i_p].real)
            dphi_p.append(drho[i_p].real)
        pl.plot(np.array(gamma2_p), np.array(dV_p), label=r'$x_p = %g$' % x_p)
    pl.title("V(x) vs ohmicity (Hall)")
    pl.xlabel(r"$\gamma''$")
    pl.legend()
    
    pl.figure()
    for x_p in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]:
        pass
        i_p = np.argmin(np.abs(x_p - x))
        V_p      = []
        phi_p    = []
        dV_p      = []
        dphi_p    = []
        gamma2_p = []
        for fname, df, drho, f, rho, gamma, gamma1 in data:
            gamma2_p.append(1.0 - gamma1)
            V_p.append(f[i_p].real)
            phi_p.append(rho[i_p].real)
            dV_p.append(df[i_p].real)
            dphi_p.append(drho[i_p].real)
        pl.plot(np.array(gamma2_p), np.array(phi_p), label='$x = %g$' % x_p)
    pl.title(r"$\phi(x)$ vs ohmicity (B = 0)")
    pl.xlabel(r"$\gamma''$")
    pl.legend()
    pl.figure()
    for x_p in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]:
        pass
        i_p = np.argmin(np.abs(x_p - x))
        V_p      = []
        phi_p    = []
        dV_p     = []
        dphi_p   = []
        gamma2_p = []
        for fname, df, drho, f, rho, gamma, gamma1 in data:
            gamma2_p.append(1.0 - gamma1)
            V_p.append(f[i_p].real)
            phi_p.append(rho[i_p].real)
            dV_p.append(df[i_p].real)
            dphi_p.append(drho[i_p].real)
        pl.plot(np.array(gamma2_p), np.array(dphi_p), label='$x = %g$' % x_p)
    pl.title(r"$\phi(x)$ vs ohmicity (Hall)")
    pl.xlabel(r"$\gamma''$")
    pl.legend()
    pl.show()

import matplotlib.colors as mpc
class Custom_Norm(mpc.Normalize):
        def __init__(self, vmin, vzero, vmax):
            self.vmin = vmin
            self.vmax = vmax
            self.vzero = vzero
        def __call__ (self, value, clip=None):
             x, y = [self.vmin, self.vzero, self.vmax], [0, 0.5, 1]
             return np.ma.masked_array(np.interp(value, x, y))
         
def show_rho_and_psi(X, Y, rho, psi, maxv, lab):
    pl.figure()
    PSI_R = psi.real
    levs = np.linspace(np.min(PSI_R), np.max(PSI_R), 21)
    #levs = 0.5 * (levs[1:] + levs[:-1])
    cs = pl.contour(X, Y, PSI_R, levs, cmap='jet')
    #pl.colorbar()
    #pl.gca().set_aspect('equal', 'box')
    pl.clf()
    
    #DPSI = 0.0 * X + 0.0j
    #PSI = 0.0 * X + 0.0j
    #for j in range(1, len(y)):
    #    djx_half = (DJX[:, j] + DJX[:, j - 1])/2.0
    #    jx_half =  (JX[:, j] + JX[:, j - 1])/2.0
    #    dy = y[j] - y[j - 1]
    #    DPSI[:, j] = DPSI[:, j - 1] + djx_half * dy
    #    PSI[:, j] = PSI[:, j - 1] + jx_half * dy

    #DPSI = np.nan_to_num(DPSI, nan=0.0)
    #maxv = 1.2
    #custom_norm = Custom_Norm(-maxv, 0.0, maxv)     
    #pl.figure()

    pl.pcolormesh(X, Y, rho.real, cmap='bwr',
                  norm = Custom_Norm(-maxv, 0.0, maxv),
                  shading='auto')
    cb = pl.colorbar()
    pl.gca().set_aspect('equal', 'box')
    cb.set_label(lab)

    src_w = 0.1
    corner_ll = (np.min(X), np.min(Y))
    left_wid  = -src_w - np.min(X)
    hei = 0.0 - np.min(Y)
    right_wid = np.max(X) - src_w
    corner_src_ll = (-src_w, np.min(Y))
    corner_src_lr = (src_w, np.min(Y))
    rect_left  = patches.Rectangle(corner_ll, left_wid, hei,
                                   edgecolor='0.5', facecolor='0.8')
    rect_right = patches.Rectangle(corner_src_lr, right_wid, hei,
                                   edgecolor='0.5', facecolor='0.8')
    rect_mid   = patches.Rectangle(corner_src_ll, 2.0*src_w, hei,
                                   edgecolor='black', facecolor='red')
    pl.gca().add_patch(rect_left)
    pl.gca().add_patch(rect_right)
    pl.gca().add_patch(rect_mid)


    segments = cs.allsegs
    levels = cs.levels
    for i in range(0, len(segments)):
        polygons = segments[i]
        for poly in polygons:
            x_seg = np.array([t[0] for t in poly])
            y_seg = np.array([t[1] for t in poly])
            pl.plot(x_seg, y_seg, 'k-')
    pl.xlabel(r"$x/l_\mathrm{ee}$")
    pl.ylabel(r"$y/l_\mathrm{ee}$")

    

def readData(fname):
    d = np.load(fname)
    for k in d.keys(): print(k)

    k = d['k']
    y = d['y']
    fs_orig = d['orig:f_s'].flatten()
    df_s    = d['corr_tot:f_s'].flatten()
    dfs_I   = d['corr_I:f_s'].flatten()
    dfs_diff = d['corr_diff:f_s'].flatten()
    k_eo, fs_even, fs_odd = even_odd(k, fs_orig)
    k_eo, df_even, df_odd = even_odd(k, df_s)
    drho = d['corr_tot:rho']
    djx  = d['corr_tot:jx']
    djy  = d['corr_tot:jy']
    djy_I = d['corr_I:jy']
    djy_s = d['corr_diff:jy']
    jx  = d['orig:jx']
    jy  = d['orig:jy']
    rho = d['orig:rho']
    gamma = d['gamma']
    if 'orig:drho' in d.keys():
       drho0 = d['orig:drho']
    else:
       drho0 = None
    #drho0 = d['orig:drho'] 

    print ("df_s = ", df_s)

    if True:
        pl.figure()
        i0 = np.argmin(np.abs(y)) + 1
        pl.plot(k, djy_I[:, i0].real, label='Re djy_I')
        pl.plot(k, djy_I[:, i0].imag, label='Im djy_I')
        pl.plot(k, djy_s[:, i0].real, label='Re djy_s')
        pl.plot(k, djy_s[:, i0].imag, label='Im djy_s')
        pl.legend()
        pl.figure()
        pl.loglog(np.abs(k), np.abs(djy_I[:, i0]), label='|djy_I|')
        pl.loglog(np.abs(k), np.abs(djy_s[:, i0]), label='|djy_s|')
        pl.legend()
        #pl.show()

    def fit_inv(k, A, B, C):
        return A / k + B * np.sign(k) + C * k
    k_fit = np.linspace(0.001, 0.1, 501)
    p_fit, p_cov, fs_fit = fit.do_restricted_fit(k, df_s.imag,
                                                0.001, 0.01, k_fit, fit_inv)

    print("fit: ", p_fit)
    #fit_func = fit.fit_inv
    
    #fs_fit = fit.do_fit(k, df_s, 0.0,
    #           0.001, 0.01, k_fit,
    #           fit.fit_inv, fit.fit_inv, True, None)

    pl.figure()
    pl.plot(k, df_s.real, label='Re df(k)')
    pl.plot(k, df_s.imag, label='Im df(k)')
    #pl.plot(k_eo, df_even.real, label='Re df_even(k)')
    #pl.plot(k_eo, df_even.imag, label='Im df_even(k)')
    #pl.plot(k_eo, df_odd.real, '--', label='Re df_odd(k)')
    #pl.plot(k_eo, df_odd.imag, '--', label='Im df_odd(k)')
    #pl.plot(k_fit, fs_fit.real, '--', label='Re fit')
    pl.plot(k_fit, fs_fit, '--', label='Im fit')
    #pl.plot(k, df_s.imag - fit_inv(k, *p_fit), label='diff')
    pl.plot(k, df_s.imag - fit_inv(k, *p_fit), label='diff')
    pl.plot(k, df_s.imag - p_fit[0] / k, label='f - A/k')
    pl.legend()
    #pl.show()
    pl.figure()
    pl.plot(k, df_s.imag * k, label='k * Im f_s')
    pl.legend()

    #k_eo, dfI_even, dfI_odd = even_odd(k, dfs_I)
    #pl.figure()
    #pl.plot(k, dfs_I.real, label='Re df_I(k)')
    #pl.plot(k, dfs_I.imag, label='Im df_I(k)')
    #pl.plot(k_eo, dfI_even.real, label='Re dfI_even(k)')
    #pl.plot(k_eo, dfI_even.imag, label='Im dfI_even(k)')
    #pl.plot(k_eo, dfI_odd.real, '--', label='Re dfI_odd(k)')
    #pl.plot(k_eo, dfI_odd.imag, '--', label='Im dfI_odd(k)')
    #pl.legend()

    
    #k_eo, dfsd_even, dfsd_odd = even_odd(k, dfs_diff)
    #pl.figure()
    #pl.plot(k, dfs_diff.real, label='Re df_s(k)')
    #pl.plot(k, dfs_diff.imag, label='Im df_s(k)')
    #pl.plot(k_eo, dfsd_even.real, label='Re dfsd_even(k)')
    #pl.plot(k_eo, dfsd_even.imag, label='Im dfsd_even(k)')
    #pl.plot(k_eo, dfsd_odd.real, '--', label='Re dfsd_odd(k)')
    #pl.plot(k_eo, dfsd_odd.imag, '--', label='Im dfsd_odd(k)')
    #pl.legend()
    
    #pl.figure()
    #pl.plot(k, fs_orig.real,    label='Re f_s(k)')
    #pl.plot(k, fs_orig.imag,    label='Im f_s(k)')
    #pl.plot(k_eo, fs_even.real, '--', label='Re f_even(k)')
    #pl.plot(k_eo, fs_even.imag, '--', label='Im f_even(k)')
    #pl.plot(k_eo, fs_odd.real,  label='Re f_odd(k)')
    #pl.plot(k_eo, fs_odd.imag,  label='Im f_odd(k)')
    #pl.legend()

    pl.figure()
    pl.loglog(np.abs(k), np.abs(fs_orig), label='f_s')
    pl.loglog(np.abs(k), np.abs(df_s), label='|df_s|')
    pl.loglog(np.abs(k), np.abs(df_s.real), label='Re df_s')
    pl.loglog(np.abs(k), np.abs(df_s.imag), label='Im df_s')
    pl.loglog(np.abs(k), np.abs(df_s - 1j * p_fit[0]/k),
              label='df - 1/4 sgn k')
    pl.legend()


    #x = np.linspace(-10.0, 10.0, 5001)
    x = np.linspace(-10.0, 10.0, 1001)
    #x = np.linspace(-10.0, 10.0, 51)
    #F = fit.Fourier(k, x)
    F = get_Fourier(k, x)
    src_k = np.exp(-0.0001*k**2)
    eps = 0.05
    # make the integrand regular by subtracting the leading 1/k singularity
    #f_invk = 1j * p_fit[0]/k * np.exp(-eps*np.abs(k))
    f_invk = 1j * 2.0/k * np.exp(-eps*np.abs(k))
    # Fourier transform of f_invk
    #f_invk_x = p_fit[0]/np.pi * np.arctan(x/eps)
    f_invk_x = 2.0/np.pi * np.arctan(x/eps)
    #df_x = np.dot(F, (df_s - f_invk) * src_k) + f_invk_x
    df_x = integrate_invk(k, df_s, x, F)
    #p_fit[0]/2.0 * np.sign(x)
    df0_x = np.dot(F, df_s * src_k)
    f_x = np.dot(F, fs_orig * src_k)
    DRHO = np.dot(F, drho - f_invk[:, None]) + f_invk_x[:, None]
    RHO  = np.dot(F, rho)

    if True:
        djy_Ix = np.dot(F, djy_I[:, i0])
        djy_sx = np.dot(F, djy_s[:, i0])
        pl.figure()
        pl.plot(x, djy_Ix.real, label='djy_I(x)')
        pl.plot(x, djy_sx.real, label='djy_s(x)')
        pl.legend()
        pl.show()

    if type(drho0) == type(None): #'orig:drho' not in d.keys():
       print ("calculate drho from K0")
       kappa = np.sqrt(k**2 + gamma**2)
       K0 = special.kn(0, np.outer(kappa, np.abs(y)))
       K0[:, y < 0] = 0.0
       drho0 = rho - 2.0 / np.pi * K0
    else:
       print ("use drho from file")
       if False:
          ddrho = rho - drho0
          for k0 in [0.01, 0.1, 0.5, 1.0, 3.0, 10.0, 20.0, 40.0, 100.0]:
              i_k = np.argmin(np.abs(k - k0))
              pl.figure()
              pl.plot(y, ddrho[i_k, :], label='ddrho')
              k_i = k[i_k]
              K_check = 2.0/np.pi * special.kn(0, np.sqrt(gamma**2 + k_i**2) * np.abs(y))
              pl.plot(y, K_check, '--', label='K0')
              pl.plot(y, K_check - ddrho[i_k, :], label='diff')
              pl.title("check for k0 = %g" % k0 )
              pl.legend()
          pl.show()
    DRHO0 = np.dot(F, drho0)
    R2  = np.outer(x**2, np.ones(np.shape(y)))
    R2 += np.outer(np.ones(np.shape(x)), y**2)
    R = np.sqrt(R2)
    RHO0 = DRHO0 + 1.0 / np.pi / R * np.exp(-gamma * R)
    RHO0 = np.nan_to_num(RHO0, nan=0.0)
    RHO0[:, y < -0.001] = 0.0
    DJX  = np.dot(F, djx)
    DJY  = np.dot(F, djy)
    PSI2 = np.dot(F,  jy / (1j * k[:, None]))
    DPSI = np.dot(F, djy / (1j * k[:, None]))
    JX   = np.dot(F, jx)
    JY   = np.dot(F, jy)
    i_zero = np.argmin(np.abs(d['y']))
    print ("i_zero = ", i_zero)
    drho_x = DRHO[:, i_zero + 1]
    rho_x  = np.array(RHO0[:, i_zero + 1])
    
    #rho0_ref = (RHO0[0, i_zero] + RHO0[-1, i_zero]) * 0.5
    rho_l = np.average(RHO0[0, i_zero:]).real
    rho_r = np.average(RHO0[-1, i_zero:]).real
    len_lr = y[-1]
    rho_top = np.average(RHO0[:, -1]).real
    len_top = x[-1] - x[0]
    rho0_ref = ((rho_l + rho_r) * len_lr +rho_top*len_top)/(2*len_lr + len_top)
    print ("ref potential: ", rho0_ref)
    RHO0[:, i_zero:]  -= rho0_ref
    rho_x             -= rho0_ref
    
    fx_ref = rho0_ref
    #fx_ref = (f_x[0] + f_x[-1]).real / 2.0
    f_x -= fx_ref

    #drho_x = np.dot(F, drho[:, i_zero])
    #rho_x   = np.dot(F, rho[:, i_zero])
    print ("shape(y) = ", np.shape(y), "shape(psi) = ", np.shape(DPSI))
    DPSI -= 0.5 * (DPSI[0, i_zero] + DPSI[-1, i_zero])
    DPSI[:, :i_zero] = 0.0

    Y, X = np.meshgrid(d['y'], x)

    B = 0.2
    show_rho_and_psi(X, Y, DRHO, PSI2 + B * DPSI, 1.2, r'$\delta\phi(x, y)$')
    pl.title("Perturbed flow, Hall potential")
    show_rho_and_psi(X, Y, RHO0 + B * DRHO, PSI2 + B * DPSI, 0.1,
                     r'\phi(x, y)')
    pl.title("Perturbed flow, full potential")
    show_rho_and_psi(X, Y, RHO0, PSI2, 0.01, r'$\phi_0(x, y)$')
    pl.title("Unperturbed flow and potential")
    #pl.figure()
    #pl.pcolormesh(X, Y, (RHO + B * DRHO).real, cmap='bwr',
    #              norm=Custom_Norm(-0.1, 0.0, 0.1), shading='auto')
    #pl.colorbar()
    #pl.gca().set_aspect('equal', 'box')
    #segments = cs.allsegs
    #levels = cs.levels
    #for i in range(0, len(segments)):
    #    polygons = segments[i]
    #    for poly in polygons:
    #        x_seg = np.array([t[0] for t in poly])
    #        y_seg = np.array([t[1] for t in poly])
    #        pl.plot(x_seg, y_seg, 'k-')

    pl.figure()
    pl.contour(X, Y, DPSI.real, 31, cmap='jet')
    pl.colorbar()
    pl.gca().set_aspect('equal', 'box')
    pl.title("Hall contribution to the stream function")


    pl.figure()
    pl.plot(x, df_x.real, label=r'$\delta V(x)$')
    #pl.plot(x, df_x.imag, label='Im df(x)')
    #pl.plot(x, df0_x.real, '--', label='Re df(x), no fit')
    #pl.plot(x, df0_x.imag, '--', label='Im df(x), no fit')
    #pl.plot(x, f_x.real, label='Re f(x)')
    #pl.plot(x, f_x.imag, label='Im f(x)')
    pl.plot(x, drho_x.real, label=r'$\delta \phi(x)$')
    #pl.plot(x, drho_x.imag, label='$\phi(x)$')
    pl.legend()
    pl.xlabel(r"$x/l_\mathrm{ee}$")
    pl.ylabel(r"$\delta\phi(x)$")
    pl.title("Edge voltage and potential")

    pl.figure()
    pl.plot(x, DJX[:, i_zero].real, label=r'$\delta j_x(x)$')
    #pl.plot(x, DJX[:, i_zero].imag, label='Im dj_x')
    pl.plot(x, DJY[:, i_zero].real, label=r'$\delta j_y(x)$')
    #pl.plot(x, DJY[:, i_zero].imag, label='Im dj_y')
    pl.plot(x, JX[:, i_zero].real, label=r'$j_x(x)$')
    #pl.plot(x, DJX[:, i_zero].imag, label='Im dj_x')
    pl.plot(x, JY[:, i_zero].real, label=r'$j_y(x)$')
    #pl.plot(x, DJY[:, i_zero].imag, label='Im dj_y')
    pl.legend()
    pl.xlabel(r"$x/l_\mathrm{ee}$")
    pl.ylabel(r"$\delta j(x)$")
    pl.title("Hall current at the edge")
    

    pl.figure()
    for y_i in [0.0, 0.01, 0.1, 0.2, 0.3, 0.5, 1.0]:
        i_y = np.argmin(np.abs(y - y_i))
        pl.plot(x, DRHO[:, i_y].real, label=r'y=%g' % y[i_y])
        #pl.plot(x, DRHO[:, i_y].imag, label='Im drho' % y[i_y])
    pl.legend()
    pl.title(r"Bulk potential $\phi(x)$")
    pl.xlabel(r"$x/l_\mathrm{ee}$")
    pl.ylabel(r"$\phi(x)$")

    pl.figure()
    from matplotlib import cm
    cmap = cm.get_cmap('seismic')
    b_norm = Custom_Norm(-0.1, 0.0, 0.1)
    for b, col in [(-0.1, 'blue'), (0.0, 'black'), (0.1, 'red')]:
        #col = cmap(b_norm(b))
        #.to_rgba(Custom_Norm(-0.1, 0, 0.1))
        #if (abs(b) < 1e-5): c = 'black'
        pl.plot(x, f_x.real + b * df_x.real, color=col,
                label=r'$B = %g$' % b)
    pl.title("Edge voltage")
    pl.xlabel(r"$x/l_\mathrm{ee}$")
    pl.ylabel(r"$V(x)$")
    pl.legend()
    
    pl.figure()
    from matplotlib import cm
    cmap = cm.get_cmap('seismic')
    b_norm = Custom_Norm(-0.1, 0.0, 0.1)
    for b, col in [(-0.1, 'blue'), (0.0, 'black'), (0.1, 'red')]:
        #col = cmap(b_norm(b))
        #.to_rgba(Custom_Norm(-0.1, 0, 0.1))
        #if (abs(b) < 1e-5): c = 'black'
        pl.plot(x, rho_x.real  + b * drho_x.real, color=col,
                label=r'$B = %g$' % b)
    pl.title("Edge potential")
    pl.xlabel(r"$x/l_\mathrm{ee}$")
    pl.ylabel(r"$\phi(x)$")
    pl.legend()
    pl.show()

#for f in sys.argv[1:]:
if len(sys.argv) == 2:
    readData(sys.argv[1])
else:
    compareData(sys.argv[1:])
    
