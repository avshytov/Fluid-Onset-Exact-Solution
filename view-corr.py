import numpy as np
import pylab as pl
import sys
import fit

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

def get_Rvic(fname, x):
    print ("get Rvic from ", fname)
    d = np.load(fname)
    k = d['k']
    y = d['y']
    df_k = d['corr_tot:f_s']
    i_zero = np.argmin(np.abs(y))
    drho_k = d['corr_tot:rho'][:, i_zero]
    
    kmin = 0.0003 * 0.999
    kmax = 0.0003 * 1.001
    i_incl = [t for t in range(len(k)) if np.abs(k[t]) < kmin or np.abs(k[t]) > kmax]
    k_excl = np.array([k[t] for t in range(len(k)) if t not in i_incl])
    df_k_new = np.array([df_k[t] for t in i_incl])
    drho_k_new = np.array([drho_k[t] for t in i_incl])
    k_new = np.array([k[t] for t in i_incl])

    print ("excluded:", k_excl)
    
    F = fit.Fourier(k_new, x)

    eps = 0.1
    f_inv_k = 2.0/k_new * np.exp(-eps * np.abs(k_new))
    f_inv_x = np.arctan(x/eps) / np.pi * 2.0
    df_x   = np.dot(F, df_k_new - f_inv_k).real + f_inv_x
    drho_x = np.dot(F, drho_k_new - f_inv_k).real + f_inv_x

    return df_x, drho_x


def compareData(fnames):
    x = np.linspace(-10.0, 10.0, 5000)
    data = []
    for fname in fnames:
        df, drho = get_Rvic(fname, x)
        data.append((fname, df, drho))
    pl.figure()
    for fname, df, drho in data:
        pl.plot(x, df, label=fname)
    pl.legend()
    pl.title("Edge flux")
    pl.figure()
    for fname, df, drho in data:
        pl.plot(x, drho, label=fname)
    pl.legend()
    pl.title("edge density")
    pl.show()

    

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
    jx  = d['orig:jx']
    jy  = d['orig:jy']

    print ("df_s = ", df_s)

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

    k_eo, dfI_even, dfI_odd = even_odd(k, dfs_I)
    pl.figure()
    pl.plot(k, dfs_I.real, label='Re df_I(k)')
    pl.plot(k, dfs_I.imag, label='Im df_I(k)')
    pl.plot(k_eo, dfI_even.real, label='Re dfI_even(k)')
    pl.plot(k_eo, dfI_even.imag, label='Im dfI_even(k)')
    pl.plot(k_eo, dfI_odd.real, '--', label='Re dfI_odd(k)')
    pl.plot(k_eo, dfI_odd.imag, '--', label='Im dfI_odd(k)')
    pl.legend()

    
    k_eo, dfsd_even, dfsd_odd = even_odd(k, dfs_diff)
    pl.figure()
    pl.plot(k, dfs_diff.real, label='Re df_s(k)')
    pl.plot(k, dfs_diff.imag, label='Im df_s(k)')
    pl.plot(k_eo, dfsd_even.real, label='Re dfsd_even(k)')
    pl.plot(k_eo, dfsd_even.imag, label='Im dfsd_even(k)')
    pl.plot(k_eo, dfsd_odd.real, '--', label='Re dfsd_odd(k)')
    pl.plot(k_eo, dfsd_odd.imag, '--', label='Im dfsd_odd(k)')
    pl.legend()
    
    pl.figure()
    pl.plot(k, fs_orig.real,    label='Re f_s(k)')
    pl.plot(k, fs_orig.imag,    label='Im f_s(k)')
    pl.plot(k_eo, fs_even.real, '--', label='Re f_even(k)')
    pl.plot(k_eo, fs_even.imag, '--', label='Im f_even(k)')
    pl.plot(k_eo, fs_odd.real,  label='Re f_odd(k)')
    pl.plot(k_eo, fs_odd.imag,  label='Im f_odd(k)')
    pl.legend()

    pl.figure()
    pl.loglog(np.abs(k), np.abs(fs_orig), label='f_s')
    pl.loglog(np.abs(k), np.abs(df_s), label='|df_s|')
    pl.loglog(np.abs(k), np.abs(df_s.real), label='Re df_s')
    pl.loglog(np.abs(k), np.abs(df_s.imag), label='Im df_s')
    pl.loglog(np.abs(k), np.abs(df_s - 1j * p_fit[0]/k),
              label='df - 1/4 sgn k')
    pl.legend()


    x = np.linspace(-10.0, 10.0, 5001)
    F = fit.Fourier(k, x)
    src_k = np.exp(-0.0001*k**2)
    eps = 0.05
    # make the integrand regular by subtracting the leading 1/k singularity
    f_invk = 1j * p_fit[0]/k * np.exp(-eps*np.abs(k))
    # Fourier transform of f_invk
    f_invk_x = p_fit[0]/np.pi * np.arctan(x/eps)
    df_x = np.dot(F, (df_s - f_invk) * src_k) + f_invk_x
    #p_fit[0]/2.0 * np.sign(x)
    df0_x = np.dot(F, df_s * src_k)
    f_x = np.dot(F, fs_orig * src_k)
    DRHO = np.dot(F, drho)
    DJX  = np.dot(F, djx)
    DJY  = np.dot(F, djy)
    PSI2 = np.dot(F, jy / (1j * k[:, None]))
    JX  = np.dot(F, jx)
    JY  = np.dot(F, jy)
    i_zero = np.argmin(np.abs(d['y']))
    print ("i_zero = ", i_zero)
    drho_x = np.dot(F, drho[:, i_zero])

    Y, X = np.meshgrid(d['y'], x)

    DPSI = 0.0 * X + 0.0j
    PSI = 0.0 * X + 0.0j
    for j in range(1, len(y)):
        djx_half = (DJX[:, j] + DJX[:, j - 1])/2.0
        jx_half =  (JX[:, j] + JX[:, j - 1])/2.0
        dy = y[j] - y[j - 1]
        DPSI[:, j] = DPSI[:, j - 1] + djx_half * dy
        PSI[:, j] = PSI[:, j - 1] + jx_half * dy

    DPSI = np.nan_to_num(DPSI, nan=0.0)
    import matplotlib.colors as mpc
    class Custom_Norm(mpc.Normalize):
        def __init__(self, vmin, vzero, vmax):
            self.vmin = vmin
            self.vmax = vmax
            self.vzero = vzero
        def __call__ (self, value, clip=None):
             x, y = [self.vmin, self.vzero, self.vmax], [0, 0.5, 1]
             return np.ma.masked_array(np.interp(value, x, y))
    maxv = 1.2
    custom_norm = Custom_Norm(-maxv, 0.0, maxv)     
    pl.figure()
    pl.pcolormesh(X, Y, DRHO.real, cmap='bwr',
                  norm=custom_norm, shading='auto')
    pl.colorbar()
    pl.gca().set_aspect('equal', 'box')

    pl.figure()
    pl.contour(X, Y, DPSI.real, 31, cmap='jet')
    pl.colorbar()
    pl.gca().set_aspect('equal', 'box')

    B = 0.25
    pl.figure()
    pl.contour(X, Y, (PSI2 + B * DPSI).real, 31, cmap='jet')
    pl.colorbar()
    pl.gca().set_aspect('equal', 'box')

    pl.figure()
    pl.plot(x, df_x.real, label='Re df(x)')
    pl.plot(x, df_x.imag, label='Im df(x)')
    pl.plot(x, df0_x.real, '--', label='Re df(x), no fit')
    pl.plot(x, df0_x.imag, '--', label='Im df(x), no fit')
    pl.plot(x, f_x.real, label='Re f(x)')
    pl.plot(x, f_x.imag, label='Im f(x)')
    pl.plot(x, drho_x.real, label='drho(x)')
    pl.plot(x, drho_x.imag, label='drho(x)')
    pl.legend()

    pl.figure()
    pl.plot(x, DJX[:, i_zero].real, label='Re dj_x')
    pl.plot(x, DJX[:, i_zero].imag, label='Im dj_x')
    pl.plot(x, DJY[:, i_zero].real, label='Re dj_y')
    pl.plot(x, DJY[:, i_zero].imag, label='Im dj_y')
    pl.legend()

    pl.figure()
    for y_i in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        i_y = np.argmin(np.abs(y - y_i))
        pl.plot(x, DRHO[:, i_y].real, label='Re drho @ y=%g' % y[i_y])
        pl.plot(x, DRHO[:, i_y].imag, label='Im drho' % y[i_y])
    pl.legend()
    pl.show()

#for f in sys.argv[1:]:
if len(sys.argv) == 2:
    readData(f)
else:
    compareData(sys.argv[1:])
    
