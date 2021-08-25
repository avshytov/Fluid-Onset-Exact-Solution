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

    pl.figure()
    pl.plot(k, df_s.real, label='Re df(k)')
    pl.plot(k, df_s.imag, label='Im df(k)')
    pl.plot(k_eo, df_even.real, label='Re df_even(k)')
    pl.plot(k_eo, df_even.imag, label='Im df_even(k)')
    pl.plot(k_eo, df_odd.real, '--', label='Re df_odd(k)')
    pl.plot(k_eo, df_odd.imag, '--', label='Im df_odd(k)')
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
    pl.loglog(np.abs(k), np.abs(df_s), label='df_s')
    pl.loglog(np.abs(k), np.abs(df_s - 0.248 * 1j * np.sign(k)),
              label='df - 1/4 sgn k')
    pl.legend()


    x = np.linspace(-25.0, 25.0, 5001)
    F = fit.Fourier(k, x)
    src_k = np.exp(-0.003*k**2)
    df_x = np.dot(F, df_s * src_k)
    f_x = np.dot(F, fs_orig * src_k)
    DRHO = np.dot(F, drho)
    DJX  = np.dot(F, djx)
    DJY  = np.dot(F, djy)
    i_zero = np.argmin(np.abs(d['y']))
    print ("i_zero = ", i_zero)
    drho_x = np.dot(F, drho[:, i_zero])

    Y, X = np.meshgrid(d['y'], x)

    DPSI = 0.0 * X + 0.0j
    for j in range(1, len(y)):
        jx_half =  -(DJX[:, j] + DJX[:, j - 1])/2.0
        dy = y[j] - y[j - 1]
        DPSI[:, j] = DPSI[:, j - 1] + jx_half * dy

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
    maxv = 0.2
    custom_norm = Custom_Norm(-maxv, 0.0, maxv)     
    pl.figure()
    pl.pcolormesh(X, Y, DRHO.real, cmap='bwr',
                  norm=custom_norm, shading='auto')
    pl.colorbar()

    pl.figure()
    pl.contour(X, Y, DPSI.real, 31)

    pl.figure()
    pl.plot(x, df_x.real, label='Re df(x)')
    pl.plot(x, df_x.imag, label='Im df(x)')
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

    
    pl.show()

for f in sys.argv[1:]:
    readData(f)

    
