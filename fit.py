import _makequad as mq
import numpy as np
from scipy import optimize


def Fourier(q, x):
    ret = mq.fourier_quad(q, x)
    for i in range(len(x)):
        for j in range(len(q)):
            if np.isnan(ret['Fre'][i, j]) or np.isnan(ret['Fim'][i, j]):
                print("quad nan", i, j, "x = ", x[i], "q = ", q[j])
    return ret['Fre'] - 1j * ret['Fim'] # change the sign: exp(-ik*r) now, WH convention

def fit_lin(k, A, B, C):
    return A + B * np.abs(k) + C * np.abs(k)**2

def fit_phij(k, A, B):
    return A * np.abs(k) + B * np.abs(k**2)

def fit_phis(k, A, B):
    return 1.0/np.pi + A * np.abs(k) + B * np.abs(k**2)

def make_fit_const(Const):
    def f_fit_const(k, A, B):
        return Const + A * np.abs(k) + B * k**2
    return f_fit_const

def fit_lin3(k, A, B, C, D):
    return A + B * np.abs(k) + C * np.abs(k)**2 + D * np.abs(k)**3

def fit_lambda(k, A, B, C):
    return A + B * np.abs(k)*(1 + np.abs(C*k))/(1.0 + 2*np.abs(C*k))

def fit_lin_zero(k, A, B):
    return A * np.abs(k) + B * np.abs(k)**2

def fit_inv_lin(k, A, B, C):
    return A / k + B + C * np.abs(k)


def fit_inv(k, A, B):
    return (A / k + B / k**2)

def fit_inv_log(k,  A, B, C, D):
    lnk = np.log(np.abs(k))
    return (A / k  + B / k**2 * lnk + C / k**2 + D)

def fit_inv_const(k, A, B, C):
    return (A + B/k + C/k**2)

def scale_exp(k, y):
    return np.exp(-np.abs(k)*y)

def fit_inv_exp(k, A, B, C, D):
    return (A + B/k + D/np.abs(k)**2)*np.exp(-C*np.abs(k))# + D/np.abs(k)


def do_restricted_fit(x, y, fit_min, fit_max, xs, fit_func, bounds=None):
    i_fit = [t for t in range(len(x)) if x[t] >= fit_min and x[t] <= fit_max]
    x_fit = np.array([x[t] for t in i_fit])
    y_fit = np.array([y[t] for t in i_fit])
    kwargs = {
        'maxfev' : 100000
    }
    if bounds:
        kwargs['bounds'] = bounds
    p_fit, p_cov = optimize.curve_fit(fit_func, x_fit, y_fit, **kwargs)
    return p_fit, p_cov, fit_func(xs, *p_fit)

def do_fit(x, y, yv, fit_min, fit_max, xs, fit_func_re, fit_func_im,
           show_params = False, bounds=None):
    i_fit = [t for t in range(len(x)) if x[t] >= fit_min and x[t] <= fit_max]
    x_fit = np.array([x[t] for t in i_fit])
    y_fit = np.array([y[t] for t in i_fit])
    if len(i_fit) < 4:
        return np.zeros(np.shape(xs), dtype=complex)
    kwargs = {
        'maxfev' : 100000,
        'ftol' : 1e-10, 
        #'method' : 'dogbox'
    }
    if bounds:
        kwargs['bounds'] = bounds
    p_fit_re, p_cov_re = optimize.curve_fit(fit_func_re, x_fit, y_fit.real, **kwargs)
    p_fit_im, p_cov_im = optimize.curve_fit(fit_func_im, x_fit, y_fit.imag, **kwargs)
    if show_params:
       print("fit re: ", p_fit_re, p_cov_re, fit_func_re)
       print("fit im: ", p_fit_im, p_cov_im, fit_func_im)
    
    #y_view = [0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    y_view = [0.29, 0.3, 0.31]
    #if (np.abs(yv - 0.01)< 0.005 ):
    #print "fit for", yv
    for y_v in y_view: 
      if False and (np.abs(yv - y_v) < 0.01):
        print("re fit: ", p_fit_re, p_cov_re)
        print("im fit: ", p_fit_im, p_cov_im)
        print("show", yv, y_v)
        pl.figure()
        pl.plot(x_fit, y_fit.real, label='Re data')
        pl.plot(xs, fit_func_re(xs, *p_fit_re), '--', label='Re fit')
        pl.plot(x_fit, fit_func_re(x_fit, *p_fit_re), '--', label='Re fit')
        pl.plot(x_fit, y_fit.imag, label='Im data')
        pl.plot(xs, fit_func_im(xs, *p_fit_im), '--', label='Im fit')
        pl.plot(x_fit, fit_func_im(x_fit, *p_fit_im), '--', label='Im fit')
        pl.title("fit @ %g" % yv)
        pl.show()
    return fit_func_re(xs, *p_fit_re) + 1j * fit_func_im(xs, *p_fit_im)

def fit_prepend(k, y, Z, fit_min, fit_max, data_min, ks, fit_func_re, fit_func_im):
    #N, M = np.shape(Z)
    #if M == 1:
    #    Z = np.transpose(Z)
    #print Z.ndim
    do_flatten = False
    if Z.ndim == 1:
        Z = np.array([Z]).transpose()
        do_flatten = True
    i_data = [t for t in range(len(k)) if k[t] >= data_min]
    k_data = np.array([k[t] for t in i_data])
    k_joint = list(ks)
    k_joint.extend(k_data)
    k_joint = np.array(k_joint)
    #print "k_joint, y", np.shape(k_joint), np.shape(y)
    Zout = np.zeros((len(k_joint), len(y)), dtype=complex)
    #print np.shape(Z), np.shape(Zout)
    Zout[len(ks):, :] = Z[-len(k_data):, :]
    for jy, yj in enumerate(y):
        #print "prepend for y = ", yj
        Zout[0:len(ks), jy] = do_fit(k, Z[:, jy], yj, fit_min, fit_max,
                                            ks, fit_func_re, fit_func_im)
    if do_flatten:
        Zout = Zout[:, 0]
    return k_joint, Zout
    
def fit_append(k, y, Z, fit_min, fit_max, data_max, ymax, ks,
               fit_func_re, fit_func_im, func_scale, h = 0, bounds=None):
    #print Z.ndim
    do_flatten = False
    if Z.ndim == 1:
        Z = np.array([Z]).transpose()
        do_flatten = True
    i_data = [t for t in range(len(k)) if k[t] <= data_max]
    k_data = np.array([k[t] for t in i_data])
    k_joint = list(k_data)
    k_joint.extend(ks)
    k_joint = np.array(k_joint)
    #print "k_joint, y", np.shape(k_joint), np.shape(y)
    Zout = np.zeros((len(k_joint), len(y)), dtype=complex)
       
    #print np.shape(Z), np.shape(Zout)
    Zout[:len(k_data), :] = Z[0:len(k_data), :]
    #i_start = np.argmin(np.abs(k - ks[0]))
    #if k[i_start] < ks[0]:
    #   i_start += 1
    for jy, yj in enumerate(y):
         #if np.abs(yj) > ymax: continue
         #print "append for y = ", yj
         #scale_data = func_scale(k_data, yj)
         #scale = func_scale(k, yj)
         #if np.min(scale_data) < 1e-10: 
         #   continue
         #try:
         #   Zout[-len(ks):, jy] = do_fit(k, Z[:, jy]/scale, yj, fit_min, fit_max,
         #                                   ks, fit_func_re, fit_func_im) * func_scale(ks, yj)
         #except: 
         #   import traceback
         #   traceback.print_exc()
         #   continue
         #if (yj > 0) and np.abs(k[-1]*yj) < 50:
        #print yj
        Zmax = np.max(np.abs(Z[:, jy]))
        Zlast = np.abs(Z[-1, jy])
        if Zlast / Zmax > 1e-5:
        #if min(np.abs(yj), np.abs(yj - h))* np.abs(k_data[-1]) < 10:
            scale_Z_in = 0.0 * k + 1.0
            scale_Z_out = 0.0 * ks + 1.0
            bounds[1][2] = 5 * min(np.abs(yj), np.abs(yj - h)) + 1.0
            #if (abs(yj) < abs(yj - h)):
            #     scale_Z_in = np.exp(np.abs(k)*yj)
            #     scale_Z_out = np.exp(-np.abs(ks)*yj)
            #else:
            #     scale_Z_in = np.exp(np.abs(k)*np.abs(yj - h))
            #     scale_Z_out = np.exp(-(np.abs(ks))*abs(yj - h))
            Zscaled = Z[:, jy] * scale_Z_in
            Zout_scaled = do_fit(k, Zscaled, yj, fit_min, fit_max,
                                 ks, fit_func_re, fit_func_im, False, bounds)
            Zout[-len(ks):, jy] = Zout_scaled * scale_Z_out
            #* np.exp(-np.abs(ks)*np.abs(yj))
            print("extend the data by fitting, ", yj, h, k_data[-1])
        else:
            Zout[-len(ks):, jy] = Z[len(k_data) - 1, jy]  \
                        *np.exp(-(np.abs(ks) - np.abs(k_data[-1]))*np.abs(yj))
    if do_flatten:
        Zout = Zout[:, 0]
    return k_joint, Zout
