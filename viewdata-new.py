import numpy as np
import pylab as pl
import sys
#from whsolve4 import Fourier
from scipy import optimize
from scipy import special
import _makequad as mq
from scipy import integrate, linalg
import matplotlib.patches as patches

#src_type = 'sin'
src_type = 'isotropic'
#selector = "sin"
#selector = "cos"
#selector = ""
#selector = ""

gamma = 1.0

#pl.rc('text', usetex=True)
pl.rc('font', size=14)
#pl.rc('font', family='serif')
pl.rc("savefig", directory="")

def visc_phi(data, x = None, scale = 1):
    if x == None: x = data.x
    f_visc = 0.0 * x + 0.0
    if data.sel == 'iso':
        C_visc = 1.0 / np.pi / (data.gamma / scale)
        x_s = x * scale
        h_s = data.h * scale
        f_visc =  C_visc * (h_s**2 - x_s**2)  / (x_s**2 + h_s**2)**2
    if data.sel == 'cos':
        C_visc = 4.0/np.pi
        x_s = x * scale
        h_s = data.h * scale
        xh2 = x_s**2 + h_s**2
        f_visc = C_visc * x_s * h_s**2/ xh2**2
    if data.sel == 'sin':
        C_visc = -4.0/np.pi
        x_s = x * scale
        h_s = data.h * scale
        xh2 = x_s**2 + h_s**2
        f_visc = C_visc * h_s**3 / xh2**2
    clip_max = np.max(np.abs(data.f_s)) * 1.2
    f_visc_clipped = f_visc.copy()
    f_visc_clipped[f_visc >  clip_max] = np.nan
    f_visc_clipped[f_visc < -clip_max] = -np.nan
    return f_visc_clipped

def save_fs(data, fname, sel):
    f = open("flux-s.dat", "w")
    f.write("# x\tPhi(x)")
    for x_i, y_i in zip(data.x, data.f_s):
        f.write("%g\t%g\n" % (x_i, y_i))
    f.close()


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
    

class FlowData:
    def __init__(self):
        self.x = None
        self.y = None
        self.i_b = None
        self.rho = None
        self.drho = None
        self.psi = None
        self.psi_alt = None
        self.h = None
        self.f_s = None
        self.rho_b = None
        self.jx = None
        self.jy = None
        self.R_dir = None
        self.jr = None
        self.jtheta = None
        self.r_src = None
        self.j_src = None
        self.gamma = None
        self.gamma1 = None
        self.gamma2 = None

def prepare_data(fname, sel, x, R_dir, **kwargs):
    params = {
        'L_left'  : -100.0,
        'L_right' : 100.0,
        'A_left'  : 0.5, 
        'A_right' : 0.5
    }
    params.update(kwargs)
    
    d = np.load(fname)
    print(list(d.keys()))
    print(sel)
    #if not 'rho_cos' in list(d.keys()):
    #    sel = ''
    try:
        h = d['h']
    except:
        import traceback
        traceback.print_exc()
        h = 0.0
    data = FlowData()
    data.h = h
    data.gamma = 1.0
    #if 'gamma' in d.keys():
    data.gamma  = d['gamma']
    data.gamma1 = d['gamma1']
    data.gamma2 = data.gamma - data.gamma1
    data.sel = sel
    y = d['y']
    k = d['k']
    data.y = y
    data.x = x
    #k = np.linspace(0.01, 30, 3000)
    C_sin = 1.0
    C_cos = 1.0
    C_rho = 1.0
    if sel == '' or sel == 'iso':
      drho_k = d['I:drho'] * C_rho
      try: 
        rho_k = d['I:rho'] * C_rho
      except:
        rho_k = 0.0 * drho_k
      jx_k = d['I:jx'] * C_rho
      jy_k = d['I:jy'] * C_rho
      f_sk = d['I:f'] * C_rho
      phi_s = d['I:flux_diff'] * C_rho
      phi_j = d['I:flux_bare'] * C_rho
      C_rho0 = 1.0 * C_rho
      C_phi_j = 0.0 * C_rho
    if sel == 'cos':
      rho_k = drho_k = d['Fx:rho'] * C_cos
      jx_k = d['Fx:jx'] * C_cos
      jy_k = d['Fx:jy'] * C_cos
      f_sk = d['Fx:f']  * C_cos 
      phi_s = d['Fx:flux_diff'] * C_cos
      phi_j = d['Fx:flux_bare'] * C_cos
      C_rho0 = 0.0 * C_cos
      C_phi_j = 0.0 * C_cos
    if sel == 'sin':
      rho_k = drho_k = d['Fy:rho'] * C_sin
      jx_k = d['Fy:jx'] * C_sin
      jy_k = d['Fy:jy'] * C_sin
      f_sk = d['Fy:f']  * C_sin 
      phi_s = d['Fy:flux_diff'] * C_sin 
      phi_j = d['Fy:flux_bare'] * C_sin
      C_rho0 = 0.0 * C_sin
      C_phi_j = 0.0 * C_sin
    #bc_drho_j = d[drho_]
    #bc_drho_j = d['bc_drho_j']
    #bc_rho_s = d['bc_rho_s']
    
    #drho_bk = d['bc_drho_j'] + d['bc_rho_s'] * f_sk
    i_b = np.argmin(np.abs(y))
    if y[i_b] < 0:
        i_b += 1
    drho_bk = drho_k[:, i_b]
    data.i_b = i_b

    fit_small_min = 0.003#0.05 #0.08 # 0.06
    fit_small_max = 0.03#0.15  ##0.17 # 0.0
    fit_large_min = 10.0
    fit_large_max = 25.0
    ymax = 0.1
    data_min = 0.01
    data_max = 25.00
    k_max = 1000.0
    k_small = np.linspace(0.0001, data_min*(1.0 - 3e-3), 500)
    #k_large = np.linspace(data_max + 0.01, k_max, 501)
    k_large_min = data_max + 0.05
    k_large = k_large_min * np.exp(np.linspace(0.0, np.log(k_max/k_large_min), 1001))

    fit_sample = np.linspace(-0.1, 0.3, 401)
    fit_large_sample = np.linspace(10.0, 30.0, 1000)

    i_last = len(phi_s) - 1
    for i in range(len(phi_s) - 1, 0, -1):
        print(phi_s[i])
        if np.abs(phi_s[i]) < 1e-10:
           i_last = i
        else:
           break
    print("last data:", i_last, k[i_last])
    if k[i_last] < fit_large_max and k[i_last] > fit_large_min:
        fit_large_max = k[i_last]
    print("limit fits to ", fit_large_min, fit_large_max)

    fit_large_bounds = (np.array([-10, -10, 0.0, -10]),
                        np.array([10,   10, 2.0, 10]))
    k_joint, drho_k_joint = fit_prepend(k, y, drho_k,
                                        fit_small_min, fit_small_max,
                                        data_min, k_small,
                                        fit_lin, fit_lin_zero)
    k_joint2, drho_k_joint2 = fit_append(k_joint, y, drho_k_joint,
                                         fit_large_min, fit_large_max, 
                                         data_max, ymax, k_large,
                                         fit_inv_exp, fit_inv_exp, 
                                         scale_exp, h, fit_large_bounds)
    k_joint, rho_k_joint = fit_prepend(k, y, rho_k,
                                       fit_small_min, fit_small_max,
                                  data_min, k_small, fit_lin, fit_lin_zero)
    k_joint2, rho_k_joint2 = fit_append(k_joint, y, rho_k_joint,
                                         fit_large_min, fit_large_max, 
                                         data_max, ymax, k_large,
                                         fit_inv_exp, fit_inv_exp, 
                                         scale_exp, h, fit_large_bounds)
    #del drho_k
    del drho_k_joint
    dummy, drho_bk_joint = fit_prepend(k, np.array([0.0]),
                                       drho_bk,
                                       fit_small_min,
                                       fit_small_max, data_min,
                                       k_small, fit_lin, fit_lin_zero)
    dummy, drho_bk_joint2 = fit_append(k_joint, np.array([0.0]),
                                       drho_bk_joint,
                                       fit_large_min, fit_large_max,
                                       data_max, ymax, k_large,
                                       fit_inv_exp, fit_inv_exp,
                                       scale_exp, h, fit_large_bounds)
    dummy, f_sk_joint = fit_prepend(k, np.array([0.0]), f_sk,
                                       fit_small_min,
                                       fit_small_max, data_min,
                                       k_small, fit_lin, fit_lin_zero)
    dummy, f_sk_joint2 = fit_append(k_joint, np.array([0.0]), f_sk_joint,
                                       fit_large_min, fit_large_max,
                                       data_max, ymax, k_large,
                                       fit_inv_exp, fit_inv_exp,
                                       scale_exp, h, fit_large_bounds)
    if False:
        import pylab as pl
        pl.figure()
        pl.plot(k_joint2, f_sk_joint2.real, label='Re f_s(k)')
        pl.plot(k_joint2, f_sk_joint2.imag, label='Im f_s(k)')
        pl.legend()
        pl.show()
    k_gamma = np.sqrt(1.0 + k_joint2**2)
    r_src = 0.003;
    def src_func(kv):
      return np.exp(-0.5 * (kv*r_src)**2)
    data.r_src = r_src
    src = src_func(k)
    src_joint2 = src_func(k_joint2)

    dummy, jy_k_joint = fit_prepend(k, y, jy_k,
                                     fit_small_min, fit_small_max, 
                                  data_min, k_small, fit_lin, fit_lin)
    dummy, jy_k_joint2 = fit_append(k_joint, y, jy_k_joint,
                                       fit_large_min, fit_large_max,
                                       data_max, ymax, k_large,
                                       fit_inv_exp, fit_inv_exp,
                                       scale_exp, h, fit_large_bounds)
    del jy_k
    del jy_k_joint
    dummy, jx_k_joint = fit_prepend(k, y, jx_k,
                                     fit_small_min, fit_small_max, 
                                  data_min, k_small, fit_lin, fit_lin)
    dummy, jx_k_joint2 = fit_append(k_joint, y, jx_k_joint,
                                       fit_large_min, fit_large_max,
                                       data_max, ymax, k_large,
                                       fit_inv_exp, fit_inv_exp,
                                       scale_exp, h, fit_large_bounds)
    del jx_k
    del jx_k_joint
    psi_k_joint2 = np.transpose(1.0 / (-1j*k_joint2) * np.transpose(jy_k_joint2))

    Y, X = np.meshgrid(y, x)
    fourier_joint2 = Fourier(k_joint2, x)
    #drho_b = np.dot(fourier_joint2 * src_joint2, drho_bk_joint2).real * 2.0

    #L0 = 100.0;
    L_left  = params['L_left']
    L_right = params['L_right']
    A_left  = params['A_left']
    A_right = params['A_right']
    print(params)
    cos_src = (1.0 - A_left * np.exp(-1j*k_joint2 * L_left)
                   - A_right*np.exp(-1j*k_joint2*L_right))
    data.j_src = np.dot(fourier_joint2 * src_joint2*cos_src, np.ones(len(src_joint2))).real * 2.0
    Rh = np.sqrt((Y - h + 1e-4)**2 + X**2)
    if h > 1e-3: 
       RHO_0 = C_rho0 * 1.0/2.0/np.pi/Rh * np.exp( - gamma * Rh )
    else:
       RHO_0 = C_rho0 * 1.0/np.pi/Rh * np.exp( - gamma * Rh )       
    iy0 = np.argmin(np.abs(y))
    RHO_0[:, 0:iy0] = 0.0
    DRHO_joint2 = np.dot(fourier_joint2 * src_joint2, drho_k_joint2).real * 2.0
    PSI_joint2 = np.dot(fourier_joint2 * src_joint2 * cos_src,
                        psi_k_joint2).real * 2.0
    JX = np.dot(fourier_joint2 * src_joint2, jx_k_joint2).real * 2.0
    print ("JX = ", JX)
    JY = np.dot(fourier_joint2 * src_joint2, jy_k_joint2).real * 2.0
    #j_zero = np.argmin(np.abs(data.y))
    JX[:, :iy0] = 0.0
    JY[:, :iy0] = 0.0

    data.psi = np.nan_to_num(PSI_joint2, nan=0.0)
    data.jx = np.nan_to_num(JX)
    data.jy = np.nan_to_num(JY)
    data.rho = np.nan_to_num(RHO_0 + DRHO_joint2)
    data.drho = np.nan_to_num(DRHO_joint2)
    
    PSI3 = 0.0 * X
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    for j in range(1, len(y)):
        print ("PIS3 col: ", PSI3[:, j-1])
        jx_half = -(JX[:, j] + JX[:, j - 1])/2.0
        print ("jx_half: ", jx_half)
        PSI3[:, j] = PSI3[:, j - 1] + jx_half * dy
        print ("PIS3 col: ", PSI3[:, j])
    print ("PSI3: ", np.max(PSI3), np.min(PSI3))
    j_zero = np.argmin(np.abs(data.y))
    data.psi_alt = np.nan_to_num(PSI3, nan=0.0)
    data.psi_alt[:, :j_zero] = 0
    data.psi[:, :j_zero] = 0
    data.rho[:, :j_zero] = 0
    data.drho[:, :j_zero] = 0

    f_s_joint2 = np.dot(fourier_joint2 * src_joint2, f_sk_joint2).real * 2.0

    data.f_s = f_s_joint2

    #return data

    if True:
      data.jr = []
      data.jtheta = []
      data.R_dir = []
      #theta = np.linspace(0.0, np.pi, 501)
      #pl.figure()
      for R in R_dir:
        i_R = [t for t in range(len(y)) if y[t] > 0 and abs(y[t] - h) < R]
        y_R =  np.array([y[t] for t in i_R])
        jxk_R = np.array([jx_k_joint2[:, t] for t in i_R])
        jyk_R = np.array([jy_k_joint2[:, t] for t in i_R])
        x_R1 = np.sqrt(R**2 - (y_R - h)**2)
        x_R2 = - x_R1
        theta_R1 = np.angle(-1j*(x_R1 + 1j * (y_R - h))) + np.pi/2.0
        fourier_R1 = Fourier(k_joint2, x_R1)
        theta_R2 = np.angle(-1j*(x_R2 + 1j * (y_R - h))) + np.pi/2.0
        fourier_R2 = Fourier(k_joint2, x_R2)
        #print "fourier:", np.shape(fourier_R), "jx:", np.shape(jxk_R)
        jx_R1 = np.diag(np.dot(fourier_R1 * src_joint2,
                              np.transpose(jxk_R))).real * 2
        jy_R1 = np.diag(np.dot(fourier_R1 * src_joint2,
                              np.transpose(jyk_R))).real * 2 
        jx_R2 = np.diag(np.dot(fourier_R2 * src_joint2,
                              np.transpose(jxk_R))).real * 2
        jy_R2 = np.diag(np.dot(fourier_R2 * src_joint2,
                              np.transpose(jyk_R))).real * 2 
        j_r1     = jx_R1 * np.cos(theta_R1) + jy_R1 * np.sin(theta_R1)
        j_theta1 = - jx_R1 * np.sin(theta_R1) + jy_R1 * np.cos(theta_R1)
        j_r2     = jx_R2 * np.cos(theta_R2) + jy_R2 * np.sin(theta_R2)
        j_theta2 = - jx_R2 * np.sin(theta_R2) + jy_R2 * np.cos(theta_R2)
        theta_rev = list(theta_R2)
        theta_rev.reverse()
        jr_rev = list(j_r2)
        jr_rev.reverse()
        jtheta_rev = list(j_theta2)
        jtheta_rev.reverse()
        theta_R = list(theta_R1)
        theta_R.extend(theta_rev)
        theta_R = np.array(theta_R)
        j_r = list(j_r1)
        j_r.extend(jr_rev)
        j_r = np.array(j_r)
        j_theta = list(j_theta1)
        j_theta.extend(jtheta_rev)
        j_theta = np.array(j_theta)
        #print theta_R
        #print j_r
        #print j_theta
        #pl.figure()
        #pl.title("Directivity at R = %g" % R)
        #pl.plot(theta_R, j_r, label='j_r')
        #pl.plot(np.pi - theta_R, j_r, label='j_r')
        #pl.plot(theta_R, j_theta, label='j_theta')
        #pl.plot(np.pi - theta_R, -j_theta, label='j_theta')
        #pl.polar(theta_R, j_r / j_r.max(), label=r'$|{\bf r} - {\bf r}_s| = %g l_{ee}$' % R)
        #pl.plot(theta_R, j_r / j_r.max(), label=r'$|{\bf r} - {\bf r}_s| = %g l_{ee}$' % R)
        #pl.plot(theta_R1, j_r1 / j_r.max(), label=r'1:$|{\bf r} - {\bf r}_s| = %g l_{ee}$' % R)
        #pl.plot(theta_R2, j_r2 / j_r.max(), label=r'2:$|{\bf r} - {\bf r}_s| = %g l_{ee}$' % R)
        total_current = integrate.trapz(j_r * R, theta_R)
        print("total current @ R = ", R, " I = ", total_current)
        data.R_dir.append(R)
        data.jr.append((theta_R, j_r))
        data.jtheta.append((theta_R, j_r))
        
      #pl.legend()
      #pl.title("Flow directivity, $h = %g l_{ee}$" % h)
      #pl.show()
    
    return data;     
    #pl.show()

pl.figure()
ax_fs = pl.gca()
pl.title("Edge probe potential")
pl.xlabel(r"Probe position $x/l_{ee}$")
pl.ylabel(r"$V_P(x)$")
def view_fs(data, fname, sel):
    ax_fs.plot(data.x, data.f_s, label=r'$h = %gl_{ee}$' % data.h )
    ax_fs.plot(data.x, visc_phi(data), 'g--', label=r'Viscous')

pl.figure()
ax_fs_sc = pl.gca()
pl.title("Edge probe potential")
pl.xlabel(r"Probe position $x/h$")
pl.ylabel(r"$V_P(x)$")
ax_fs_sc_zoom = pl.axes([0.4, 0.3, 0.5, 0.3])
def view_fs_scaled(data, fname, sel):
    if abs(data.h) < 1e-3:
        l_ee_str = "\infty"
    else:
        l_ee_str = "%2gh" % (1.0/data.h)
    ax_fs_sc.plot(data.x / data.h, data.f_s * data.h, label=r'$l_{ee} = %s$' % l_ee_str)
    ax_fs_sc_zoom.plot(data.x/data.h, data.f_s * data.h)


pl.figure()
ax_fs_pp = pl.gca()
pl.ylabel(r"Probe potential $V_P$, a.u.")
pl.xlabel(r"Relaxation rate $h / l_\mathrm{ee}$")
x_pp = [0.0, 0.5, 1.0, 2.0, 5.0]
data_pp = {}
for x in x_pp:
    data_pp[x] = []
def view_fs_pp(data, fname, sel):
    for x_p in x_pp:
        i_p = np.argmin(np.abs(data.x / data.h - x_p))
        data_pp[x_p].append((data.h, data.x[i_p], data.f_s[i_p]))
    ax_fs_pp.clear()
    for x_p in x_pp:
        d = data_pp[x_p]
        h_d = np.array([t[0] for t in d])
        x_d = np.array([t[1] for t in d])
        f_d = np.array([t[2] for t in d])
        x_s = x_d / h_d
        f_s = f_d * h_d
        gamma_s = 1.0 * h_d
        ax_fs_pp.plot(gamma_s, f_s, '-o', label='$x_P =  %g h$' % x_p)
    ax_fs_pp.legend()
    ax_fs_pp.set_ylabel(r"Probe potential $V_P$, a.u.")
    ax_fs_pp.set_xlabel(r"Relaxation rate $h / l_\mathrm{ee}$")

    
pl.figure()
pl.title("Edge probe potential")
pl.xlabel(r"Relaxation rate $\gamma x_P$")
pl.ylabel(r"$V_P(x_P)$, a.u.")
ax_fs_gamma = pl.gca()
def view_fs_gamma(data, fname, sel):
    for x_0 in [1.0]:
       i_x = [t for t in range(len(data.x)) if data.x[t] > 0.01]
       x_p = np.array([data.x[t] for t in i_x])
       f_p = np.array([data.f_s[t] for t in i_x]).flatten()
       print ("view_fs_gamma; f_p = ", f_p, np.shape(f_p))
       gamma_p = np.abs(x_p) / x_0
       print ("view_fs_gamma; gamma_p = ", gamma_p, np.shape(gamma_p))
       #ax_fs_gamma.plot(gamma_p, gamma_p * f_p, label='x_0 = %g' % x_0)
       ax_fs_gamma.plot(gamma_p, gamma_p * f_p, 'k', label='WH solution')
       i_min = 0
       for i in range(1, len(gamma_p)):
           if (gamma_p[i] * f_p[i] > gamma_p[i - 1] * f_p[i - 1]):
               i_min = i
               break
       ax_fs_gamma.plot(gamma_p[i_min], gamma_p[i_min] * f_p[i_min], 'ro',
                        ms=5.0, fillstyle='none')
       gamma_small = np.linspace(0.001, 0.35, 101)
       gamma_large = np.linspace(5.0, 20.0, 101)
       ax_fs_gamma.plot(gamma_small,
                     -gamma_small/np.pi*np.log(1.0/gamma_small/x_0),
                     'b--', label='Backscattering')
       f_visc = visc_phi(data, x_0, gamma_large)
       #f_visc = 0.0 * gamma_large
       #if data.sel == 'iso':
       #   f_visc = -1.0/gamma_large/np.pi / (x_0**2 + data.h**2)
       ax_fs_gamma.plot(gamma_large, f_visc,
                     'g--', label='Fluid mechanics')
       ax_fs_gamma.annotate(r'min: $\gamma x_P = %.2g$' % (gamma_p[i_min]*x_0),
             xy=(gamma_p[i_min]*1.05, gamma_p[i_min] * f_p[i_min]),
             xytext=(gamma_p[i_min] + 1.0, 0.99*gamma_p[i_min] * f_p[i_min]),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>', edgecolor='red',
                             facecolor='red'),
             horizontalalignment='left',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='red'))
       i_0 = np.argmin(np.abs(gamma_p - 0.05))       
       ax_fs_gamma.annotate(r'Ballistic',
             xy=(gamma_p[i_0]*1.07, gamma_p[i_0] * f_p[i_0]),
             xytext=(gamma_p[i_0]+0.8, gamma_p[i_0] * f_p[i_0]),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>', edgecolor='blue',
                             facecolor='blue'),
             horizontalalignment='left',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='white'))
       i_1 = np.argmin(np.abs(gamma_p - 7.5))
       ax_fs_gamma.annotate(r'Viscous',
             xy=(gamma_p[i_1]*1, gamma_p[i_1] * f_p[i_1]),
             xytext=(gamma_p[i_1], gamma_p[i_1] * f_p[i_1] + 0.02),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>', edgecolor='green',
                             facecolor='green'),
             horizontalalignment='center',
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='white'))
       i_2 = np.argmin(np.abs(gamma_p - 3.5))
       ax_fs_gamma.annotate(r'Exact solution',
             xy=(gamma_p[i_2]*1, gamma_p[i_2] * f_p[i_2]),
             xytext=(gamma_p[i_2] + 0.8, gamma_p[i_2] * f_p[i_2]),
             textcoords='data',
             arrowprops=dict(arrowstyle='-|>', edgecolor='black',
                             facecolor='black'),
             horizontalalignment='left',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='white'))
    #ax_fs_gamma.legend(loc=2)

pl.figure()
ax_fs_and_rho = pl.gca()
pl.xlabel(r"Probe position $x/l_{ee}$")
pl.ylabel(r"Potential, a.u.")
ax_fs_and_rho_zoom = pl.axes([0.45, 0.45, 0.35, 0.3])
#pl.title("Flux and density at the edge")
def view_fs_and_rho(data, fname, sel):
    ax_fs_and_rho.plot(data.x, data.f_s,
                       label=r'Edge probe potential $V_P(x)\propto f_s(x)$')
    ax_fs_and_rho.plot(data.x, data.rho[:, data.i_b],
                       label=r'Electric potential $\phi(x) \propto \rho(x)$')
    ax_fs_and_rho.plot(data.x, visc_phi(data), '--', label='Viscous result')
    
    ax_fs_and_rho_zoom.plot(data.x, data.f_s)
    ax_fs_and_rho_zoom.plot(data.x, data.rho[:, data.i_b])
    
pl.figure()
ax_rho_b = pl.gca()
pl.title("Electric potential")
pl.xlabel(r"Probe position $x/l_{ee}$")
pl.ylabel(r"$\phi(x)$")
def view_rho_b(data, fname, sel):
    ax_rho_b.plot(data.x, data.rho[:, data.i_b],
                  label=r'$h = %gl_{ee}$' % data.h )
    ax_rho_b.plot(data.x, visc_phi(data), 'k--',
                  label=r'Viscous')

def draw_edge(data, ax=pl.gca()):
    if data.h < 1e-6: 
      rect1 = patches.Rectangle((data.x.min(), data.y.min()),
                             (-data.r_src - data.x.min()),
                             (0.0 - data.y.min()) ,
                             edgecolor='0.5',
                              facecolor='0.8')
      rect2 = patches.Rectangle((data.r_src, data.y.min()),
                             (data.x.max() - data.r_src),
                             (0.0 - data.y.min()) ,
                             edgecolor='0.5',
                             facecolor='0.8')
      rect3 = patches.Rectangle((-data.r_src, data.y.min()),
                              (2*data.r_src), (0.0 - data.y.min()),
                              edgecolor='red', facecolor='red')
      ax.add_patch(rect1)
      ax.add_patch(rect2)
      ax.add_patch(rect3)
    else:
       rect = patches.Rectangle((data.x.min(), data.y.min()),
                             (data.x.max() - data.x.min()),
                             (0.0 - data.y.min()) ,
                             edgecolor='0.5',
                             facecolor='0.8')
       ax.add_patch(rect)
       
def view_psi(data, fname, sel):
    Y, X = np.meshgrid(data.y, data.x)
    pl.figure()
    #V = np.linspace(-0.495, 0.495, 32)
    print("psi: ", data.psi.min(), data.psi.max())
    #pl.contour(X, Y, data.psi, V, cmap='hsv')
    pl.contour(X, Y, data.psi, 32, cmap='hsv')
    draw_edge(data, pl.gca())
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.xlabel("$x/l_{ee}$")
    pl.ylabel("$y/l_{ee}$")
    pl.title(r"Stream function $\psi(x,y)$, $h = %g l_{ee}$" % data.h)
    #pl.show()
    
def view_psi_alt(data, fname, sel):
    Y, X = np.meshgrid(data.y, data.x)
    pl.figure()
    pl.contour(X, Y, data.psi_alt, 30, cmap='hsv')
    draw_edge(data, pl.gca())
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.xlabel("$x/l_{ee}$")
    pl.ylabel("$y/l_{ee}$")
    pl.title(r"Stream function $\psi(x,y)$, $h = %g l_{ee}$" % data.h)
    #pl.show()

    
def view_rho_comb(data, fname, sel, maxv = 0.05):
    pl.figure()
    import matplotlib.colors as mpc
    from matplotlib.collections import LineCollection
    class Custom_Norm(mpc.Normalize):
      def __init__(self, vmin, vzero, vmax):
        self.vmin = vmin
        self.vmax = vmax
        self.vzero = vzero
      def __call__ (self, value, clip=None):
        x, y = [self.vmin, self.vzero, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

    print("rho: min = ", data.rho.min(), "max = ", data.rho.max())
    #maxv = 0.2
    custom_norm = Custom_Norm(-maxv, 0.0, maxv)

    Y, X = np.meshgrid(data.y, data.x)
    if data.h > 0.001 and data.sel == 'iso' or data.sel=='sin':
        psi_XY = data.psi_alt.copy()
    #   cs= pl.contour(X, Y, data.psi_alt, 31, colors='black',
    #           lw=0.5, linestyles='solid')
    else:
      psi_XY = data.psi
    min_lev = np.min(data.psi)
    max_lev = np.max(data.psi)
    levs = np.linspace(min_lev, max_lev, 25)
    levs = 0.5 * (levs[1:] + levs[:-1])
    print ("levels: ", list(levs))
    print ("levels = ", levs)
    cs = pl.contour(X, Y, psi_XY, levs,
                    colors='green', lw=0.5, linestyle='solid')
    pl.clf()
    pl.pcolor(X, Y, data.rho, cmap='bwr', norm=custom_norm, shading='auto')
    draw_edge(data, pl.gca())
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.xlabel("$x/l_{ee}$")
    pl.ylabel("$y/l_{ee}$")
    cb.set_label(r"$\phi(x, y)$")
    #import matplotlib._cntr as cntr
    #c = cntr.Cntr(X, Y, psi_XY, 31)
    segments = cs.allsegs
    levels = cs.levels
    arrows = []
    for i in range(0, len(segments)):
        #if np.abs(levels[i]) < 0.001: continue
        polygons = segments[i]
        for poly in polygons:
           #if i == 0 or i == len(segments) - 1:
           #poly = poly[:int(len(poly)/2)]
           #print ("lev = ", levs[i], "poly:",  poly[0:2], "...", poly[-2:])
           def cut(t):
               if data.sel != 'iso' and data.sel != '': return False
               x_t = t[0]
               y_t = t[1]
               if np.abs(data.h) < 0.001:
                   if np.abs(y_t) < 0.1: return True
                   return False
               print ("cut?", x_t, y_t)
               if abs(x_t - 0.0) < max(0.01, 0.1*np.abs(y_t - data_h))  \
                  and y_t > data.h: return True
               return False
           x_seg = [t[0] for t in poly if not cut(t)]
           y_seg = [t[1] for t in poly if not cut(t)]
           if data.sel == 'iso' and data.h < 0.001:
               pass
               x_seg.insert(0, 0.0)
               y_seg.insert(0, data.h)
           x_seg = np.array(x_seg)
           y_seg = np.array(y_seg)
           if len(x_seg) < 10: continue
           if np.abs(levs[i]) < 0.05:
               print ("x = ", x_seg, "y = ", y_seg)
           print ("filtered:", len(x_seg), len(poly))
           z_seg = x_seg + 1j * y_seg
           R_0 = np.max(Y)*0.9 - data.h 
           z0 = 0.0 + 1j * data.h
           sgn = 1.0
           if i % 2 : sgn = -1.0
           i_arr = np.argmin(np.abs(np.abs(z_seg - z0) - R_0)
                             - 0.001 *  sgn * x_seg)
           i_end = i_arr - 5
           if i_arr >= len(x_seg): continue
           #i_arr = int(1*len(x_seg)/3)
           #pl.plot(x_seg, y_seg, 'k-')
           #new_segs = np.array([[[x_seg[t], y_seg[t]]
           #                      for t in range(len(x_seg))]])
           lw = []
           for i in range(len(x_seg) - 1):
                new_segs.append([[[x_seg[i], y_seg[i]],
                                  [x_seg[i + 1], y_seg[i + 1]]]])
                xm = (x_seg[i] + x_seg[i + 1])/2
                ym = (y_seg[i] + y_seg[i + 1])/2
                ix = np.argmin(np.abs(X[:, 0] - xm))
                iy = np.argmin(np.abs(Y[0, :] - ym))
                jx = data.jx[ix, iy]
                jy = data.jy[ix, iy]
                J   = np.sqrt(jx**2 + jy**2) 
                J0  = 1.0/2.0/np.pi * 0.3
                lw0 = 1.5
                lwid = 1.5 * np.abs(J)/J0 
                #lwid = 1.5 + np.log((np.abs(J) + 1e-4)/J0) * lw0
                if lwid < 0.5: lwid = 0.5
                if lwid > 5.0: lwid = 5.0
                lw.append(lwid)
           #if data.sel == 'iso':
           #    x_seg.append(0.0)
           #    y_seg.append(data.h)
           new_segs = np.array(new_segs)
           #print ("lw = ", lw)
           lw = np.array(lw)
           lc = LineCollection(new_segs, linewidths=np.array(lw),
                               color='black')
           pl.gca().add_collection(lc)
           
           x_arr = float(x_seg[i_arr])
           y_arr = float(y_seg[i_arr])
           #i_end = i_arr - 5
           dx_arr = x_seg[i_end] - x_arr
           dy_arr = y_seg[i_end] - y_arr
           print ("arr", x_arr, y_arr, dx_arr, dy_arr)
           arr = pl.gca().arrow(x_arr, y_arr, dx_arr, dy_arr, color='black',
                                shape='full', overhang=0.25, width=0.13)
           arrows.append(arr)
    #nlist = c.trace()
    #print ("cs.colls = ", cs.collections)
    #paths = cs.collection[1].get_paths()
    #for path in paths:
    #    print ("path = ", path)
    # cs = pl.contour(X, Y, data.psi, 31, colors='black',
    #         lw=0.5, linestyles='solid')
    #for i in range(len(cs.collections)):
    #    path = cs.collections[i].get_paths()[0].vertices
    #    x = path[:, 0]
    #    y = path[:, 1]
    #    plt.plot(x, y, 'g--')
    pl.title(r"Electric potential  $\phi(x, y)$, $h = %gl_{ee}$" % data.h)
    #pl.show()
    
def view_rho(data, fname, sel, maxv = 0.05):
    pl.figure()
    Y, X = np.meshgrid(data.y, data.x)
    pl.pcolormesh(X, Y, data.rho, cmap='bwr', vmin=-maxv, vmax=maxv,
              shading='auto')
    draw_edge(data, pl.gca())
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.xlabel("$x/l_{ee}$")
    pl.ylabel("$y/l_{ee}$")
    cb.set_label(r"$\phi(x, y)$")
    pl.title(r"Electric potential  $\phi(x, y)$, $h = %gl_{ee}$" % data.h)
    #pl.show()
    
def view_rho_comb2(data, fname, sel, maxv = 0.05, minv=0.0):
    pl.figure()
    from matplotlib.colors import LogNorm
    Y, X = np.meshgrid(data.y, data.x)
    pl.pcolormesh(X, Y, data.jx**2 + data.jy**2,
              norm=LogNorm(vmin=0.01, vmax=10.0))
    pl.colorbar()
    pl.title("$j^2$")
    
    fig = pl.figure(figsize=(10, 6))
    import matplotlib.colors as mpc
    from matplotlib.collections import LineCollection
    class Custom_Norm(mpc.Normalize):
      def __init__(self, vmin, vzero, vmax):
        self.vmin = vmin
        self.vmax = vmax
        self.vzero = vzero
      def __call__ (self, value, clip=None):
        x, y = [self.vmin, self.vzero, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

    print("rho: min = ", data.rho.min(), "max = ", data.rho.max())
    #maxv = 0.2
    if minv > -1e-6: minv = -maxv
    custom_norm = Custom_Norm(minv, 0.0, maxv)

    Y, X = np.meshgrid(data.y, data.x)
    if data.h > 0.001 and data.sel == 'iso' or data.sel=='sin':
        psi_XY = data.psi_alt
    #   cs= pl.contour(X, Y, data.psi_alt, 31, colors='black',
    #           lw=0.5, linestyles='solid')
    else:
        psi_XY = data.psi
    min_lev = np.min(data.psi)
    max_lev = np.max(data.psi)
    levs = np.linspace(min_lev, max_lev, 25)
    if data.h < 0.001:
       levs = np.linspace(min_lev, max_lev, 25)
       levs = 0.5 * (levs[1:] + levs[:-1])
    else:
       #levs = np.linspace(min_lev, max_lev, 26)[:-1]
       #levs = np.linspace(min_lev, max_lev, 26)
       levs = 0.5 * (levs[1:] + levs[:-1])
    print ("levels: ", list(levs))
    print ("levels = ", levs)
    cs = pl.contour(X, Y, psi_XY, levs,
                    colors='green', lw=0.5, linestyle='solid')
    #pl.show()
    pl.clf()
    y_max = np.max(Y)
    if data.h < 10: 
       y_max = min(10.0, np.max(Y))
    from matplotlib import gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #gs = gridspec.GridSpec(2, 2)
    ax_phi = fig.add_axes([0.05, 0.1, 0.85, 0.8])
    #pl.subplot(gs[0])
    divider = make_axes_locatable(ax_phi)
    ax_flux = divider.append_axes("bottom",
                                  size="66%",pad="1%", sharex=ax_phi)
    ax_cb = divider.append_axes("right", size="5%", pad="6%")
    ax_flux.set_facecolor('0.95')
    # Does not work
    #ax_phi = pl.subplot2grid((10, 8), (0, 0), colspan=6, rowspan=7)
    # ax_flux = pl.subplot2grid((10, 8),  (0, 1), colspan=4, rowspan=7)
    #ax_cb = pl.subplot2grid((10, 8), (1, 0), colspan=6, rowspan=1)
    # old, ok for h < 10
    #ax_phi  = fig.add_axes([0.15, 0.4,   0.625, 0.55])
    #ax_cb   = fig.add_axes([0.8,  0.4,   0.04,  0.55])
    #ax_flux = fig.add_axes([0.15, 0.12,  0.625, 0.33], facecolor='0.95')
    pc = ax_phi.pcolormesh(X, Y, data.rho,
                       cmap='bwr', norm=custom_norm, shading='auto')
    ax_phi.set_ylim(-0.5, y_max)
    ax_phi.set_aspect('equal', 'box')
    print ("phi axes:", ax_phi.get_axes_locator())
    draw_edge(data, ax_phi)
    from matplotlib.ticker import FixedLocator, ScalarFormatter
    cb = fig.colorbar(pc, cax=ax_cb,
                      format=ScalarFormatter(useMathText=True))
    cb_ticks = np.linspace(custom_norm.vmin, custom_norm.vmax, 5)
    cb.ax.yaxis.set_major_locator(FixedLocator(5))
    #cb.formatter = ScalarFormatter(useMathText=True)
    cb.formatter.set_scientific(True)
    cb.formatter.set_powerlimits((-1, 1))
    cb.set_ticks(cb_ticks)
    #loc_labels = [(t, r'$%g \times 10^{-2}$' % (t * 100)) for t in cb_ticks]
    #loc_labels = [(t, r'$%g$' % (t * 100)) for t in cb_ticks]
    #for i, loc_label in enumerate(loc_labels):
    #    loc, label = loc_label
    #    if abs(loc) < 1e-6:
    #        loc_labels[i] = (0, "$0$")
    #cb.ax.set_yticklabels([t[1] for t in loc_labels])
    #cb = pl.colorbar()
    #ax_phi.set_xlabel("$x/l_{ee}$")
    pl.setp(ax_phi.get_xticklabels(), visible=False)
    #pl.setp(ax_phi.get_xtickmarks(), visible=False) # no such function
    #ax_phi.set_xticks([])
    ax_phi.set_ylabel("$y/l_{ee}$")
    cb.set_label(r"Bulk potential $\phi(x, y)$")
    #import matplotlib._cntr as cntr
    #c = cntr.Cntr(X, Y, psi_XY, 31)
    segments = cs.allsegs
    levels = cs.levels
    arrows = []
    for i_lev in range(0, len(segments)):
        #if np.abs(levels[i]) < 0.001: continue
        polygons = segments[i_lev]
        for poly in polygons:
           #if i == 0 or i == len(segments) - 1:
           #poly = poly[:int(len(poly)/2)]
           #print ("lev = ", levs[i], "poly:",  poly[0:2], "...", poly[-2:])
           def cut(t):
               if data.sel != 'iso' and data.sel != '': return False
               x_t = t[0]
               y_t = t[1]
               #if np.abs(data.h) < 0.001:
               #    if np.abs(y_t) < 0.05: return True
               #    return False
               #print ("cut?")
               #if abs(x_t - 0.0) < 0.02 and y_t > data.h: return True
               #if data.sel == 'iso' and data.h > 0.001:
               #    pass
                   #if np.abs(levs[i_lev]) > 0.499 and y_t > data.h:
                   #    if abs(x_t) < 0.03:
                   #        return True
               i_x = np.argmin(np.abs(x_t - X[:, 0]))
               i_y = np.argmin(np.abs(y_t - Y[0, :]))
               print ("cut?: ", i_x, i_y)
               i_x = max(2, min(i_x, len(X[:, 0])- 2))
               if abs(psi_XY[i_x - 1, i_y] - psi_XY[i_x + 1, i_y]) > 0.4:
                   return True
               return False
           x_seg = [t[0] for t in poly if not cut(t)]
           y_seg = [t[1] for t in poly if not cut(t)]
           if data.sel == 'iso' and data.h < 0.001:
               pass
               x_seg.insert(0, 0.0)
               y_seg.insert(0, data.h)
           x_seg = np.array(x_seg)
           y_seg = np.array(y_seg)
           if len(x_seg) < 10: continue
           if np.abs(levs[i_lev]) < 0.05:
               print ("x = ", x_seg, "y = ", y_seg)
           print ("filtered:", len(x_seg), len(poly))
           z_seg = x_seg + 1j * y_seg
           seg_length = np.sum(np.abs(z_seg[1:] - z_seg[:-1]))
           if seg_length < 1: continue
           R_0 = y_max*0.9 - data.h 
           z0 = 0.0 + 1j * data.h
           sgn = 1.0
           if i_lev % 2 : sgn = -1.0
           i_end = np.argmin(np.abs(np.abs(z_seg - z0) - R_0)
                             - 0.001 *  sgn * x_seg)
           i_arr = i_end - 5
           if i_arr >= len(x_seg): continue
           #i_arr = int(1*len(x_seg)/3)
           #pl.plot(x_seg, y_seg, 'k-')
           #new_segs = np.array([[[x_seg[t], y_seg[t]]
           #                      for t in range(len(x_seg))]])
           lw = []
           new_segs = []
           for i in range(len(x_seg) - 1):
                new_segs.append([ [x_seg[i],     y_seg[i]    ],
                                  [x_seg[i + 1], y_seg[i + 1]] ])
                xm = (x_seg[i] + x_seg[i + 1])/2
                ym = (y_seg[i] + y_seg[i + 1])/2
                ix = np.argmin(np.abs(data.x - xm))
                iy = np.argmin(np.abs(data.y - ym))
                if abs(data.x[ix] - xm) > 0.1:
                    print ("large deviation in x: ", xm, data.x[ix], ix)
                if abs(data.y[iy] - ym) > 0.1:
                    print ("large deviation in x: ", ym, data.y[ix], iy)
                jx = data.jx[ix, iy]
                jy = data.jy[ix, iy]
                J   = np.sqrt(np.abs(jx)**2 + np.abs(jy)**2) 
                if data.sel == 'iso':
                    J0  = 1.0/2.0/np.pi * 0.2
                    lw0 = 1.5
                    lwid = 1.5 * np.abs(J)/J0
                else:
                    J0  = 0.05
                    lwid = np.abs(J)/J0 * 1.5
                #lwid = 1.5 + np.log((np.abs(J) + 1e-4)/J0) * lw0
                if lwid < 0.2: lwid = 0.2
                if lwid > 3.5: lwid = 3.5
                #lwid = 1.0
                lw.append(lwid)
           new_segs = np.array(new_segs)
           #print ("lw = ", lw)
           lw = np.array(lw)
           lc = LineCollection(new_segs, linewidths=np.array(lw),
                               color='black')
           lc.set_alpha(0.5)
           ax_phi.add_collection(lc)
           
           x_arr = float(x_seg[i_arr])
           y_arr = float(y_seg[i_arr])
           #i_end = i_arr - 5
           dx_arr = x_seg[i_end] - x_arr
           dy_arr = y_seg[i_end] - y_arr
           print ("arr", x_arr, y_arr, dx_arr, dy_arr)
           if data.sel != 'sin':
              arr = ax_phi.arrow(x_arr, y_arr, dx_arr, dy_arr, color='black',
                                shape='full', overhang=0.25,
                                width=0.001, head_width=0.3)
           else:
              arr = ax_phi.arrow(x_arr, y_arr, dx_arr, dy_arr, color='black',
                                shape='full', overhang=0.25,
                                width=0.001, head_width=0.1)
               
           arrows.append(arr)
    #nlist = c.trace()
    #print ("cs.colls = ", cs.collections)
    #paths = cs.collection[1].get_paths()
    #for path in paths:
    #    print ("path = ", path)
    # cs = pl.contour(X, Y, data.psi, 31, colors='black',
    #         lw=0.5, linestyles='solid')
    #for i in range(len(cs.collections)):
    #    path = cs.collections[i].get_paths()[0].vertices
    #    x = path[:, 0]
    #    y = path[:, 1]
    #    plt.plot(x, y, 'g--')
    #pl.title(r"Electric potential  $\phi(x, y)$, $h = %gl_{ee}$" % data.h)
    #pl.show()
    ax_flux.plot(data.x, data.f_s, 'g-', label='Edge flux')
    ax_flux.plot(data.x, visc_phi(data), 'k--', label='Viscous')
    j0 = np.argmin(np.abs(data.y))
    print ("j0 = ", j0, "y = ", data.y[j0])
    #ax_flux.plot(data.x, data.jx[:, j0], 'm-', label='Edge current')
    #ax_flux.plot(data.x, data.jy[:, j0], 'b-', label='Edge source')
    #ax_flux.set_xlabel("")
    ax_flux.set_xlabel("$x/l_{ee}$")
    ax_flux.set_ylabel("Edge potential $V_P(x)$")
    ax_flux.set_xlim(np.min(X), np.max(X))
    print ("X lim: ", np.min(X), np.max(X))
    ax_flux.plot([np.min(X), np.max(X)], [0.0, 0.0], 'k--')
    ax_flux.legend()
    ax_flux.xaxis.set_major_locator(pl.AutoLocator())
    print ("ticks: ", ax_flux.get_xticks())
    #ax_flux.axhspan(-1e-5, 1e-5, np.min(X), np.max(X))
    #cb.set_label(r"$\phi(x, y)$")
    
def view_rho(data, fname, sel, maxv = 0.05):
    pl.figure()
    Y, X = np.meshgrid(data.y, data.x)
    pl.pcolormesh(X, Y, data.rho, cmap='bwr', vmin=-maxv, vmax=maxv,
              shading='auto')
    draw_edge(data, pl.gca())
    pl.gca().set_aspect('equal', 'box')
    cb = pl.colorbar()
    pl.xlabel("$x/l_{ee}$")
    pl.ylabel("$y/l_{ee}$")
    cb.set_label(r"$\phi(x, y)$")
    pl.title(r"Electric potential  $\phi(x, y)$, $h = %gl_{ee}$" % data.h)
    #pl.show()
    

def view_directivity(data, fname, sel):
    fig = pl.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
    pl.title("Directivity")
    print ("directivity:", data.R_dir)
    for i in range(len(data.R_dir)):
        print ("show directivity", data.R_dir[i])
        #print ("directivity data: ", data.jr[i][0],
        #        data.jr[i][1] / data.jr[1][1].max())
        ax.plot(data.jr[i][0],
                data.R_dir[i] * data.jr[i][1] / data.jr[1][1].max(),
                label=r'$|{\bf r} - {\bf r}_s| = %g l_{ee}$' % data.R_dir[i])
    pl.legend()
    #pl.show()

def view_data(data, fname, sel):
    #print "view_data: placeholder"
    #pl.figure()
    #pl.plot(data.x, data.f_s, label='f_s')
    #pl.show()
    view_fs_and_rho(data, fname, sel)
    view_fs(data, fname, sel)
    save_fs(data, fname, sel)
    view_fs_scaled(data, fname, sel)
    view_fs_pp(data, fname, sel)
    
    view_fs_gamma(data, fname, sel)
    view_rho_b(data, fname, sel)
#<<<<<<< HEAD
    view_rho(data, fname, sel, 0.3)
    vmax = 0.02
    if data.sel == 'sin' or data.sel == 'cos':
        vmax = 0.05
    #view_rho_comb2(data, fname, sel, vmax)
    view_rho_comb2(data, fname, sel, vmax, -vmax)
    #view_psi(data, fname, sel)
    #view_psi_alt(data, fname, sel)
    #view_directivity(data, fname, sel)
#=======

#    view_rho(data, fname, sel, 0.3)
#    view_rho(data, fname, sel, 0.05)
#    view_psi(data, fname, sel)
#    view_psi_alt(data, fname, sel)
#    view_directivity(data, fname, sel)
#>>>>>>> 3530290e7de93f95028febeab6a211e698cdadb2
    #pl.show()


def load_or_prepare_data(fname, selector, x, R_dir):
    import os
    import os.path
    preproc_fname = "preproc/%s-%s" % (selector, fname)
    try:
        if not os.path.exists(preproc_fname):
            raise Exception("preprocessed file does not exist")
        if os.path.getmtime(fname) > os.path.getmtime(preproc_fname):
            raise Exception("obsolete preprocessed file")
        d = np.load(preproc_fname)
        if len(d['x']) != len(x):
            raise Exception("x is different, need to update")
        if linalg.norm(d['x'] - x) > 1e-6:
            raise Exception("precomputed data for different values of x")
        if d['selector'] != selector:
            raise Exception("data for a different selector")
        if d['orig_data'] != fname:
            raise Exception("precomuted data for another data file")
        data = FlowData()
        data.x = d['x']
        data.y = d['y']
        data.sel = d['selector']
        data.i_b = d['i_b']
        data.rho = d['rho']
        data.drho = d['drho']
        data.psi = d['psi']
        data.psi_alt = d['psi_alt']
        data.h = d['h']
        data.gamma = d['gamma']
        data.f_s = d['f_s']
        #data.rho_b = d['rho_b']
        data.jx = d['jx']
        data.jy = d['jy']
        data.R_dir = d['R_dir']
        data.jr = d['jr']
        data.jtheta =d['jtheta']
        data.r_src = d['r_src']
        data.j_src = d['j_src']
        if 'gamma1' in d.keys():
            data.gamma1 = d['gamma1']
            data.gamma2 = data.gamma - data.gamma1
        return data
    
    except:
        import traceback
        traceback.print_exc()
        data = prepare_data(fname, selector, x, R_dir)
        np.savez(preproc_fname, x_orig=x, x=data.x, y=data.y, i_b = data.i_b,
                 rho=data.rho, drho=data.drho,
                 psi=data.psi, psi_alt=data.psi_alt, h=data.h,
                 f_s=data.f_s, jx=data.jx, jy=data.jy,
                 R_dir=np.array(data.R_dir), jr=data.jr,
                 jtheta=data.jtheta, r_src=data.r_src, j_src=data.j_src,
                 orig_data=fname, selector=selector,
                 gamma=data.gamma, gamma1=data.gamma1)
        return data
        
    
#for fname in ['wh-data-kmin=0.01-kmax=30-qmax=5000.npz']: #sys.argv[1:]:
if __name__ == '__main__':
    selector = sys.argv[1]
    R_dir = [] # [0.3, 0.5, 1.0, 2.0, 3.0, 5.0]
    #R_dir = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]
    for fname in sys.argv[2:]:
        x = np.linspace(-10.1, 10.1, 1001)
        #x = np.linspace(-10.1, 30.1, 2001)
        #x = np.linspace(-1.0, 30.1, 2001)
        data = load_or_prepare_data(fname, selector, x, R_dir)
        view_data(data, fname, selector)

    #x_large = np.linspace(3.0, 20.0, 100)
    #ax_fs_and_rho_zoom.plot(x_large, -1.0/np.pi/x_large**2, '--',
    #                        label='Hydrodynamics')
    #ax_fs_and_rho.plot(x_large, -1.0/np.pi/x_large**2, '--',
    #                   label='Hydrodynamics')
    #x_small = np.linspace(0.01, 0.5, 100)
    #ax_fs_and_rho.plot(x_small, -1.0/np.pi * np.log(1.0/np.abs(x_small)),
    #                   '--', label='Backscattering')
    #ax_fs_and_rho.plot(x, 0.0 + 0.0*x,'k--')
    #ax_fs_and_rho_zoom.plot(x, 0.0 + 0.0*x,'k--')
    for ax in [ax_fs, ax_rho_b, ax_fs_and_rho, ax_fs_sc]:
        ax.legend()
    #ax_fs_and_rho.set_xlim(-1.0, 10.0)
    #ax_fs_and_rho.set_ylim(-1.6, 0.2)
    #ax_fs_and_rho_zoom.set_xlim(2.0, 20.0)
    #ax_fs_and_rho_zoom.set_ylim(-0.025, 0.005)
    #ax_fs_and_rho.legend(loc=4)
    pl.show()
