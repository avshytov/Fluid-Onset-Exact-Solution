import contours
from edge import EdgeInjectedFlow_sym, EdgeInjectedFlow
from diffuse import DiffuseFlow_sym, DiffuseFlow
from generic import GenericFlow
import pylab as pl
import numpy as np
from xkernel_new import WHKernels
from contours import make_paths_and_kernels
from flows import CombinedFlow
from datasaver import DataSaver

def find_correction(flow, k, diffuse_flow, dF_rho, dF_omega):
    K_up = flow.K_up
    K_dn = flow.K_dn
    path_up = flow.path_up
    path_dn = flow.path_dn
    q_up = flow.q_up
    q_dn = flow.q_dn
    K = K_up.K
    gamma  = K.gamma
    gamma1 = K.gamma1

    def drho_dct(q, omega_p):
        kqgamma = np.sqrt(k**2 + q**2 + gamma**2)
        drho = 1j * gamma1 * omega_p / kqgamma**3
        drho += 1j * dF_rho(q)
        return drho

    def dOmega_dct(q, rho_p, D_p):
        k2 = k**2 + q**2
        kqgamma = np.sqrt(k2 + gamma**2)
        O1 = gamma * (rho_p - 2.0 * 1j * gamma1 * D_p / k2)
        dOmega = O1 * (-0.5j * k2) / kqgamma**3
        dOmega += 1j * dF_omega(q)
        return dOmega

    def dJ(q):
        return 0.0 * q + 0.0j

    rho_dct_up   = drho_dct(flow.q_up, flow.Omega_plus_up())
    rho_dct_dn   = drho_dct(flow.q_dn, flow.Omega_plus_dn())
    Omega_dct_up = dOmega_dct(flow.q_up, flow.rho_plus_up(), flow.D_plus_up())
    Omega_dct_dn = dOmega_dct(flow.q_dn, flow.rho_plus_dn(), flow.D_plus_dn())
    J_up = J_dn = 0.0 * flow.q_up # vanishes for hall corrections
    rho_dct_star = 0.0     # Not in use now?
    Omega_dct_star = 0.0
    J_star = 0.0
    
    flux_down = 0.0
    ind_flow = GenericFlow(k, K_up, path_up, K_dn, path_dn)
    #ind_flow.solve(drho_dct, dJ, dOmega_dct)
    ind_flow._solve(rho_dct_up,   rho_dct_dn,
                    Omega_dct_up, Omega_dct_dn,
                    J_up, J_dn,
                    rho_dct_star, Omega_dct_star, J_star, flux_down)

    ind_flux = ind_flow.wall_flux()
    diff_flux = diffuse_flow.wall_flux()
    df_s = ind_flux / (1.0/np.pi - diff_flux)

    corr_flow = CombinedFlow(k, K_up, path_up, K_dn, path_dn)
    corr_flow.add(ind_flow, 1.0)
    corr_flow.add(diffuse_flow,  df_s)

    return corr_flow, df_s

    
    

def solve_for_correction(gamma, gamma1, k, yvals, use_sym = False):
    K = WHKernels(gamma, gamma1)
    path_up, K_up, path_dn, K_dn = make_paths_and_kernels(K, k)
    orig_I_sym   = EdgeInjectedFlow_sym(k, K_up, path_up, K_dn, path_dn)
    diffuse_sym  = DiffuseFlow_sym(k, K_up, path_up, K_dn, path_dn)
    orig_I_one   = EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn)
    diffuse_one  = DiffuseFlow(k, K_up, path_up, K_dn, path_dn)

    #use_sym = False
    if use_sym:
       orig_I = orig_I_sym
       diffuse = diffuse_sym
    else:
       orig_I = orig_I_one
       diffuse = diffuse_one
    
    orig_I   = EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn)
    diffuse  = DiffuseFlow(k, K_up, path_up, K_dn, path_dn)

    orig_flux = orig_I.wall_flux()
    diff_flux = diffuse.wall_flux()
    orig_fs   = orig_flux / (1.0/np.pi - diff_flux)

    orig_flow = CombinedFlow(k, K_up, path_up, K_dn, path_dn)
    orig_flow.add(orig_I, 1.0)
    orig_flow.add(diffuse,  orig_fs)

    def dF_rho_I_sym(q):
        return 0.0 * q + 0.0j

    def dF_rho_I_one(q):
        return k * gamma / (k**2 + gamma**2)**2 / np.pi * 2.0

    def dF_omega_I_sym(q):
        k2 = k**2 + q**2
        kqgamma = np.sqrt(k2 + gamma**2)
        return - k2 / 2.0 / kqgamma**3 * 2.0

    def dF_omega_I_one(q):
        k2 = q**2 + k**2
        kqgamma = np.sqrt(k2 + gamma**2) * 2.0
        kgamma = np.sqrt(k**2 + gamma**2)
        log_q = np.log((q + kqgamma) / kgamma)
        f1 = (1.0 + 2.0j / np.pi * log_q) * k2 / kqgamma
        f2a = (k2 * (gamma**2 - k**2) + 2.0 * gamma**4) / kgamma**4 
        f2 = 2.0j * q / np.pi * f2a
        return -(f1 - f2) / 4.0 / kqgamma**2 * 2.0

    def dF_rho_s_sym(q):
        k2 = k**2 +q**2
        kqgamma = np.sqrt(k2 + gamma**2)
        return k / 2.0 / kqgamma**3

    def dF_rho_s_one(q):
        k2 = q**2 + k**2
        kqgamma = np.sqrt(k2 + gamma**2)
        kgamma = np.sqrt(k**2 + gamma**2)
        log_q = np.log((q + kqgamma) / kgamma)
        f2 = 2.0j / np.pi * q * kqgamma / kgamma**2
        return k / 4.0 / kqgamma**3 * (1.0 + 2j / np.pi * log_q + f2)
        

    def dF_omega_s_sym(q):
        k2 = k**2 +q**2
        kqgamma = np.sqrt(k2 + gamma**2)
        return 0.5 * 1j * q * gamma / kqgamma**3 


    def dF_omega_s_one(q):
        k2 = q**2 + k**2
        kqgamma = np.sqrt(k2 + gamma**2)
        kgamma = np.sqrt(k**2 + gamma**2)
        log_q = np.log((q + kqgamma) / kgamma)
        f2 = 2.0j / np.pi * q * kqgamma / kgamma**2
        return 1j * q * gamma/ 4.0 / kqgamma**3 * (1.0 + 2j / np.pi * log_q + f2)
        
    
    if use_sym:
        dF_rho_I   = dF_rho_I_sym
        dF_omega_I = dF_omega_I_sym
        dF_rho_s   = dF_rho_s_sym
        dF_omega_s = dF_omega_s_sym
    else:
        dF_rho_I   = dF_rho_I_one
        dF_omega_I = dF_omega_I_one
        dF_rho_s   = dF_rho_s_one
        dF_omega_s = dF_omega_s_one
        
    def dF_rho_tot(q):
        return dF_rho_I(q) + orig_fs * dF_rho_s(q)
    
    def dF_omega_tot(q):
        return dF_omega_I(q) + orig_fs * dF_omega_s(q)

    
    
    corr_I, df_I     = find_correction(orig_I,    k, diffuse,
                                       dF_rho_I, dF_omega_I)
    corr_s, df_s     = find_correction(diffuse,   k, diffuse,
                                       dF_rho_s, dF_omega_s)
    corr_tot, df_tot = find_correction(orig_flow, k, diffuse,
                                       dF_rho_tot, dF_omega_tot) 

    #yvals = np.linspace(-1.0, 10.0, 1101)
    result = dict()
    res_orig = dict()
    res_I = dict()
    res_tot = dict()
    res_s = dict()
    res_orig['f_s'] = orig_fs
    res_orig['rho'] = orig_flow.rho_y(yvals)
    res_orig['jx']  = orig_flow.jx_y(yvals)
    res_orig['jy']  = orig_flow.jy_y(yvals)
    result['orig'] = res_orig

    res_tot['f_s'] = df_tot
    res_tot['rho'] = corr_tot.rho_y(yvals)
    res_tot['jx']  = corr_tot.jx_y(yvals)
    res_tot['jy']  = corr_tot.jy_y(yvals)
    result['corr_tot'] = res_tot

    res_I['f_s'] = df_I
    res_I['rho'] = corr_I.rho_y(yvals)
    res_I['jx']  = corr_I.jx_y(yvals)
    res_I['jy']  = corr_I.jy_y(yvals)
    result['corr_I'] = res_I
    
    res_s['f_s'] = df_s
    res_s['rho'] = corr_s.rho_y(yvals)
    res_s['jx']  = corr_s.jx_y(yvals)
    res_s['jy']  = corr_s.jy_y(yvals)
    result['corr_diff'] = res_s
    print ("f_s = ", orig_fs, "df_I = ", df_I, "df_s = ", df_s, "df_tot = ", df_tot, "diff = ", df_tot - df_I - orig_fs * df_s)
    

    return result


def run(gamma, gamma1, kvals, yvals, use_sym, fname):
    data = DataSaver(gamma=gamma, gamma1=gamma1, y=yvals)
    for i_k, k in enumerate(kvals):
        print ("run: k = ", k, "gamma1 = ", gamma1)
        result = solve_for_correction(gamma, gamma1, k, yvals, use_sym)
        data.append_result(k, result)
        if i_k % 10 == 0:
            data.save(fname)
    data.save(fname)


    
def join_arrays(*arrays):
    res_list = []
    for a in arrays:
        res_list.extend(list(a))
    return np.array(res_list)

if __name__ == '__main__':
    kvals_pos = join_arrays( np.linspace(0.0003, 0.0096, 32),
                         np.linspace(0.01, 0.995, 199),
                         np.linspace(1.0, 10.0, 361),
                         np.linspace(10.1, 30.0, 200),
                         np.linspace(31.0, 50.0, 20),
                         np.linspace(52.5, 100.0, 20),
                         np.linspace(105, 300.0, 40)
    )
    kvals_neg = list(-kvals_pos)
    kvals_neg.reverse()
    kvals = join_arrays(np.array(kvals_neg), kvals_pos)
    yvals = np.linspace(-1.0, 10.0, 1101)

    gamma  = 1.0
    gamma1 = 1.0
    use_sym = False
    #use_sym = True
    ver = "01d"

    suffix=""
    if use_sym: suffix="-sym"
    else:       suffix="-one" 
    for gamma1 in [1.0, 0.999, 0.99, 0.975, 0.95, 0.925,
                   0.9, 0.85, 0.8, 0.7, 0.6, 0.5]:
       fname = "hallvisc-data-ver%s-gamma1=%g%s" % (ver, gamma1, suffix)
       run(gamma, gamma1, kvals, yvals, use_sym, fname)


