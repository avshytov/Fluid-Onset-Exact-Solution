import contours
from edge import EdgeInjectedFlow
from diffuse import DiffuseFlow
from generic import GenericFlow
import pylab as pl
import numpy as np
from xkernel_new import WHKernels
from contours import make_paths_and_kernels
from flows import CombinedFlow

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

    def drho_dct(q):
        kqgamma = np.sqrt(k**2 + q**2 + gamma**2)
        drho = 1j * gamma1 * flow.omega_plus_dn() / kqgamma**3
        drho += dF_rho(q)
        return drho

    def dOmega_dct(q):
        k2 = k**2 + q**2
        kqgamma = np.sqrt(k2 + gamma**2)
        O1 = gamma * flow.rho_plus_dn() - 2.0 * 1j * D_plus_dn() / k2
        dOmega = O1 * (-1j * k2) / kqgamma**3
        dOmega += dF_omega(q)

    def dJ(q):
        return 0.0 * q + 0.0j

    corr_flow = GenericFlow(k, K_up, path_up, K_dn, path_dn)
    ind_flow.solve(drho_dct, dJ, dOmega_dct)

    ind_flux = ind_flow.wall_flux()
    diff_flux = diffuser.wall_flux()
    df_s = ind_flux / (1.0/np.pi - diff_flux)

    corr_flow = GenericFlow(k, K_up, path_up, K_dn, path_dn)
    corr_flow.add(ind_flow, 1.0)
    corr_flow.add(diffuse,  df_s)

    return corr_flow, df_s

    
    

def solve_for_correction(gamma, gamma1, k):
    K = WHKernels(gamma, gamma1)
    path_up, K_up, path_dn, K_dn = make_paths_and_kernels(K, k)
    orig_I   = EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn)
    diffuse  = DiffuseFlow(k, K_up, path_up, K_dn, path_dn)

    orig_flux = orig_I.wall_flux()
    diff_flux = diffuse.wall_flux()
    orig_fs   = orig_flux / (1.0/np.pi - diff_flux)

    orig_flow = CombinedFlow(K_up, path_up, K_dn, path_dn)
    orig_flow.add(orig_I, 1.0)
    orig_flow.add(diffuse,  f_s)

    def dF_rho_I(q):
        return 0.0 * q + 0.0j

    def dF_omega_I(q):
        k2 = k**2 +q**2
        kqgamma = np.sqrt(k2 + gamma**2)
        return - k2 / kqgamma**3

    def dF_rho_s(q):
        k2 = k**2 +q**2
        kqgamma = np.sqrt(k2 + gamma**2)
        return 1j * k / 2.0 / kqgamma

    def dF_omega_s(q):
        k2 = k**2 +q**2
        kqgamma = np.sqrt(k2 + gamma**2)
        return q * gamma / kqgamma**3 
        
    
    def dF_rho_tot(q):
        return dF_rho_I + f_s * dF_rho_s
    
    def dF_omega_tot(q):
        return dF_omega_I + f_s * dF_omega_s

    
    
    corr_I, df_I     = find_correction(orig_I,    k, diffuse,
                                       dF_rho_I, dF_omega_I)
    coff_s, df_s     = find_correction(diffuse,   k, diffuse,
                                       dF_rho_s, dF_omega_s)
    corr_tot, df_tot = find_correction(orig_flow, k, diffuse,
                                       dF_rho_tot, dF_omega_tot) 

    yvals = np.linspace(-1.0, 10.0, 1101)
    result = dict()
    result['orig:f_s'] = f_s
    result['orig:rho'] = orig_flow.rho_y(y)
    result['orig:jx']  = orig_flow.jx_y(y)
    result['orig:jy']  = orig_flow.jy_y(y)

    result['corr:f_s'] = df_tot
    result['corr:rho'] = corr_tot.rho_y(y)
    result['corr:jx']  = corr_tot.jx_y(y)
    result['corr:jy']  = corr_tot.jy_y(y)

    result['corr_I:f_s'] = df_I
    result['corr_I:rho'] = corr_I.rho_y(y)
    result['corr_I:jx']  = corr_I.jx_y(y)
    result['corr_I:jy']  = corr_I.jy_y(y)
    
    result['corr_s:f_s'] = df_s
    result['corr_s:rho'] = corr_s.rho_y(y)
    result['corr_s:jx']  = corr_s.jx_y(y)
    result['corr_s:jy']  = corr_s.jy_y(y)

    return result

    
    
