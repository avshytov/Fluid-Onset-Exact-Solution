import numpy as np
from scipy import special
from flows import Flow
import frhs

class DiffuseFlow(Flow):
    def __init__ (self, k, K_up, path_up, K_dn, path_dn):
        Flow.__init__(self, k, K_up, path_up, K_dn, path_dn)
        #self.gamma     = self.K.gamma

    def J(self, q):
        return 0 * 1.0/np.pi + 0.0 * q
    
    def rho_direct(self, q):
        return frhs.Frho(self.gamma, self.k, q)
        #- frhs.Frho(self.k, -self.q)
    
    def Omega_direct(self, q):
        return frhs.Fomega(self.gamma, self.k, q)
        #+ frhs.Fomega(self.k, -self.q)
        
    def wall_flux(self):
        gamma = self.gamma
        gamma1 = self.gamma1
        Krho_star   = self.Krho_star
        Komega_star = self.Komega_star
        abs_k = np.abs(self.k)
        sgn_k = np.sign(self.k)
        
        flux = 1.0 / np.pi
        flux += -0.25 * gamma / abs_k / Krho_star**2
        flux += - 0.5 * abs_k / gamma1 * (1.0 - 1.0 / Komega_star**2)
        #flux += sgn_k * frhs.Fomega_star(self.gamma, self.k)
        return flux
        
    def _rho_plus(self, q, Krho_p, chi_p):
        #q = self.q_dn 
        k2 = self.k**2 + q**2
        abs_k = np.abs(self.k)
        #Krho_p = self.K_dn.rho_plus()
        Krho_star = self.Krho_star
        rho = 1j * q / k2 + 0.5 /(abs_k + 1j * q) * Krho_p / Krho_star
        return rho 

    def _rho_minus(self, q, Krho_m, chi_m):
        #Krho = self.K.rho_plus(q)
        #gamma1 = self.gamma1
        #q = self.q_up
        #Krho_m = self.K_up.rho_minus()
        Krho_star = self.Krho_star
        k = self.k
        abs_k = np.abs(k)
        rho = frhs.Frho(self.gamma, k, -q)
        rho += -0.5 * Krho_m / Krho_star / (abs_k + 1j * q)
        #rho += - gamma1 / abs_k / (abs_k + 1j * q) * Krho_m / self.Krho_star
        return rho

    def _D_plus(self, q):
        #q = self.q_up 
        return 1j / np.pi * 0.0  + 0.0 * q
    
    #def D_minus(self):
    #    return -1j * self.gamma * self.rho_minus()

    def _Omega_plus(self, q, Komega_p, psi_p):
        #q = self.q_dn
        #Komega_p = self.K_dn.omega_plus()
        omega = Komega_p / self.Komega_star - 1.0
        return omega * self.k / self.gamma1 * 0.5
    
    def _Omega_minus(self, q, Komega_m, psi_m):
        #q = self.q_up
        #Komega_m = self.K_up.omega_minus()
        omega  = (1.0 - Komega_m / self.Komega_star)
        omega *= 0.5 * self.k / self.gamma1
        omega -= frhs.Fomega(self.gamma, self.k, -q)
        return omega

class DiffuseFlow_sym(Flow):
    def __init__ (self, k, K_up, path_up, K_dn, path_dn):
        Flow.__init__(self, k, K_up, path_up, K_dn, path_dn)
        #self.gamma     = self.K.gamma

    def J(self, q):
        return 0 * 1.0/np.pi + 0.0 * q
    
    def rho_direct(self, q):
        k2 = self.k**2 + q**2
        return 1j * q / k2 * self.K_up.K.rho(self.k, q)
        #frhs.Frho(self.gamma, self.k, q)
        #- frhs.Frho(self.k, -self.q)
    
    def Omega_direct(self, q):
        return self.k / self.gamma * 0.5 * (1.0 - self.K_up.K.omega(self.k, q))
        #return frhs.Fomega(self.gamma, self.k, q)
        #+ frhs.Fomega(self.k, -self.q)

    def flux_down(self):
        return -1.0/np.pi
        
    def wall_flux(self):
        gamma = self.gamma
        gamma1 = self.gamma1
        Krho_star   = self.Krho_star
        Komega_star = self.Komega_star
        abs_k = np.abs(self.k)
        sgn_k = np.sign(self.k)
        
        flux = 1.0 / np.pi
        flux += -0.25 * gamma / abs_k / Krho_star**2
        flux += - 0.5 * abs_k / gamma1 * (1.0 - 1.0 / Komega_star**2)
        #flux += sgn_k * frhs.Fomega_star(self.gamma, self.k)
        return flux
        
    def _rho_plus(self, q, Krho_p, chi_p):
        #q = self.q_dn 
        k2 = self.k**2 + q**2
        abs_k = np.abs(self.k)
        #Krho_p = self.K_dn.rho_plus()
        Krho_star = self.Krho_star
        rho = 1j * q / k2 + 0.5 /(abs_k + 1j * q) * Krho_p / Krho_star
        return rho 

    def _rho_minus(self, q, Krho_m, chi_m):
        #Krho = self.K.rho_plus(q)
        #gamma1 = self.gamma1
        #q = self.q_up
        #Krho_m = self.K_up.rho_minus()
        Krho_star = self.Krho_star
        k = self.k
        abs_k = np.abs(k)
        #rho = frhs.Frho(self.gamma, k, -q)
        rho = -0.5 * Krho_m / Krho_star / (abs_k + 1j * q)
        #rho += - gamma1 / abs_k / (abs_k + 1j * q) * Krho_m / self.Krho_star
        return rho

    def _D_plus(self, q):
        #q = self.q_up 
        return 1j / np.pi * 0.0  + 0.0 * q
    
    #def D_minus(self):
    #    return -1j * self.gamma * self.rho_minus()

    def _Omega_plus(self, q, Komega_p, psi_p):
        #q = self.q_dn
        #Komega_p = self.K_dn.omega_plus()
        omega = Komega_p / self.Komega_star - 1.0
        return omega * self.k / self.gamma1 * 0.5
    
    def _Omega_minus(self, q, Komega_m, psi_m):
        #q = self.q_up
        #Komega_m = self.K_up.omega_minus()
        omega  = (1.0 - Komega_m / self.Komega_star)
        omega *= 0.5 * self.k / self.gamma1
        #omega -= frhs.Fomega(self.gamma, self.k, -q)
        return omega

