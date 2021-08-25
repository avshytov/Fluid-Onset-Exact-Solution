import numpy as np
from scipy import special
from flows import Flow
import frhs

class EdgeInjectedFlow(Flow):
    def __init__ (self, k, K_up, path_up, K_dn, path_dn):
        Flow.__init__(self, k, K_up, path_up, K_dn, path_dn)

    def J(self, q):
        return 1.0 + 0.0 * q
    
    def rho_direct(self, q):
        #return 2.0 + 0.0 * self.q  #
        return 2.0 * frhs.Fm(self.gamma, self.k, q)
        
    def Omega_direct(self, q):
        #return 0.0 + 0.0 * self.q
        return 2.0 / np.pi * np.arctan(self.k / self.gamma) + 0.0 * q

    def wall_flux(self):
        k = self.k
        flux  = 1.0 - 2.0 / self.Krho_star
        flux += - self.gamma * self.gamma1 * 0.5 / k**2 / self.Krho_star**2
        flux += 1.0 / self.Komega_star**2
        return flux

    def _rho_plus(self, q, Krho_p, chi_p): 
        gamma  = self.gamma
        gamma1 = self.gamma1
        k = self.k
        #q = self.q_up
        #Krho_p = self.Krho_p
        k2 = self.k**2 + q**2
        rho  = 2.0/gamma * (Krho_p - 1.0) - 2.0 * gamma1 / k2
        abs_k = np.abs(self.k)
        rho += gamma1 / abs_k / (abs_k + 1j * q) * Krho_p / self.Krho_star
        return rho

    def drho_plus_dn(self):
        rho =  self._rho_plus(self.q_dn, self.K_dn.rho_plus(), self.chi_p_dn)
        rho -= 2.0/self.gamma * (1.0 - self.K_up.rho(self.q_dn))
        return rho

    def _rho_minus(self, q, Krho_m, chi_m): 
        #q = self.q_dn
        gamma1 = self.gamma1
        #Krho_m = self.Krho_m
        rho =    - 2.0  * frhs.Fm(self.gamma, self.k, -q)
        #rho = 
        rho += 2.0/self.gamma * (- Krho_m + 1.0) 
        abs_k = np.abs(self.k)
        rho += - gamma1 * Krho_m / self.Krho_star / (abs_k + 1j * q) / abs_k
        return rho

    def _D_plus(self, q):
        return 1j + 0.0 * q
    
    def _Omega_plus(self, q, Ko_p, psi_p):
        sgn_k = np.sign(self.k)
        return Ko_p / self.Komega_star * sgn_k
    
    def _Omega_minus(self, q, Ko_m, psi_m):
        #Ko_m = self.Komega_m
        omega =  2.0 / np.pi * np.arctan(self.k/self.gamma)
        omega -= Ko_m / self.Komega_star
        return omega

    def rho_sing_y(self, y):
        kappa = np.sqrt(self.k**2 + self.gamma**2)
        theta_y = 1.0 + 0.0 * y
        theta_y[y < 0] = 0.0
        return 2.0/np.pi * special.kn(0, kappa * np.abs(y)) * theta_y

class EdgeInjectedFlow_sym(Flow):
    def __init__ (self, k, K_up, path_up, K_dn, path_dn):
        Flow.__init__(self, k, K_up, path_up, K_dn, path_dn)

    def J(self, q):
        return 1.0 + 0.0 * q
    
    def rho_direct(self, q):
        #return 2.0 + 0.0 * self.q  #
        return 2.0/self.gamma * ( - self.K_up.K.rho(self.k, q) + 1.0)
        #2.0 * frhs.Fm(self.gamma, self.k, q)
        
    def Omega_direct(self, q):
        #return 0.0 + 0.0 * self.q
        return 0.0 * q + 0.0j
        #return 2.0 / np.pi * np.arctan(self.k / self.gamma) + 0.0 * q

    def flux_down(self): return 1.0
    
    def wall_flux(self):
        k = self.k
        flux  = 1.0 - 2.0 / self.Krho_star
        flux += - self.gamma * self.gamma1 * 0.5 / k**2 / self.Krho_star**2
        flux += 1.0 / self.Komega_star**2
        return flux

    def _rho_plus(self, q, Krho_p, chi_p): 
        gamma  = self.gamma
        gamma1 = self.gamma1
        k = self.k
        #q = self.q_up
        #Krho_p = self.Krho_p
        k2 = self.k**2 + q**2
        rho  = 2.0/gamma * (Krho_p - 1.0) - 2.0 * gamma1 / k2
        abs_k = np.abs(self.k)
        rho += gamma1 / abs_k / (abs_k + 1j * q) * Krho_p / self.Krho_star
        return rho

    def drho_plus_dn(self):
        rho =  self._rho_plus(self.q_dn, self.K_dn.rho_plus(), self.chi_p_dn)
        rho -= 2.0/self.gamma * (1.0 - self.K_up.rho(self.q_dn))
        return rho

    def _rho_minus(self, q, Krho_m, chi_m): 
        #q = self.q_dn
        gamma1 = self.gamma1
        #Krho_m = self.Krho_m
        #rho =    - 2.0  * frhs.Fm(self.gamma, self.k, -q)
        #rho = 
        rho = 2.0/self.gamma * (- Krho_m + 1.0) 
        abs_k = np.abs(self.k)
        rho += - gamma1 * Krho_m / self.Krho_star / (abs_k + 1j * q) / abs_k
        return rho

    def _D_plus(self, q):
        return 1j + 0.0 * q

    def _D_minus(self, q):
        return 1j + 0.0 * q
    
    def _Omega_plus(self, q, Ko_p, psi_p):
        sgn_k = np.sign(self.k)
        return Ko_p / self.Komega_star * sgn_k
    
    def _Omega_minus(self, q, Ko_m, psi_m):
        sgn_k = np.sign(self.k)
        #Ko_m = self.Komega_m
        #omega =  2.0 / np.pi * np.arctan(self.k/self.gamma)
        omega  = -Ko_m / self.Komega_star * sgn_k
        return omega

    def rho_sing_y(self, y):
        kappa = np.sqrt(self.k**2 + self.gamma**2)
        theta_y = 1.0 + 0.0 * y
        theta_y[y < 0] = 0.0
        return 2.0/np.pi * special.kn(0, kappa * np.abs(y)) * theta_y

