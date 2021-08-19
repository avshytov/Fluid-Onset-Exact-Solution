import numpy as np
from scipy import special
from flows import Flow

class InjectedFlow(Flow):
    def __init__ (self, h, k, K_up, path_up, K_dn, path_dn):
        Flow.__init__(self, K_up, path_up, K_dn, path_dn)
        #self.K = K
        self.h = h
        #self.path_up = path_up
        #self.path_dn = path_dn
        self.exp_kh = np.exp(-np.abs(k) * h)
        self.chi_p_up = self.chi_plus(self.q_up)
        #self.path_up.eval(self.chi_plus)
        self.chi_m_dn = self.chi_minus(self.q_dn)
        #self.path_dn.eval(self.chi_minus)
        #self.chi_up = self.chi_up()
        #self.chi_dn = self.chi_dn()
        self.chi_p_dn = self.chi_dn() - self.chi_m_dn
        self.chi_m_up = self.chi_up() - self.chi_p_up

    def J(self, q):
        return np.exp(1j * q * self.h)

    def rho_direct(self, q):
        exp_qh = np.exp(1j * self.h * q)
        Krho = self.K_up.rho(q)
        #Krho = self.K.rho_minus() / self.K.rho_plus()
        return 1.0/self.gamma * (1 - Krho) * exp_qh

    def Omega_direct(self, q):
        return 0.0 * q + 0.0

    def chi_star(self):
        return self.chi_minus(-1j * np.abs(self.k))
        #z = -1j * abs(self.k)
        #z1 = self.path_up.points()
        #exp_qh = np.exp(1j * z1 * self.h)
        #Kinv = 1.0/self.K_up.rho_minus() - 1.0
        #f_chi = exp_qh / (z - z1) * Kinv / self.gamma
        #return self.path_up.integrate_array(f_chi) / 2.0 / np.pi / 1j

    def chi(self, Kminus, q):
        Krhoinv = 1.0/Kminus - 1.0
        exp_qh = np.exp(1j * q * self.h)
        return 1.0/self.gamma * Krhoinv * exp_qh
    
    def chi_up(self):
        return self.chi(self.K_up.rho_minus(), self.q_up)

    def chi_dn(self):
        return self.chi(self.K_dn.rho_minus(), self.q_dn)
     
    #
    # Integral over path_up, evaluated at path_dn
    #
    def chi_minus(self, z):
        z1 = self.path_up.points()
        #exp_qh = np.exp(1j * z1 * self.h)
        #Kinv = 1.0/self.K_up.rho_minus() - 1.0
        z_z1 = np.outer(z, 1.0 + 0.0 * z1) - np.outer(1.0 + 0.0 * z, z1)
        f_chi = self.chi_up() / z_z1 #/ (z - z1) 
        return self.path_up.integrate_array(f_chi) / 2.0 / np.pi / 1j

    
    #
    # Integral over path_dn, evaluated at path_up
    #
    def chi_plus(self, z):
        #gamma = self.gamma
        #h = self.h
        z1 = self.path_dn.points()
        #exp_qh = np.exp(1j * z1 * self.h)
        #Kinv = 1.0/self.K_dn.rho_minus() - 1.0
        z_z1 = np.outer(z, 1.0 + 0.0 * z1) - np.outer(1.0 + 0.0 * z, z1)
        f_chi = -self.chi_dn() / z_z1 #/ (z - z1)
        #f_chi = - exp_qh / (z - z1) * Kinv / gamma
        I =  self.path_dn.integrate_array(f_chi) / 2.0 / np.pi / 1j
        return I
    
    def wall_flux(self):
        k = self.k
        gamma = self.gamma
        gamma1 = self.gamma1
        exp_kh =  self.exp_kh
        flux  = - gamma * gamma1 / 2.0 / k**2 / self.Krho_star**2 * exp_kh
        flux += exp_kh / self.Komega_star**2 
        flux += self.chi_star() / self.Krho_star * gamma
        return flux

    #
    # The following code is somewhat convoluted. It solves the following issue:
    # the function rho+ includes an oscillating term exp(iqh). When
    # one deforms the contour to the lower half-plane, this turns into
    # a growing exponential if y < h. This indicates a singularity
    # at y=h. To handle this, we extract all the contributions
    # proportional to exp(iqh) and handle them separately.
    #
    # For this reason, we write
    #
    # rho_p = rho_sing * exp(iqh) + rho_reg
    #
    # An important point: the function chi+ also contributes
    # to the singularity. We write it as
    #
    # chi_p = chi - chi_m    where chi = (1/K_m - 1) exp(iqh)
    #
    # Then K_p chi_p = K_p chi - K_p chi_m
    #
    # The term K_p chi_m contributes to the regular part, while
    #
    # K_p chi can be transformed to the form
    #
    # (1/K - K_p) * exp(i q h)
    #
    # Together with (K_p - 1)*exp(iqh), this gives
    # (1/K - 1) * exp(i * q * h)
    #
    # This may look like a free-space contribution, but it is not:
    # the latter is given by
    #
    #  (1 - K) * exp(i q h)
    #
    # Hence one may write the excess delta rho = (1/K + K - 2) * exp(i q h)
    #
    #
    def _rho_plus_reg(self, q, Krho_p, chi_minus):
        gamma  = self.gamma
        gamma1 = self.gamma1
        k = self.k
        Krho_star = self.Krho_star
        abs_k  = np.abs(k)
        k2 = k**2 + q**2
        exp_kh    = self.exp_kh
        rho  = gamma1 * exp_kh / abs_k / (abs_k + 1j * q) * Krho_p / Krho_star
        rho += - Krho_p * chi_minus
        return rho

    #
    #  The singular part: the sum of (K_p - 1) + K_p * chi
    #  and the divergence term. Note the exponential factor is not included
    #
    def _rho_plus_sing(self, q, Krho):
        k2 = self.k**2 + q**2
        #K = self.K_dn.rho(q)
        rho = -2.0 * self.gamma1 / k2 
        rho +=  1.0/self.gamma * (1.0 / Krho - 1.0)
        return rho
    #
    # The excess: minus free space singularity
    #
    def _drho_plus_sing(self, q, Krho):
        k2 = self.k**2 + q**2
        rho = -2.0 * self.gamma1 / k2 
        rho +=  1.0/self.gamma * (Krho - 2.0 + 1.0/Krho)
        return rho 
        #return 1.0/self.gamma * (Krho_p - 2.0 + self.K_up.rho(q))

    #
    # Full rho_p is given by the sum of the regular and singular terms
    #
    def _rho_plus(self, q, Krho_p, chi_p):
        exp_qh = np.exp(1j * q * self.h)
        Krho = self.K_dn.rho(q)
        Krho_m = Krho * Krho_p
        chi = self.chi(Krho_m, q)
        chi_m = chi - chi_p
        rho  =  self._rho_plus_reg(q, Krho_p, chi_m)
        rho +=  self._rho_plus_sing(q, Krho) * exp_qh 
        #rho += 1.0/gamma * exp_qh * (Krho_p - 1.0)
        return rho

    #
    # The more standard definition of rho_p, should be
    # equivalent to the custom-tailored above
    #
    def _rho_plus_old(self, q, Krho_p, chi_p):
        exp_qh = np.exp(1j * q * self.h)
        Krho_star = self.Krho_star
        k2 = k**2 + q**2
        exp_kh    = self.exp_kh
        rho  = gamma1 * exp_kh / abs_k / (abs_k + 1j * q) * Krho_p / Krho_star
        rho += Krho_p * chi_p
        rho += -2.0 * self.gamma1 / k2 * exp_qh 
        rho +=  1.0/self.gamma * (Krho_p - 1.0) * exp_qh
        return rho
         

    #
    # Evaluate the excess (reg + sing) at the lower contour
    #
    def drho_plus_dn(self):
        #print ("bulK: drho_plus")
        q = self.q_dn
        Krho_p = self.K_dn.rho_plus()
        Krho = self.K_dn.rho(q)
        chi_m = self.chi_m_dn
        exp_qh = np.exp(1j * q * self.h)
        rho  =  self._rho_plus_reg(q, Krho_p, chi_m)
        rho +=  self._drho_plus_sing(q, Krho) * exp_qh 
        #rho += 1.0/gamma * exp_qh * (Krho_p - 1.0)
        return rho 

    #
    # rho_minus is uneventful
    #
    def _rho_minus(self, q, Krho_m, chi_m):
        abs_k = np.abs(self.k)
        rho  =  Krho_m * chi_m
        Krho_star = self.Krho_star
        exp_k = self.exp_kh
        gamma1 = self.gamma1
        rho += -gamma1 * Krho_m / Krho_star * exp_k / abs_k / (abs_k + 1j * q) 
        return rho
    
    
    def _D_plus(self, q):
        return 1j * np.exp(1j * self.h * q)
    
    def _Omega_plus(self, q, Ko_p, psi_p):
        #Ko_p = self.Komega_p
        #Ko_p = self.K_up.omega_plus()
        sgn_k = np.sign(self.k)
        return Ko_p / self.Komega_star * sgn_k * self.exp_kh
    
    def _Omega_minus(self, q, Ko_m, psi_m):
        #Ko_m = self.K_up.omega_minus()
        sgn_k = np.sign(self.k)
        return - Ko_m / self.Komega_star * sgn_k * self.exp_kh

    #
    # Few simple wrappers
    #
    def _rho_plus_reg_dn(self):
        return self._rho_plus_reg(self.q_dn,
                                  self.K_dn.rho_plus(), self.chi_m_dn)

    def _rho_plus_sing_dn(self):
        return self._rho_plus_sing(self.q_dn, self.K_dn.rho(self.q_dn))
    
    def _drho_plus_sing_dn(self):
        return self._drho_plus_sing(self.q_dn, self.K_dn.rho(self.q_dn))

    #
    # Custom density evaluation
    #
    def _rho_y(self, y):
        #print ("bulk: rho(y)")
        res = 0.0 + 0.0j * y
        y_neg = y[ y < 0 ]
        res[ y < 0 ] = self._fourier(self.path_up, self.rho_minus_up(), y_neg) 
        #if y < 0:
        #    return self._fourier(self.path_up, self.rho_minus_dn(), y)
        # Handle the singular term via |y - h|
        y_pos = y[ y >= 0 ]
        yh = np.abs(y_pos - self.h) # even integrand
        rho_sing = self._fourier(self.path_dn, self._rho_plus_sing_dn(), yh)
        rho_reg  = self._fourier(self.path_dn, self._rho_plus_reg_dn(), y_pos)
        res[ y>= 0] = rho_sing + rho_reg
        return res

    #
    # Custom evaluator for the excess term
    #
    def _drho_y(self, y):
        #print ("bulk: rho(y)")
        res = 0.0 + 0.0j * y
        y_neg = y[ y < 0 ]
        res[ y < 0 ] = self._fourier(self.path_up, self.rho_minus_up(), y_neg) 
        #if y < 0:
        #    return self._fourier(self.path_up, self.rho_minus_dn(), y)
        # Handle the singular term via |y - h|
        y_pos = y[ y >= 0 ]
        yh = np.abs(y_pos - self.h) # even integrand
        rho_sing = self._fourier(self.path_dn, self._drho_plus_sing_dn(), yh)
        rho_reg  = self._fourier(self.path_dn, self._rho_plus_reg_dn(), y_pos)
        res[ y>= 0] = rho_sing + rho_reg
        return res
    
        #print ("bulk: drho(y)")
        #if y < 0:
        #    return self._rho_y(y)
        #yh = np.abs(y - self.h) # even integrand
        #drho_sing = self._fourier(self.path_dn,
        #                          self._drho_plus_sing_dn(), yh)
        #rho_reg  = self._fourier(self.path_dn, self._rho_plus_reg_dn(),  y)
        #return drho_sing + rho_reg

    #
    # The free-space singularity excluded from drho
    #
    def rho_sing_y(self, y):
        #if y < 0: return 0.0 * y + 0.0j
        theta = 1.0 + 0.0 * y
        theta[y < 0] = 0.0
        kappa = np.sqrt(self.k**2 + self.gamma**2)
        return 1.0/np.pi * special.kn(0, kappa * np.abs(y - self.h)) * theta

    def _jx_sing(self):
        # exclude the oscillating factor
        return self.jx_q(1j  + 0.0 * self.q_dn, 0.0 * self.q_dn, self.q_dn)
    
    def _jy_sing(self):
        # exclude the oscillating factor
        return self.jy_q(1j  + 0.0 * self.q_dn, 0.0 * self.q_dn, self.q_dn)
    
    def _jx_reg(self):
        # D only gives a singular contribution
        return self.jx_q(0.0 * self.q_dn, self.Omega_plus_dn(), self.q_dn)
    
    def _jy_reg(self):
        # D only gives a singular contribution
        return self.jy_q(0.0 * self.q_dn, self.Omega_plus_dn(), self.q_dn)
    
    def _jx_y(self, y):
        res = 0.0 + 0.0j * y
        y_pos = y[ y >= 0 ]
        y_neg = y[ y <  0 ]
        yh = y_pos - self.h
        abs_yh = np.abs(yh)
        sgn_yh = np.sign(yh)

        res[ y < 0 ] = self._fourier(self.path_up, self.jx_minus_up(), y_neg)
        jx_pos = self._fourier(self.path_dn, self._jx_reg(), y_pos)
        #jy_pos = self._fourier(self.path_dn, self._jy_reg(), y_pos)
        # Divergence contribution is just the free-space term:
        jx_pos += self._fourier(self.path_dn, self._jx_sing(), abs_yh)
        #jy_pos += self._fourier(self.path_dn, self._jy_sing(), abs_yh)* sgn_yh
        res[y >= 0] = jx_pos
        return res
    
    def _jy_y(self, y):
        res = 0.0 + 0.0j * y
        y_pos = y[ y >= 0 ]
        y_neg = y[ y <  0 ]
        yh = y_pos - self.h
        abs_yh = np.abs(yh)
        sgn_yh = np.sign(yh)
        #sgn_yh = 1.0 + 0.0 * yh
        #sgn_yh[ yh < 0 ] = -1

        res[ y < 0 ] = self._fourier(self.path_up, self.jy_minus_up(), y_neg)
        #jx_pos = self._fourier(self.path_dn, self._jx_reg(), y_pos)
        jy_pos = self._fourier(self.path_dn, self._jy_reg(), y_pos)
        # Divergence contribution is just the free-space term:
        #jx_pos += self._fourier(self.path_dn, self._jx_sing(), abs_yh)
        jy_pos += self._fourier(self.path_dn, self._jy_sing(), abs_yh)* sgn_yh
        res[y >= 0] = jy_pos
        return res

        
    
