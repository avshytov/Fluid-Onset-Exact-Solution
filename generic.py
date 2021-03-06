import numpy as np
from flows import Flow
from cauchy import cauchy_integral_array
from scipy import linalg
from scipy import optimize

def re_to_complex(p):
    return p[::2] + 1j * p[1::2]

def do_fit(f_fit, q, chi, n):

    n_fit = 10
    i_fit = list([t for t in range(len(q)) if t < n_fit or t > len(q) - n_fit])
    q_fit = np.array([q[t] for t in i_fit])
    chi_fit = np.array([chi[t] for t in i_fit])
    #print ("i_fit", i_fit, "q_fit = ", q_fit, "chi_fit = ", chi_fit)
    def mismatch(p):
        #p_re = p[0::2]
        #p_im = p[1::2]
        #p_complex = p_re + 1j * p_im
        return linalg.norm(f_fit(q_fit, *re_to_complex(p)) - chi_fit)**2

    p0 = np.zeros((2 * n))
    res = optimize.minimize(mismatch, p0)
    p = res['x']
    p_complex = re_to_complex(p)
    print ("fit: p_opt = ", p_complex, 'mismatch:', mismatch(p))
    print (res['message'], "success = ", res['success'])
    #print ("res = ", res)
    return tuple(p_complex)


def fit_const_inv(q, chi):
    def f_fit(q, A, B, C):
        return A + B / q + C / q**2
    return do_fit(f_fit, q, chi, 3)

def fit_lin_const_inv(q, psi):
    def f_fit(q, A, B, C, D):
        return A * q + B + C / q +D / q**2
    return do_fit(f_fit, q, psi, 4)
    

class GenericFlow(Flow):
    def __init__ (self, k,  K_up, path_up, K_dn, path_dn):
        Flow.__init__(self, k, K_up, path_up, K_dn, path_dn)
        #self.k = k
        self.chi_p_up = 0.0 * path_up.points() + 0.0j
        self.chi_m_up = 0.0 * path_up.points() + 0.0j
        self.chi_p_dn = 0.0 * path_dn.points() + 0.0j
        self.chi_m_dn = 0.0 * path_dn.points() + 0.0j
        self.psi_p_up = 0.0 * path_up.points() + 0.0j
        self.psi_m_up = 0.0 * path_up.points() + 0.0j
        self.psi_p_dn = 0.0 * path_dn.points() + 0.0j
        self.psi_m_dn = 0.0 * path_dn.points() + 0.0j
        self._Dplus_dn = 0.0 * path_dn.points() + 0.0j
        self._Dminus_up = 0.0 * path_up.points() + 0.0j
        self.rho_p_dn = 0.0 * path_dn.points() + 0.0j
        self.rho_p_up = 0.0 * path_dn.points() + 0.0j
        self.rho_m_dn = 0.0 * path_up.points() + 0.0j
        self.rho_m_up = 0.0 * path_up.points() + 0.0j
        self.Omega_p_dn = 0.0 * path_dn.points() + 0.0j
        self.Omega_p_up = 0.0 * path_dn.points() + 0.0j
        self.Omega_m_dn = 0.0 * path_up.points() + 0.0j
        self.Omega_m_up = 0.0 * path_up.points() + 0.0j
        self._chi_up    = 0.0 * path_up.points() + 0.0j
        self._chi_dn    = 0.0 * path_dn.points() + 0.0j
        self._psi_up    = 0.0 * path_up.points() + 0.0j
        self._psi_dn    = 0.0 * path_dn.points() + 0.0j
        self._flux = 0.0

    def solve(self, rho_dct, J, Omega_dct, flux_down = 0.0):
        self.rho_direct = rho_dct
        self.J = J
        self.Omega_direct = Omega_dct
        abs_k   = np.abs(self.k)
        def take_star_lim(f):
            eps = 1e-5
            return (f(1j * abs_k + eps) + f(1j * abs_k - eps)) * 0.5

        path_up = self.path_up
        path_dn = self.path_dn
        
        # Extract the sources
        rho_dct_up = self.rho_direct(path_up.points())
        rho_dct_dn = self.rho_direct(path_dn.points())
        rho_dct_star = take_star_lim(self.rho_direct)

        Omega_dct_up   = self.Omega_direct(path_up.points())
        Omega_dct_dn   = self.Omega_direct(path_dn.points())
        Omega_dct_star = take_star_lim(self.Omega_direct)
        #+ self.Omega

        J_up = self.J(path_up.points())
        J_dn = self.J(path_dn.points())
        J_star = self.J(1j * abs_k)

        self._solve(rho_dct_up, rho_dct_dn, Omega_dct_up,
                    Omega_dct_dn, J_up, J_dn, rho_dct_star,
                    Omega_dct_star, J_star, flux_down)
        
    def _solve(self, rho_dct_up, rho_dct_dn, Omega_dct_up,
               Omega_dct_dn, J_up, J_dn,
               rho_dct_star, Omega_dct_star, J_star, flux_down = 0.0):


        abs_k   = np.abs(self.k)
        path_up = self.path_up
        path_dn = self.path_dn
        
# ... and kernels
        Krho_p_dn   = self.K_dn.rho_plus()
        Komega_p_dn = self.K_dn.omega_plus()
        Krho_m_dn   = self.K_dn.rho_minus()
        Komega_m_dn = self.K_dn.omega_minus()
        Krho_m_up   = self.K_up.rho_minus()
        Komega_m_up = self.K_up.omega_minus()
        Krho_p_up   = self.K_up.rho_plus()
        Komega_p_up = self.K_up.omega_plus()
        
        Krho_dn = self.K_dn.rho(self.q_dn)
        Krho_up = self.K_up.rho(self.q_up)
        Komega_dn = self.K_dn.omega(self.q_dn)
        Komega_up = self.K_up.omega(self.q_up) 
       
        # Determine the functions chi and psi on the contours
        #
        #  rho_p / K_p + rho_m / K_m = chi = rho_dct / K_m
        #
        self._chi_up = rho_dct_up / Krho_m_up
        self._chi_dn = rho_dct_dn / Krho_m_dn

        # Determine the behaviour at infinity.
        # Normally, chi should tend to zero at q->infty.
        # If it does not, this indicates a delta-like singularity in
        # the density.
        #self.chi_inf  = 0.5 * (self._chi_up[0] + self._chi_up[-1])
        self.chi_inf, B, C = fit_const_inv(self.q_up, self._chi_up)
        self.chi_inf, B, C = fit_const_inv(self.q_up, rho_dct_up)
        #self.chi_inf += 0.25 * (self._chi_dn[0] + self._chi_dn[-1])
        print ("chi_inf = ", self.chi_inf)
        #self.chi_inf = 0.0
        #
        # Subtract the delta-like singularity
        #
        self._chi_up -= self.chi_inf
        self._chi_dn -= self.chi_inf

        #
        # The same for vorticity
        #        
        #  Omega_p / K_p + Omega_m / K_m = chi = Omega_dct / K_m
        #
        self._psi_up = Omega_dct_up / Komega_m_up
        self._psi_dn = Omega_dct_dn / Komega_m_dn

        #
        # Psi should tend to const at infinity, but sometimes there is
        # a linear term which indicates delta' contribution to Omega,
        # i.e. a delta-like contribution to the current
        #
        dq_up1 = self.q_up[-1] - self.q_up[-2]
        dpsi_lin_up1 = (self._psi_up[-1] - self._psi_up[-2]) / dq_up1
        dq_up0 = self.q_up[1] - self.q_up[0]
        dpsi_lin_up0 = (self._psi_up[1] - self._psi_up[0]) / dq_up0
        dq_dn1 = self.q_dn[-1] - self.q_dn[-2]
        dpsi_lin_dn1 = (self._psi_dn[-1] - self._psi_dn[-2]) / dq_dn1
        dq_dn0 = self.q_dn[1] - self.q_dn[0]
        dpsi_lin_dn0 = (self._psi_dn[1] - self._psi_dn[0]) / dq_dn0
        #dpsi_inf  = 0.5 * (dpsi_lin_up1 + dpsi_lin_up0)
        #dpsi_inf += 0.25 * (dpsi_lin_dn1 + dpsi_lin_dn0)
        #dpsi_inf, psi_inf, C, D = fit_lin_const_inv(self.q_up, self._psi_up)
        dpsi_inf, psi_inf, C, D = fit_lin_const_inv(self.q_up, Omega_dct_up)

        # Subtract the delta'-function
        self._psi_up -= dpsi_inf * self.q_up
        self._psi_dn -= dpsi_inf * self.q_dn

        # Behaviour at infinity: hard to integrate, single it out
        #psi_inf  = 0.5 * (self._psi_up[0] + self._psi_up[-1])
        print ("dpsi_inf = ", dpsi_inf, "psi_inf = ", psi_inf)
        dpsi_inf = 0.0
        psi_inf  = 0.0
        #psi_inf += 0.25 * (self._psi_dn[0] + self._psi_dn[-1]) 
        
        # Obtain chi+ above and chi- below
        self.chi_m_dn = - cauchy_integral_array(self.path_up,
                                                self._chi_up, self.q_dn)
        self.chi_p_up =   cauchy_integral_array(self.path_dn,
                                                self._chi_dn, self.q_up)
        # chi must tend to 0 when q->infty, which is likely to occur
        # automagically, so the code below may be not necessary.
        # But if chi tends to a finite limit instead, here
        # is how we subtract the relevant constant:
        #self.chi_p_up -= self.chi_inf # * 0.5
        #self.chi_m_dn += self.chi_inf # * 0.5

        # Analytically continue via difference
        self.chi_m_up = self._chi_up - self.chi_p_up
        self.chi_p_dn = self._chi_dn - self.chi_m_dn
        # Needed for flux calculations
        self.chi_m_star = - cauchy_integral_array(self.path_up,
                                                  self._chi_up, -1j * abs_k)
        #self.chi_m_star += self.chi_inf

        

        # The same with psi, but subtract the value at infinity
        self.psi_m_dn = - cauchy_integral_array(self.path_up,
                                                self._psi_up - psi_inf,
                                                self.q_dn)
        self.psi_m_dn += 0.5 * psi_inf
        self.psi_p_up =   cauchy_integral_array(self.path_dn,
                                                self._psi_dn - psi_inf,
                                                self.q_up)
        self.psi_p_up += 0.5 * psi_inf

        
        #self.psi_m_dn += dpsi_inf * self.q_dn
        #self.psi_p_up -= dpsi_inf * self.q_up
        # We need to determine the constant in the WH decomposition
        # via the consistency requirement at q = i|k|. Let us find
        # psi at this point.
        self.psi_p_star = cauchy_integral_array(self.path_up,
                                                self._psi_up - psi_inf,
                                                 1j * abs_k)
        self.psi_p_star += 0.5 * psi_inf
        #self.psi_p_star -=  dpsi_inf * 1j * abs_k
        # If we subtract psi* from psi+ and add it to psi-,
        # we obtain the solution with Omega* = 0.
        #
        # The requirement: D* = i sgn_k Omega*
        # But  D* = i J* (below)
        #
        # Hence Omega* = J* sgn_k
        #
        # psi*  = J* sgn_k / Ko*
        #
        sgn_k = np.sign(self.k)
        self.psi_p_star += - J_star * sgn_k / self.Komega_star
        self.psi_p_up -= self.psi_p_star
        #self.psi_p_dn -= self.psi_p_star
        self.psi_m_dn += self.psi_p_star
        #self.psi_m_up += self.psi_p_star

        # Once the constants are fixed, we can
        # analytically continue psi+ and psi-
        self.psi_m_up = self._psi_up - self.psi_p_up
        self.psi_p_dn = self._psi_dn - self.psi_m_dn

        # Needed psi- (-i|k|), for the flux calculation
        self.psi_m_star = - cauchy_integral_array(self.path_up,
                                                  self._psi_up - psi_inf,
                                                  -1j * abs_k)
        self.psi_m_star += 0.5 * psi_inf
        self.psi_m_star += self.psi_p_star

        # Now we can determine the vorticity and density.
        # For vorticity, the equation is particularly simple.
        # We write Omega+ and Omega- in terms of psi-
        self.Omega_p_dn = Omega_dct_dn / Komega_dn - Komega_p_dn * self.psi_m_dn
        self.Omega_p_up = Omega_dct_up / Komega_up - Komega_p_up * self.psi_m_up
        self.Omega_m_up = Komega_m_up * self.psi_m_up
        self.Omega_m_dn = Komega_m_dn * self.psi_m_dn

        #
        # The same for rho
        #
        self.rho_p_dn = rho_dct_dn / Krho_dn - Krho_p_dn * self.chi_m_dn
        self.rho_p_up = rho_dct_up / Krho_up - Krho_p_up * self.chi_m_up
        self.rho_m_up = Krho_m_up * self.chi_m_up
        self.rho_m_dn = Krho_m_dn * self.chi_m_dn
        # Handle the extra current injection term: subtract the pole
        # from rho+ and add it to rho-
        gamma1 = self.gamma1
        pole_J_dn = gamma1 / abs_k / (abs_k + 1j * self.q_dn)
        pole_J_up = gamma1 / abs_k / (abs_k + 1j * self.q_up)
        self.rho_p_dn += J_star * pole_J_dn * Krho_p_dn / self.Krho_star
        self.rho_p_up += J_star * pole_J_up * Krho_p_up / self.Krho_star
        self.rho_m_up -= J_star * pole_J_up * Krho_m_up / self.Krho_star
        self.rho_m_dn -= J_star * pole_J_dn * Krho_m_dn / self.Krho_star
        k2_dn = self.k**2 + path_dn.points()**2
        k2_up = self.k**2 + path_up.points()**2
        self.rho_p_dn += -2.0 * gamma1 / k2_dn * J_dn
        self.rho_p_up += -2.0 * gamma1 / k2_up * J_up

        #
        # Eqs for D+ and D- are solved automagically
        #
        self._Dplus_dn  =  1j * J_dn
        self._Dplus_up  =  1j * J_up
        self._Dminus_dn = -1j * self.gamma * self.rho_m_dn
        self._Dminus_up = -1j * self.gamma * self.rho_m_up

        gamma = self.gamma
        gamma_01 = gamma * gamma1
        self._flux  = gamma * self.chi_m_star / self.Krho_star
        self._flux -= J_star * gamma_01 / (2.0 * abs_k**2) / self.Krho_star**2
        self._flux -= self.psi_m_star / self.Komega_star * np.sign(self.k)
        #self._flux += J_star / self.Komega_star**2  # already in psi!
        self._flux -= flux_down
        

        
    def wall_flux(self):
        return self._flux

    def D_plus_dn(self):
        return self._Dplus_dn
    
    def D_plus_up(self):
        return self._Dplus_up

    def D_minus_dn(self):
        return self._Dminus_dn
    
    def D_minus_up(self):
        return self._Dminus_up

    def rho_plus_dn(self):
        return self.rho_p_dn
    
    def rho_plus_up(self):
        return self.rho_p_up
    
    def rho_minus_up(self):
        return self.rho_m_up
    
    def rho_minus_dn(self):
        return self.rho_m_dn

    def Omega_plus_up(self):
        return self.Omega_p_up
    
    def Omega_plus_dn(self):
        return self.Omega_p_dn
    
    def Omega_minus_up(self):
        return self.Omega_m_up
    
    
    def Omega_minus_dn(self):
        return self.Omega_m_dn

    def rho_plus(self):
        return self.rho_p_dn

    def rho_minus(self):
        return self.rho_m_dn

    def D_plus(self):
        return self._Dplus_dn
    
    def D_minus(self):
        return self._Dminus_dn

    def Omega_plus(self):
        return self.Omega_p_dn

    def Omega_minus(self):
        return self.Omega_m_dn

    def drho_plus_dn(self):
        return self.rho_p_dn - self.rho_direct(self.path_dn.points())

    
    


        
        
