import numpy as np
import frhs as frhs
from scipy import special

class Flow:
    def __init__(self, K_up, path_up, K_dn, path_dn):
        if np.abs(K_up.k - K_dn.k) > 1e-6:
            raise Exception("mismatched k")
        if np.abs(K_up.gamma - K_dn.gamma) > 1e-6:
            raise Exception("Mismatched gamma")
        if np.abs(K_up.gamma1 - K_dn.gamma1) > 1e-6:
            raise Exception("Mismatched gamma")
        self.k      = K_up.k
        self.q_up   = K_up.q
        self.q_dn   = K_dn.q

        self.K_up   = K_up
        self.K_dn   = K_dn
        self.path_up = path_up
        self.path_dn = path_dn

        self.gamma  = K_up.gamma
        self.gamma1 = K_up.gamma1

        self.Krho_p   = K_dn.rho_plus()
        self.Komega_p = K_dn.omega_plus()
        self.Krho_m   = K_up.rho_minus()
        self.Komega_m = K_up.omega_minus()

        self.Komega_star = self.K_up.omega_star()
        self.Krho_star   = self.K_up.rho_star()
        self.chi_p_up = None
        self.chi_p_dn = None
        self.chi_m_up = None
        self.chi_m_dn = None
        self.psi_p_up = None
        self.psi_p_dn = None
        self.psi_m_up = None
        self.psi_m_dn = None
        
    def wall_flux  (self): pass

    def rho_direct(self, q): pass
    def Omega_direct(self, q): pass
    def J(self, q): pass

    def D_plus_up(self):
        return self._D_plus(self.q_up)
    
    def D_minus_up(self):
        return -1j * self.gamma * self.rho_minus_up()
        #return self._D_minus(self, self.q_up)
    
    def D_plus_dn(self):
        return self._D_plus(self.q_dn)
    
    def D_minus_dn(self):
        return -1j * self.gamma * self.rho_minus_dn()
    
    #self._D_minus(self, self.q_dn)
    
    def rho_plus_up   (self):
        return self._rho_plus(self.q_up, self.K_up.rho_plus(), self.chi_p_up)
    
    def rho_plus_dn   (self):
        return self._rho_plus(self.q_dn, self.K_dn.rho_plus(), self.chi_p_dn)
    
    def rho_minus_up   (self):
        return self._rho_minus(self.q_up, self.K_up.rho_minus(), self.chi_m_up)
    
    def rho_minus_dn   (self):
        return self._rho_minus(self.q_dn, self.K_dn.rho_minus(), self.chi_m_dn)
    
    def Omega_plus_up   (self):
        return self._Omega_plus(self.q_up, self.K_up.omega_plus(),
                                self.psi_p_up)
    
    def Omega_plus_dn   (self):
        return self._Omega_plus(self.q_dn, self.K_dn.omega_plus(),
                                self.psi_p_dn)
    
    def Omega_minus_up   (self):
        return self._Omega_minus(self.q_up, self.K_up.omega_minus(),
                                 self.psi_m_up)
    
    def Omega_minus_dn   (self):
        return self._Omega_minus(self.q_dn, self.K_dn.omega_minus(),
                                 self.psi_m_dn)
    def rho_plus(self):
        return self.rho_plus_up()
    
    def rho_minus(self):
        return self.rho_minus_up()
    
    def D_plus(self):
        return self.D_plus_up()
    
    def D_minus(self):
        return self.D_minus_up()
    
    def Omega_plus(self):
        return self.Omega_plus_up()
    
    def Omega_minus(self):
        return self.Omega_minus_up()

    def j_q(self, q):
        D_q = self.D_plus()
        O_q = self.Omega_plus()
        k2 = self.k**2 + q**2
        jx_q = (D_q * self.k + Omega_q * q) / k2
        jy_q = (D_q * q - Omega_q * self.k) / k2
        return jx_q, jy_q

    #def drho_plus_up(self):
    #    return self.rho_plus_up() # - self.rho_direct(self.q_up)
    
    def drho_plus_dn(self):
        return self.rho_plus_dn() #(self.q_up, self.K_up.rho_plus(),
                              #self.chi_p_dn)  # - self.rho_direct(self.q_dn)

    def _fourier(self, path, F, y):
        q = path.points()
        exp_qy = np.exp(-1j * np.outer(y, q))
        return path.integrate_array(F * exp_qy) / 2.0 / np.pi
    
    def _rho_y(self, y):
        #y_pos = y[ y >= 0 ]
        #y_neg = y[ y < 0  ]
        #res   = 0.0 + 0.0j * y
        #res[ y >=  0 ] =
        return self._fourier(self.path_dn,
                             self.rho_plus_dn(), y)
        #res[ y < 0 ]   = self._fourier(self.path_up,
        #                               self.rho_minus_up(), y_neg)
        return res

    def _drho_y(self, y):
        #y_pos = y[ y >= 0 ]
        #y_neg = y[ y < 0  ]
        #res   = 0.0 + 0.0j * y
        #res[ y >=  0 ] =
        return self._fourier(self.path_dn,
                             self.drho_plus_dn(), y)
        #res[ y < 0 ]   =
        #return self._fourier(self.path_up,
        #                               self.rho_minus_up(), y_neg)
        #return res
        #print ("generic drho_y")
        #if y >= 0:
        #    return self._fourier(self.path_dn, self.drho_plus_dn(), y)
        #else:
        #    return self._rho_y(y)

    def _get_positive(self, func, y):
        res = 0.0 * y + 0.0j
        eps = 1e-6
        res[y < -eps] = 0.0
        y_pos = y[y > -eps]
        res[y > -eps] =  func(y_pos)  #np.vectorize(self._rho_y)(y)
        return res
    
    def rho_y(self, y):
        #res = 0.0 * y + 0.0j
        #eps = 1e-6
        #res[y < -eps] = 0.0
        #y_pos = y[y > -eps]
        #res[y > -eps] =  self._rho_y(y_pos)  #np.vectorize(self._rho_y)(y)
        return self._get_positive(self._rho_y, y)
    
    def drho_y(self, y):
        return self._get_positive(self._drho_y, y)
        #return self._drho_y(y) #np.vectorize(self._drho_y)(y)
    
    def rho_sing_y(self, y):
        return 0.0 * y + 0.0j

    def jx_q(self, D, Omega, q):
        k2 = self.k**2 + q**2
        return (D * self.k - Omega * q) / k2
    
    def jy_q(self, D, Omega, q):
        k2 = self.k**2 + q**2
        return (D * q + Omega * self.k) / k2

    def jx_plus_dn(self):
        return self.jx_q(self.D_plus_dn(), self.Omega_plus_dn(), self.q_dn)
    
    def jy_plus_dn(self):
        return self.jy_q(self.D_plus_dn(), self.Omega_plus_dn(), self.q_dn)
    
    def jx_minus_up(self):
        return self.jx_q(self.D_minus_up(), self.Omega_minus_up(), self.q_up)
    
    def jy_minus_up(self):
        return self.jy_q(self.D_minus_up(), self.Omega_minus_up(), self.q_up)

    def _jx_y(self, y):
        #res = 0.0 + 0.0j * y
        #y_neg = y[ y <  0 ]
        #y_pos = y[ y >= 0 ]
        #res[ y <  0 ] = self._fourier(self.path_up, self.jx_minus_up(), y_neg)
        #res[ y >= 0 ] =
        return  self._fourier(self.path_dn, self.jx_plus_dn(), y)
        #return res
        #if y < 0:
        #    return self._fourier(self.path_up, self.jx_minus_up(), y)
        #else:
        #    return self._fourier(self.path_dn, self.jx_plus_dn(), y)

    def _jy_y(self, y):
        #res = 0.0 + 0.0j * y
        #y_neg = y[ y <  0 ]
        #y_pos = y[ y >= 0 ]
        #res[ y <  0 ] = self._fourier(self.path_up, self.jy_minus_up(), y_neg)
        #res[ y >= 0 ] =
        return self._fourier(self.path_dn, self.jy_plus_dn(), y)
        #return res
        #if y < 0:
        #    return self._fourier(self.path_up, self.jy_minus_up(), y)
        #else:
        #    return self._fourier(self.path_dn, self.jy_plus_dn(), y)

    def jx_y(self, y):
        return self._get_positive(self._jx_y, y)
       #return self._jx_y(y) #np.vectorize(self._jx_y)(y)
   
    def jy_y(self, y):
        return self._get_positive(self._jy_y, y)
        #return self._jy_y(y) #np.vectorize(self._jy_y)(y)
      

class CombinedFlow(Flow):
    def __init__(self, K_up, path_up, K_dn, path_dn):
        self.flows = []
        Flow.__init__(self, K_up, path_up, K_dn, path_dn)
        
    def add(self, flow, weight):
        self.flows.append((flow, weight))
        
    def wall_flux(self):
        flux = 0.0 + 0.0j
        for flow, weight in self.flows:
            #print ("flux was:", flux, "add", flow.wall_flux())
            flux += flow.wall_flux() * weight
        #print ("flux now:", flux)
        return flux

    def rho_direct(self, q):
        rho = 0.0*q + 0.0j
        for flow, weight in self.flows:
            rho += flow.rho_direct(q) * weight
        return rho
            
    def Omega_direct(self, q):
        omega = 0.0*q + 0.0j
        for flow, weight in self.flows:
            omega += flow.Omega_direct(q) * weight
        return omega
    
    def J(self, q):
        j = 0.0*q + 0.0j
        for flow, weight in self.flows:
            j += flow.J(q) * weight
        return j
            
    def rho_plus_up(self):
        rho = 0.0 + 0.0j
        for flow, weight in self.flows:
            rho += flow.rho_plus_up() * weight
        return rho
    
    def rho_minus_up(self):
        rho = 0.0 + 0.0j
        for flow, weight in self.flows:
            rho += flow.rho_minus_up() * weight
        return rho
    
    def rho_plus_dn(self):
        rho = 0.0 + 0.0j
        for flow, weight in self.flows:
            rho += flow.rho_plus_dn() * weight
        return rho
    
    def rho_minus_dn(self):
        rho = 0.0 + 0.0j
        for flow, weight in self.flows:
            rho += flow.rho_minus_dn() * weight
        return rho
    
    def Omega_plus_up(self):
        omega = 0.0 + 0.0j
        for flow, weight in self.flows:
            omega += flow.Omega_plus_up() * weight
        return omega
    
    def Omega_minus_up(self):
        omega = 0.0 + 0.0j
        for flow, weight in self.flows:
            omega += flow.Omega_minus_up() * weight
        return omega
    
    def Omega_plus_dn(self):
        omega = 0.0 + 0.0j
        for flow, weight in self.flows:
            omega += flow.Omega_plus_dn() * weight
        return omega
    
    def Omega_minus_dn(self):
        omega = 0.0 + 0.0j
        for flow, weight in self.flows:
            omega += flow.Omega_minus_dn() * weight
        return omega

    
    def D_minus_up(self):
        D = 0.0 + 0.0j
        for flow, weight in self.flows:
            D += flow.D_minus_up() * weight
        return D
    
    def D_plus_up(self):
        D = 0.0 + 0.0j
        for flow, weight in self.flows:
            D += flow.D_plus_up() * weight
        return D
    
    def D_minus_dn(self):
        D = 0.0 + 0.0j
        for flow, weight in self.flows:
            D += flow.D_minus_dn() * weight
        return D
    
    def D_plus_dn(self):
        D = 0.0 + 0.0j
        for flow, weight in self.flows:
            D += flow.D_plus_dn() * weight
        return D

    def rho_y(self, y):
        rho = 0.0 + 0.0j * y
        for flow, weight in self.flows:
            rho += flow.rho_y(y) * weight
        return rho
    
    def drho_y(self, y):
        rho = 0.0 + 0.0j * y
        for flow, weight in self.flows:
            rho += flow.drho_y(y) * weight
        return rho

    def jx_y(self, y):
        jx = 0.0 + 0.0j * y
        for flow, weight in self.flows:
            jx += flow.jx_y(y) * weight
        return jx
    
    def jy_y(self, y):
        jy = 0.0 + 0.0j * y
        for flow, weight in self.flows:
            jy += flow.jy_y(y) * weight
        return jy
        



    
