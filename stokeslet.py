import numpy as np
from scipy import special
from flows import Flow
from cauchy import cauchy_integral, cauchy_integral_array

class Stokeslet(Flow):
    def __init__ (self, Fx, Fy, h, k, K_up, path_up, K_dn, path_dn):
        Flow.__init__(self, k, K_up, path_up, K_dn, path_dn)
        #self.K = K
        self.h = h
        self.Fx = Fx
        self.Fy = Fy
        self.exp_kh = np.exp(-np.abs(k) * h)
        #self.path_up = path_up
        #self.path_dn = path_dn
        self.psi_p_up = self.psi_plus(self.q_up)
        #self.path_up.eval(self.psi_plus)
        self.psi_m_dn = self.psi_minus(self.q_dn)
        #self.path_dn.eval(self.psi_minus)
        self.psi_p_dn = self.psi_dn() - self.psi_m_dn
        self.psi_m_up = self.psi_up() - self.psi_p_up

    def J(self, q):
        return 0.0 * q + 0.0j

    def rho_direct(self, q):
        k2 = self.k**2 + q**2
        rho =  2.0j * (self.k * self.Fx + q * self.Fy) / k2
        rho *= np.exp(1j * q * self.h)
        rho *= self.K_up.rho(q)
        return rho
    
    def Omega_direct(self, q):
        kF = self.k * self.Fy - q * self.Fx
        K = self.K_up.omega(q)
        omega = kF / self.gamma1 * (1.0 - K) * np.exp(1j * q * self.h)
        return omega

    def psi(self, Kminus, q):
        Kinv = 1.0/Kminus - 1.0
        kF = (self.k * self.Fy - q * self.Fx) 
        exp_qh = np.exp(1j * q * self.h)
        return kF * exp_qh * Kinv / self.gamma1

    def psi_up(self):
        return self.psi(self.K_up.omega_minus(), self.q_up)
    
    def psi_dn(self):
        return self.psi(self.K_dn.omega_minus(), self.q_dn)

    def psi_minus(self, z):
        z1 = self.path_up.points()
        abs_k = np.abs(self.k)
        ###z_z1 = np.outer(z, 1.0 + 0.0*z1) - np.outer(1.0 + 0.0*z, z1)
        ###abs_kz1 = np.outer(1.0 + 0.0 * z, 1j * abs_k - z1)
        ###f_psi = self.psi_up() * (1.0/z_z1 - 1.0/abs_kz1)
        psi_m  = -cauchy_integral_array(self.path_up, self.psi_up(), z)
        psi_m +=  cauchy_integral_array(self.path_up, self.psi_up(),
                                        1j * abs_k)
        #f_psi = self.psi_up() * (1.0/(z - z1) - 1.0/(1j * abs_k - z1))
        ###psi_m = self.path_up.integrate_array(f_psi) / 2.0 / np.pi / 1j
        psi_m -= self.psi_inf()
        return psi_m #- self.psi_inf()

    def psi_plus(self, z):
        z1 = self.path_dn.points()
        abs_k = np.abs(self.k)
        ###z_z1 = np.outer(z, 1.0 + 0.0*z1) - np.outer(1.0 + 0.0*z, z1)
        ###abs_kz1 = np.outer(1.0 + 0.0 * z, 1j * abs_k - z1)
        #print ("shapes:", np.shape(z_z1), np.shape(abs_kz1),
        #       type(z_z1), type)
        ###f_psi = - self.psi_dn() * (1.0/z_z1 - 1.0/abs_kz1)
        #f_psi = - self.psi_dn() * (1.0/(z - z1) - 1.0/(1j * abs_k - z1))
        ###psi_p =  self.path_dn.integrate_array(f_psi) / 2.0 / np.pi / 1j
        psi_p  = cauchy_integral_array(self.path_dn, self.psi_dn(), z)
        psi_p -= cauchy_integral_array(self.path_dn, self.psi_dn(), 1j*abs_k) 
        psi_p += self.psi_inf()
        return psi_p
    
    def psi_inf (self):
        sgn_k = np.sign(self.k)
        abs_k = np.abs(self.k)
        kF_star = (self.k * self.Fy - 1j * abs_k * self.Fx)
        C = - kF_star / self.gamma1 * (1.0 - 1.0/self.Komega_star)
        return C * self.exp_kh
    
    def psi_star(self):
        psi_m = self.psi_minus(-1j * np.abs(self.k))
        return psi_m #- self.psi_inf()
    
    def psi_star1(self):
        z1 = self.path_up.points()
        abs_k = np.abs(self.k)
        z_z1 = -1j * abs_k - z1
        #z_z1 = np.outer(z, 1.0 + 0.0*z1) - np.outer(1.0 + 0.0*z, z1)
        #abs_kz1 = np.outer(1.0 + 0.0 * z, 1j * abs_k - z1)
        abs_kz1 = 1j * abs_k - z1
        #print ("shapes:", np.shape(z_z1), np.shape(abs_kz1),
        #       type(z_z1), type)
        f_psi =  self.psi_up() * (1.0/z_z1 - 1.0/abs_kz1)
        if False:
            I = self.path_up.integrate_array(f_psi)
            print ("integral: ", I)
            import pylab as pl
            pl.figure()
            #cs = np.cumsum(f_psi)
            x_arc = self.path_up.arc_lengths()
            x_arc -= x_arc[-1]/2.0
            f_mid = (f_psi[1:] + f_psi[:-1])/2.0
            z_p = self.path_up.points()
            dz = z_p[1:] - z_p[:-1]
            fdz = f_mid * dz / np.abs(dz)
            cs = np.cumsum(fdz * np.abs(dz)) 
            x_mid = (x_arc[1:] + x_arc[:-1])/2
            pl.plot(x_mid, fdz.real, label='Re F dz')
            pl.plot(x_mid, fdz.imag, label='Im F dz')
            pl.plot(x_mid, cs.real, label='Re CS')
            pl.plot(x_mid, cs.imag, label='Im CS')
            pl.plot(x_arc, 0*x_arc + I.real, 'k--')
            pl.xlim(-1.5, 1.5)
            pl.ylim(-1.0, 6.2)
            pl.legend()
            pl.figure()
            pl.plot(x_arc, self.path_up.points().real, label='Re z(s)')
            pl.xlim(-1.5, 1.5)
            pl.ylim(-0.8, 0.8)
            pl.figure()
            pl.plot(x_arc, self.path_up.points().imag, label='Im z(s)')
            pl.xlim(-1.5, 1.5)
            pl.ylim(0, 1.5)
            pl.legend()
            pl.figure()
            Komega_p = self.K_up.Komega_p
            def Ko_func(q):
                return self.K_up.K.omega_plus(self.k, q)
            Komega_p2 = np.vectorize(Ko_func) (self.path_up.points())
            pl.plot(x_arc, Komega_p.real, label='Re K')
            pl.plot(x_arc, Komega_p.imag, label='Im K')
            pl.plot(x_arc, Komega_p2.real, label='Re K2')
            pl.plot(x_arc, Komega_p2.imag, label='Im K2')
            pl.ylim(-1.0, 1.0)
            pl.legend()
            pl.xlim(-1.5, 1.5)
            pl.show()
        #f_psi = - self.psi_dn() * (1.0/(z - z1) - 1.0/(1j * abs_k - z1))
        psi_m =  self.path_up.integrate_array(f_psi) / 2.0 / np.pi / 1j
        psi_m -= self.psi_inf()
        return psi_m
   
    def wall_flux(self):
        k = self.k
        gamma = self.gamma
        gamma1 = self.gamma1
        exp_kh =  self.exp_kh
        abs_k = np.abs(k)
        sgn_k = np.sign(k)
        Fk = 1j * self.Fx * sgn_k - self.Fy
        flux  = gamma / 2.0 / abs_k * Fk / self.Krho_star**2 * exp_kh
        #print ("first term in flux:", flux)
        #flux += 1.0/self.Komega_star * self.psi_inf() 
        #print ("first two terms in flux", flux)
        flux += - 1.0/self.Komega_star * self.psi_star() * np.sign(k)
        #print ("correction: ", self.psi_star() / self.Komega_star)
        #print ("correction: ", self.psi_star1() / self.Komega_star)
        #print ("total: ", flux)
        #self.psi_minus(-1j * abs_k)
        #flux  = - gamma * gamma1 / 2.0 / k**2 / self.Krho_star**2 * exp_kh
        #flux += exp_kh / self.Komega_star**2 
        #flux += self.chi_minus(-1j * k) / self.Krho_star * gamma
        return flux

    #
    # Below we separate rho_plus into three contributions:
    # the regular one, the singularity ~ Fx, the singularity ~ Fy
    #
    def _rho_plus_sing_x(self, q, Krho_p):
        F_kq   = self.k * self.Fx
        k2 = self.k**2 + q**2
        rho = 2j * F_kq / k2 
        return rho
    
    def _rho_plus_sing_y(self, q, Krho_p):
        F_kq   = q * self.Fy
        k2 = self.k**2 + q**2
        rho = 2j * F_kq / k2
        return rho
    
    def _rho_plus_reg(self, q, Krho_p, chi_p):
        gamma  = self.gamma
        gamma1 = self.gamma1
        k = self.k
        k2 = k**2 + q**2
        exp_kh = self.exp_kh
        #exp_qh = np.exp(1j * q * self.h)
        abs_k = np.abs(self.k)
        Krho_star = self.Krho_star
        sgn_k = np.sign(self.k)
        #F_kq   = self.k * self.Fx + q * self.Fy
        F_sgnk = 1j * sgn_k * self.Fx - self.Fy
        #rho = 2j * F_kq / k2 * exp_qh
        rho = - F_sgnk / (abs_k + 1j * q) * Krho_p / Krho_star * exp_kh
        return rho
    
    def _rho_plus(self, q, Krho_p, chi_p):
        rho  =  self._rho_plus_reg(q, Krho_p, chi_p)
        exp_qh = np.exp(1j * q * self.h)
        rho +=  self._rho_plus_sing_x(q, Krho_p) * exp_qh
        rho +=  self._rho_plus_sing_y(q, Krho_p) * exp_qh
        return rho

    def _rho_minus(self, q, Krho_m, chi_m):
        gamma = self.gamma
        gamma1 = self.gamma1
        k = self.k
        Krho_star = self.Krho_star
        exp_kh = self.exp_kh
        abs_k = np.abs(k)
        sgn_k = np.sign(k)
        F_sgnk = 1j * sgn_k * self.Fx - self.Fy
        rho = F_sgnk / (abs_k + 1j * q) * Krho_m / Krho_star * exp_kh
        return rho
    
    def _D_plus(self, q):
        return 0.0 * q + 0.0j

    #
    # The singularity in Omega is handled similar to the one in edge.py:
    # we separate all the oscillating terms, writing
    #
    # Ko_p psi_p = Ko_p * (psi - psi_m)
    #
    # Since psi ~ [kxF] (1/Ko_m - 1) * exp(iqh), together with the term
    # (Ko_p - 1) * exp(iqh), we obtain
    #
    # omega_p = [kxF] / gamma (1/Ko - 1) * exp(i q h) - Ko_p * psi_m
    #
    
    def _Omega_plus_reg(self, q, Ko_p, psi_m):
        Ko_star = self.Komega_star
        sgn_k = np.sign(self.k)
        abs_k = np.abs(self.k)
        kF_star = (self.k * self.Fy - 1j * abs_k * self.Fx)
        #C = - kF_star / self.gamma * (1.0 - 1.0/self.Komega_star)
        # the constant is needed to make O* = 0
        return - Ko_p * psi_m  #+ #self.psi_inf() * Ko_p * self.exp_kh
        
    def _Omega_plus_sing_x(self, q, Ko):
        kF = (self.k * self.Fy * 0 - q * self.Fx) 
        return (1.0/Ko - 1.0) * kF / self.gamma1
    
    def _Omega_plus_sing_y(self, q, Ko):
        kF = (self.k * self.Fy * 1 - 0 * q * self.Fx) 
        return (1.0/Ko - 1.0) * kF / self.gamma1
        
    def _Omega_plus(self, q, Ko_p, psi_p):
        #return self._Omega_plus_old(q, Ko_p, psi_p)
        sgn_k = np.sign(self.k)
        #kF = (self.k * self.Fy - q * self.Fx) / self.gamma
        exp_qh = np.exp(1j * q * self.h)
        Ko = self.K_dn.omega(q)
        Ko_m = Ko * Ko_p
        psi = self.psi(Ko_m, q)
        psi_m = psi - psi_p
        omega = self._Omega_plus_reg(q, Ko_p, psi_m)
        omega += self._Omega_plus_sing_x(q, Ko) * exp_qh
        omega += self._Omega_plus_sing_y(q, Ko) * exp_qh
        return omega

    def _Omega_plus_old(self, q, Ko_p, psi_p):
        sgn_k = np.sign(self.k)
        kF = (self.k * self.Fy - q * self.Fx) / self.gamma1
        exp_qh = np.exp(1j * q * self.h)
        return Ko_p * psi_p + kF * exp_qh * (Ko_p - 1.0)
    
    def _Omega_minus(self, q, Ko_m, psi_m):
        sgn_k = np.sign(self.k)
        abs_k = np.abs(self.k)
        # Offsets the constant in psi_plus_reg
        #kF_star = (self.k * self.Fy - 1j * abs_k * self.Fx)
        #C = - kF_star / self.gamma * (1.0 - 1.0/self.Komega_star)
        return Ko_m * psi_m
        #- self.psi_inf() * Ko_m * self.exp_kh

    def _rho_sing_x_dn(self):
        return self._rho_plus_sing_x(self.q_dn, self.K_dn.rho_plus())

    def _rho_sing_y_dn(self):
        return self._rho_plus_sing_y(self.q_dn, self.K_dn.rho_plus())
    
    def _rho_plus_reg_dn(self):
        return self._rho_plus_reg(self.q_dn,
                                  self.K_dn.rho_plus(), self.chi_p_dn)
    
    def _rho_y(self, y):
        #res = 0.0 + 0.0j * y
        #y_neg = y[ y < 0  ]
        #y_pos = y[ y >= 0 ]
        #print ("stokeslet rho_y")
        #res[ y < 0 ] = self._fourier(self.path_up, self.rho_minus(), y_neg) 
        #if y < 0:
        #    return self._fourier(self.path_up, self.rho_minus(), y)
        yh = np.abs(y - self.h) # even/odd integrand
        sgn_yh = np.sign(yh)
        rho_sing_x = self._fourier(self.path_dn, self._rho_sing_x_dn(),
                                   yh)
        rho_sing_y = self._fourier(self.path_dn, self._rho_sing_y_dn(),
                                   yh) * sgn_yh
        #if y < self.h: # the y contribution is odd, the x contrib is even
        #    rho_sing_y *= -1
        rho_reg = self._fourier(self.path_dn, self._rho_plus_reg_dn(), y)
        #res[y >= 0] =
        return rho_sing_x + rho_sing_y + rho_reg
        #return res

    def _drho_y(self, y):
        return self._rho_y(y)
    
    #def _drho_y(self, y): return 0.0 * y
    def rho_sing(self, y):
        return 0.0 * y + 0.0j

    def _omega_reg(self):
        return self._Omega_plus_reg(self.q_dn,self.K_dn.omega_plus(),
                                    self.psi_m_dn)
    def _omega_sing_x(self):
        return self._Omega_plus_sing_x(self.q_dn,
                                       self.K_dn.omega(self.q_dn))

    def _omega_sing_y(self):
        return self._Omega_plus_sing_y(self.q_dn,
                                       self.K_dn.omega(self.q_dn))
        
    def _jx_y(self, y):
        #res = 0.0 * y + 0.0j
        #y_pos = y[ y >= 0 ]
        #y_neg = y[ y <  0 ]
        #res[ y < 0 ] = self._fourier(self.path_up, self.jx_minus_up(), y_neg)
        yh = y - self.h
        abs_yh = np.abs(yh)
        sgn_yh = np.sign(yh)

        jx_reg    = self.jx_q( 0.0 * self.q_dn, self._omega_reg(), self.q_dn)
        jx_sing_x = self.jx_q( 0.0 * self.q_dn, self._omega_sing_x(),
                               self.q_dn)
        jx_sing_y = self.jx_q( 0.0 * self.q_dn, self._omega_sing_y(),
                               self.q_dn)

        jx_pos = self._fourier(self.path_dn,  jx_reg, y)
        jx_pos += self._fourier(self.path_dn, jx_sing_x, abs_yh) 
        jx_pos += self._fourier(self.path_dn, jx_sing_y, abs_yh) * sgn_yh
        return jx_pos
        #res[ y >= 0 ] = jx_pos
        #return res
        
    def _jy_y(self, y):
        #res = 0.0 * y + 0.0j
        #y_pos = y[ y >= 0 ]
        #y_neg = y[ y <  0 ]
        #res[ y < 0 ] = self._fourier(self.path_up, self.jy_minus_up(), y_neg)
        yh = y - self.h
        abs_yh = np.abs(yh)
        sgn_yh = np.sign(yh)

        jy_reg    = self.jy_q( 0.0 * self.q_dn, self._omega_reg(), self.q_dn)
        jy_sing_x = self.jy_q( 0.0 * self.q_dn, self._omega_sing_x(),
                               self.q_dn)
        jy_sing_y = self.jy_q( 0.0 * self.q_dn, self._omega_sing_y(),
                               self.q_dn)

        jy_pos  = self._fourier(self.path_dn, jy_reg,    y)
        jy_pos += self._fourier(self.path_dn, jy_sing_x, abs_yh) * sgn_yh
        jy_pos += self._fourier(self.path_dn, jy_sing_y, abs_yh) 
        #res[ y>= 0 ] = jy_pos
        #return res
        return jy_pos

        return 0.0 + 0.0j * y
    #def _jy_y(self, y): return 0.0 + 0.0j * y
