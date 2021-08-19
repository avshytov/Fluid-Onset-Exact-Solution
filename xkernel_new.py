from analytic import get_analytic_plus, get_analytic_minus
from analytic import get_analytic_plus_z, get_analytic_minus_z
import numpy as np
from scipy import integrate, linalg
import scipy

#
# Helper function:
# The square root with properly defined branches
#
def sqrt_z_old(gamma, k, q):
    kappa = np.sqrt(k * k + gamma * gamma)
    sq = np.sqrt(abs( kappa * kappa + q * q)) + 0.0j
    sq *= np.exp(0.5j * (np.angle(kappa - 1j * q)
                         + np.angle(kappa + 1j * q)))
    return sq

def sqrt_z(gamma, k, q):
    kappa = np.sqrt(k * k + gamma * gamma)
    return np.sqrt((kappa - 1j * q) * (kappa + 1j * q))


class WHKernel:
    def __init__ (self, gamma, a):
        self.gamma = gamma
        self.a = a
        print ("kernel: ", gamma, a)


    #
    # The kernel itself. It has a zero at
    # q = +-i kappa_a , with kappa_a = sqrt(k^2 + gamma^2 - a^2)
    #
    def __call__(self, k, q):
        return 1.0 - self.a  / sqrt_z(self.gamma, k, q)

    def log_prime(self, k, q):
        sq = sqrt_z(self.gamma, k, q)
        return self.a * q * (a + sq) / (sq**2 * (sq**2 - self.a**2)) 

    #
    #  The factor Kplus(q) is regular in the upper half-plane
    #
    def plus(self, k, q):
        if q.imag < 0:
            return self.minus(k, q) / self(k, q) 
        def lnX(x):
            return np.log(self(k, x))
        return np.exp(get_analytic_plus_z(q, lnX))
    
    def plus_prime(self, k, q):
        if q.imag < 0:
            return self.minus_prime(k, q) -  self.log_prime(k, q)
        def lnX_prime(x):
            return self.log_prime(k, x)
        return np.exp(get_analytic_plus_z(q, lnX_prime))
    

    #
    #  The factor Kminus(q) is regular in the lower half-plane
    #
    def minus(self, k, q):
        if q.imag > 0:
            return self.plus(k, q) * self(k, q)
        def lnX(x):
            return  np.log(self(k, x))
        return np.exp(-get_analytic_minus_z(q, lnX))

    def minus_prime(self, k, q):
        if q.imag > 0:
            return self.plus_prime(k, q) +  self.log_prime(k, q)
        def lnX_prime(x):
            return self.log_prime(k, x)
        return np.exp(-get_analytic_minus_z(q, lnX_prime))
    
    def pole(self, k):
        return np.sqrt(k**2 + self.gamma**2 - self.a**2)
    #
    # The value of Kplus at the pole q = i kappa_a
    #
    def star(self, k):
        def f(x):
            #atan = np.arctan(x)
            #xi = self.a / self.gamma
            #asin = np.asin(xi * x / np.sqrt(1.0 + x**2))
            #sq = np.sqrt(1.0 + x**2 * (1 - xi**2))
            #return asin / sq / x
            return np.arctan(x) / x
        kappa   = np.sqrt(k**2 + self.gamma**2)
        kappa_a = np.sqrt(k**2 + self.gamma**2 - self.a**2)
        #x0 = self.a / np.sqrt(k**2 + self.gamma**2 - self.a**2)
        x_a = self.a / kappa_a
        # Use the identity I(x_a) = pi/2 log(x_a) + I(1/x_a)
        # to select the smaller integration domain
        if abs(x_a) < 1:
           I, eps = integrate.quad(f, 0, x_a)
        else:
           I, eps = integrate.quad(f, 0, 1.0 / x_a)
           I += 0.5 * np.pi * np.log(x_a)
        #abs_k = np.abs(k)
        log_Kstar = 0.5 * np.log((kappa_a + kappa) / 2.0 / kappa_a) + I / np.pi
        return np.exp(log_Kstar)

    #
    # The derivative of log Kplus at the pole
    #
    def prime(self, k):
        kappa = np.sqrt(k**2 + self.gamma**2)
        kappa_a = np.sqrt(kappa**2 - self.a**2)
        result  =  - self.a**2 / 4.0 / kappa_a / (kappa + kappa_a)**2
        resumt +=  0.5/np.pi/self.a
        asin = np.arcsin (self.a/kappa)
        result += - kappa**2 / 2.0 / np.pi / self.a**2 / kappa_a * asin
        return result

    # The residue of 1/Kminus at the pole
    def res_minus(self, k):
        return -1j * self.a**2 / kappa_a / self.star(k)

    # The regular part of 1/Kminus at the pole
    def reg_minus(self, k):
        kappa_a = np.sqrt(k**2 + self.gamma**2 - self.a**2)
        return 1j * self.prime(k) * self.a**2 / self.star(k) / kappa_a
    
class WHKernels:
    def __init__(self, gamma, gamma1):
        self.gamma  = gamma
        self.gamma1 = gamma1
        self.gamma2 = self.gamma - self.gamma1
        self.Krho = WHKernel(gamma, self.gamma)
        self.Ko   = WHKernel(gamma, self.gamma1 - self.gamma2)

    # The density kernel
    def rho(self, k, q):
        return self.Krho(k, q)

    def omega(self, k, q):
        kappa2 = self.gamma**2 + k**2
        return self.Krho(k, q) * self.Ko(k, q) * (q**2 + kappa2)/(q**2 + k**2)

    def rho_plus(self, k, q):
        return self.Krho.plus(k, q)

    def rho_minus(self, k, q):
        return self.Krho.minus(k, q)

    def omega_plus(self, k, q):
        kappa = np.sqrt(self.gamma**2 + k**2)
        Krho_plus = self.Krho.plus(k, q)
        Ko_plus   = self.Ko.plus  (k, q)
        fact = (q + 1j * np.abs(k)) / (q + 1j * np.abs(kappa))
        return Krho_plus * Ko_plus * fact
    
    def omega_minus(self, k, q):
        kappa = np.sqrt(self.gamma**2 + k**2)
        Krho_minus = self.Krho.plus(k, q)
        Ko_minus = self.Ko.minus(k, q)
        fact = (q - 1j * np.abs(k)) / (q - 1j * np.abs(kappa))
        return Krho_minus * Ko_minus / fact

    def rho_star(self, k):
        return self.Krho.star(k)
    
    def omega_star(self, k):
        kappa = np.sqrt(self.gamma**2 + k**2)
        abs_k = np.abs(k)
        Krho_star = self.Krho.star(k)
        Ko_star = self.Ko.star(k)
        return Krho_star * Ko_star * 2.0 * abs_k / (abs_k + kappa)

    #def rho_prime(self, k):
    #    return self.Krho.prime(k)

    def rho_reg(self, k):
        return self.Krho.reg(k)

    def rho_residue(self, k):
        return self.Krho.residue(k)

    #def omega_prime(self, k):
    #    kappa_o = self.Ko.pole(k)
    #    Krho_pole = self.Krho.plus(1j * kappa_o)
    #    abs_k = np.abs(k)
    #    kappa = np.sqrt(k**2 + self.gamma**2)
    #    res_o = self.Ko.residue(k)
    #    res = (pole_o - abs_k)/(pole_o - kappa) / Krho_pole * res_o
    #    pass

    def omega_residue(self, k):
        kappa_o = self.Ko.pole(k)
        Krho_pole = self.Krho.plus(1j * kappa_o)
        abs_k = np.abs(k)
        kappa = np.sqrt(k**2 + self.gamma**2)
        res_o = self.Ko.residue(k)
        res = (pole_o - abs_k)/(pole_o - kappa) / Krho_pole * res_o
        return res

    def omega_reg(self, k):
        kappa_o = self.Ko.pole(k)
        Krho_pole = self.Krho.plus(1j * kappa_o)
        Krho_prime = self.Krho.plus_prime(1j * kappa_o) / 1j
        abs_k = np.abs(k)
        kappa = np.sqrt(k**2 + self.gamma**2)
        reg_o = self.Ko.reg(k)
        res_o = self.Ko.residue(k)
        reg_corr = reg_o - Krho_prime * res_o
        reg = (pole_o - abs_k)/(pole_o - kappa) / Krho_pole * reg_corr
        return reg

    #
    # For two merging poles
    #
    def omega_prime2(self, k):
        if np.abs(self.gamma2) > 0.001 * self.gamma:
          raise Exception("omega_prime for distinct poles not implemented yet")
        abs_k = np.abs(k)
        kappa = np.sqrt(abs_k**2 + self.gamma**2)
        pole_rho = self.Krho.pole(k)
        pole_o   = self.Ko.pole(o)
        result  = self.Krho.prime(k)
        result += self.Ko.prime(k)
        # The factors (q + ik)/(q + i kappa)
        result += 1.0/(pole_o + abs_k)
        result += -1.0/(pole_o + kappa)
        return result

    def omega_reg2(self, k):
        if np.abs(self.gamma2) > 0.001 * self.gamma:
          raise Exception("omega_prime for distinct poles not implemented yet")
        abs_k = np.abs(k)
        kappa = np.sqrt(abs_k**2 + self.gamma**2)
        pole_rho = self.Krho.pole(k)
        pole_o   = self.Ko.pole(o)
        res  = self.Krho.residue(k) * self.Ko.reg(k)
        res += self.Ko.residue(k)   * self.Krho.reg(k)
        res *= 1.0/(pole_o - kappa)
        return result

    def omega_resiude2(self, k):
        if np.abs(self.gamma2) > 0.001 * self.gamma:
          raise Exception("omega_prime for distinct poles not implemented yet")
        abs_k = np.abs(k)
        kappa = np.sqrt(abs_k**2 + self.gamma**2)
        pole_rho = self.Krho.pole(k)
        pole_o   = self.Ko.pole(o)
        res = 1.0 / (pole_o - kappa) * self.Ko.residue(k)
        res *= self.Krho.residue(k)
        return res

class TabulatedKernels:
    def __init__(self, K, k, q, Krho_p, Komega_p):
        self.k        = k
        self.gamma    = K.gamma
        self.gamma1   = K.gamma1
        self.q        = q
        self.Krho_p   = Krho_p
        self.Komega_p = Komega_p
        self.K = K
        self.Krho        = K.rho(self.k, self.q)
        self.Komega      = K.omega(self.k, self.q)
        self.Krho_star   = K.rho_star(k)
        self.Komega_star = K.omega_star(k)
        
    def rho(self, q):
        return self.K.rho(self.k, q)
    
    def omega(self, q):
        return self.K.omega(self.k, q)
    
    def rho_plus(self):
        return self.Krho_p

    def omega_plus(self):
        return self.Komega_p

    def rho_star(self):
        return self.Krho_star

    def omega_star(self):
        return self.Komega_star

    def omega_minus(self):
        return self.Komega * self.Komega_p
    
    def rho_minus(self):
        return self.Krho   * self.Krho_p


def tabulate_kernel(K, k, q, tabulate_omega = False):
    # Tabulate Krho_plus
    print ("Tabulate Krho, ", len(q))
    Krho_p   = np.vectorize( lambda z: K.rho_plus(k, z))(q)
    # In the ohmic case, tabulate Komega_plus as well
    if abs(K.gamma - K.gamma1) > 0.001 * abs(K.gamma) or tabulate_omega:
       print ("tabulate Komega_plus")
       Komega_p = np.vectorize( lambda z: K.omega_plus(k, z))(q)
    else: # economize on the relation between Krho and Komega otherwise
       print ("skip tabulation of Komega_plus")
       kappa = np.sqrt(k**2 + q**2 + K.gamma**2)
       abs_k = np.abs(k)
       q_fact = (q + 1j * abs_k) / (q + 1j * kappa)
       Komega_p = Krho_p**2 * q_fact
    return TabulatedKernels(K, k, q, Krho_p, Komega_p)

def load_kernel(K, k, q, suffix="", tabulate_omega = False):
    
    def fname():
        fname_tmpl =  "xkernel-%s-k=%g-gam=%g-gam1=%g.npz"
        filename = fname_tmpl % (suffix, k,
                                 K.gamma, K.gamma1)
        return filename
    
    def load():
        d = np.load(fname())

        if abs(d['k'] - k) > 1e-6:
            raise Exception("k does not match")
        if len(d['q']) != len(q):
            raise Exception("len(q) does not match")
        if linalg.norm(d['q'] - q) > 1e-6:
            raise Exception("q does not match")
        if abs(d['gamma'] - K.gamma) > 1e-6:
            raise Exception("gamma does not match")
        if abs(d['gamma1'] - K.gamma1) > 1e-6:
            raise Exception("gamma does not match")

        Krho_p = d['Krho_p']
        Komega_p = d['Komega_p']
        #self.Krho = d['Krho']
        #self.Komega = d['Komega']
        #self.Krho_star   = d['Krho_star']
        #self.Komega_star = d['Komega_star']
        tabK = TabulatedKernels(K, k, q, Krho_p, Komega_p)
        return tabK
        
    def save(tabK):
        np.savez(fname(),
                 k=k,
                 gamma=K.gamma, gamma1=K.gamma1,
                 q=q, 
                 Krho_p = tabK.Krho_p, Komega_p = tabK.Komega_p,
                 Krho = tabK.Krho, Komega = tabK.Komega,
                 Krho_star = tabK.Krho_star,
                 Komega_star = tabK.Komega_star)
    
    always_tabulate = False
    if always_tabulate:
        tabK = tabulate_kernel(K, k, q, tabulate_omega)
        return tabK

    try:
        tabK = load()
    except:
        import traceback
        traceback.print_exc()
        tabK = tabulate_kernel(K, k, q, tabulate_omega)
        save(tabK)
    print ("Krho* = ", tabK.Krho_star, "Ko* = ", tabK.Komega_star)
    return tabK    

if __name__ == '__main__':
    from xkernel_old import Xomega_plus_z, Xrho_plus_z, Xrho_minus_z, \
         Xomega_minus_z, Xrho_star

    q = np.linspace(-10.0, 10.0, 501)
    s = np.linspace(-10.0, 10.0, 501)
    #kvals = [0.5, 1.0, 1.5]
    kvals = [1.2]
    gamma  = 1.0
    gamma1 = 1.0
    K = WHKernels(gamma, gamma1)
    for k in kvals:
        im_q = 0.01j 
        print("make tabulated K")
        K_kq = tabulate_kernel(K, k, q + im_q)#TabulatedKernels(K, k, q + im_q)
        K_ks = tabulate_kernel(K, k, 1j * s)  #TabulatedKernels(K, k, 1j * s)
        print ("tabulate old K: rho")
        Xrho_plus_old_q = np.vectorize(lambda z: Xrho_plus_z(k, z))(-q - im_q)
        Krho_plus_new_q = K_kq.rho_plus()
        Xrho_plus_old_s = np.vectorize(lambda z: Xrho_plus_z(k, z))(-1j * s)
        Krho_plus_new_s = K_ks.rho_plus()
        Xomega_plus_old_q = np.vectorize(lambda z:
                                          Xomega_plus_z(k, z))(-q - im_q)
        Xomega_plus_old_s = np.vectorize(lambda z:
                                          Xomega_plus_z(k, z))(-1j * s)
        Komega_plus_new_q = K_kq.omega_plus()
        Komega_plus_new_s = K_ks.omega_plus()
        import pylab as pl
        pl.figure()
        pl.plot(q, Xrho_plus_old_q.real, label='Re old Krho+ (k,q)')
        pl.plot(q, Xrho_plus_old_q.imag, label='Im old Krho+(k, q)')
        pl.plot(q, Krho_plus_new_q.real, '--', label='Re new Krho+(k, q)')
        pl.plot(q, Krho_plus_new_q.imag, '--', label='Im new Krho+(k, q)')
        pl.plot(q, Xomega_plus_old_q.real, label='Re old Ko+ (k,q)')
        pl.plot(q, Xomega_plus_old_q.imag, label='Im old Ko+(k, q)')
        pl.plot(q, Komega_plus_new_q.real, '--', label='Re new Ko+(k, q)')
        pl.plot(q, Komega_plus_new_q.imag, '--', label='Im new Ko+(k, q)')
        pl.legend()
        pl.xlabel("q")
        pl.title("Krho(k, q) @ k = %g" % k)
        pl.figure()
        pl.plot(s, Xrho_plus_old_s.real, label='Re old Krho+ (k, is)')
        pl.plot(s, Xrho_plus_old_s.imag, label='Im old Krho+ (k, is)')
        pl.plot(s, Krho_plus_new_s.real, '--', label='Re new Krho+(k, is)')
        pl.plot(s, Krho_plus_new_s.imag, '--', label='Im new Krho+(k, is)')
        pl.plot(s, Xomega_plus_old_s.real, label='Re old Ko+ (k, is)')
        pl.plot(s, Xomega_plus_old_s.imag, label='Im old Ko+ (k, is)')
        pl.plot(s, Komega_plus_new_s.real, '--', label='Re new Ko+(k, is)')
        pl.plot(s, Komega_plus_new_s.imag, '--', label='Im new Ko+(k, is)')
        pl.legend()
        pl.xlabel("s")
        pl.title("Krho(k, is) @ k = %g" % k)
    pl.show()
        
    
if False:
    k0 = 1.0
    pl.figure()
    pl.title("Re Xo")
    for im_q in [-5.0, -2.0, -1.5, -0.99, -0.5, 0.0, 0.5, 0.99, 1.5, 2.0, 5.0]:
        q = 1j * im_q + np.linspace(-10.0, 10.0, 1001)
        Xo_q = np.vectorize(lambda z: Xomega_z(k0, z))(q)
        pl.plot(q.real, Xo_q.real, label='Im q = %g' % im_q)
    pl.legend()
    pl.figure()
    pl.title("Im Xo")
    for im_q in [-5.0, -2.0, -1.5, -0.99, -0.5, 0.0, 0.5, 0.99, 1.5, 2.0, 5.0]:
        q = 1j * im_q + np.linspace(-10.0, 10.0, 1001)
        Xo_q = np.vectorize(lambda z: Xomega_z(k0, z))(q)
        pl.plot(q.real, Xo_q.imag, label='Im q = %g' % im_q)
    pl.legend()
    pl.show()


if False:
    pl.figure()
    yvals = np.linspace(-10.0, 10.0, 1000)
    for k in [0.1, 0.3, 1.0, 3.0]:
       print ("******** k = ", k)
       Xrho_im = np.vectorize(lambda y: Xrho_plus_z(k, 1j * y))(yvals)
       pl.plot(yvals, Xrho_im.real, label='Re @ %g' % k)
       pl.plot(yvals, Xrho_im.imag, label='Im @ %g' % k)
    pl.show()
if False:
  pl.figure()
  for k in [0.01, 0.1, 0.5, 1.0, 3.0, 10.0]:
    #qvals = np.linspace(-10.0, 10.0, 1001)
    #Xvals = np.vectorize(lambda qt: Xrho_minus(k, qt))(q)
    Xrho_vals = np.vectorize(lambda qt: Xrho(k, qt))(q)
    #lnX = np.log(Xrho_vals)
    #lnX_m = 0.5 * lnX - 0.5j * np.dot(H, lnX)
    #Xrho_m = np.exp(lnX_m)
    Xrho_m = np.exp(analytic_minus(np.log(Xrho_vals)))

    pl.figure()
    pl.title("X @ k = %g" % k)
    #pl.plot(q, Xvals.real, label='Re')
    #pl.plot(q, Xvals.imag, label='Im')
    pl.plot(q, Xrho_m.real, label='Re H*X ..')
    pl.plot(q, Xrho_m.imag, label='Im H*X ..')
    pl.legend()
    #np.savez("Xvals-k=%g" % k, q=q, X=Xvals)
  pl.show()


