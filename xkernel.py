from analytic import get_analytic_plus, get_analytic_minus
from analytic import get_analytic_plus_z, get_analytic_minus_z
import numpy as np
from scipy import integrate
import scipy

gamma = 1.0

def Xrho(k, q):
    return 1.0 - gamma / np.sqrt(gamma**2 + k**2 + q**2)

def sqrt_z(k, q):
    sq = np.sqrt(np.abs(gamma**2 + k**2 + q**2))
    kappa = np.sqrt(k**2 + gamma**2)
    sq *= np.exp(0.5j * (np.angle(kappa - 1j * q) + np.angle(kappa + 1j * q)))
    return sq

def Xrho_z(k, q):
    return 1.0 - gamma / sqrt_z(k, q)


def Xomega(k, q):
    sq = np.sqrt(gamma**2 + k**2 + q**2)
    absk2 = q**2 + k**2
    return (sq - gamma)**2 / absk2

def Xomega_z(k, q):
    return (sqrt_z(k, q) - gamma)**2 / (k**2 + q**2)

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

def Xrho_minus(k, q):
    def lnX(x):
        return np.log(Xrho(k, x))
    #def f(x):
    #    return (lnX(x) - lnX(q))/(x - q)
    #I, eps = integrate.quad(f, -scipy.infty, scipy.infty, limit=1000)
    #lnXq = 0.5 * lnX(q) - 1j / 2.0 / np.pi * I
    return np.exp(get_analytic_minus(q, lnX))

def Xrho_plus(k, q):
    def lnX(x):
        return np.log(Xrho(k, x))
    return np.exp(-get_analytic_plus(q, lnX))

def Xrho_plus_z(k, q):
    if q.imag > 0:
       return Xrho_minus_z(k, q) / Xrho_z(k, q)
    def lnX(x):
        return np.log(Xrho(k, x))
    return np.exp(-get_analytic_plus_z(q, lnX))

def Xrho_minus_z(k, q):
    if q.imag < 0:
        return Xrho_z(k, q) * Xrho_plus_z(k, q)
    def lnX(x):
        return np.log(Xrho(k, x))
    return np.exp(get_analytic_minus_z(q, lnX))



def Xomega_minus(k, q):
    def lnX(x):
        return np.log(omega(k, x))
    #def f(x):
    #    return (lnX(x) - lnX(q))/(x - q)
    #I, eps = integrate.quad(f, -scipy.infty, scipy.infty, limit=1000)
    #lnXq = 0.5 * lnX(q) - 1j / 2.0 / np.pi * I
    return np.exp(get_analytic_minus(q, lnX))

def Xomega_plus(k, q):
    def lnX(x):
        return np.log(Xomega(k, x))
    return np.exp(-get_analytic_plus(q, lnX))

def Xomega_plus_z(k, q):
    if q.imag > 0:
       return Xomega_minus_z(k, q) / Xomega_z(k, q)
    def lnX(x):
        return np.log(Xomega(k, x))
    return np.exp(-get_analytic_plus_z(q, lnX))

def Xomega_minus_z(k, q):
    if q.imag < 0:
        return Xomega_z(k, q) * Xomega_plus_z(k, q)
    A = 0.0
    kappa = np.sqrt(k**2 + gamma**2)
    def lnX(x):
        return np.log(Xomega(k, x)) - A * (np.log(k**2 + x**2) - np.log(x**2 + kappa**2))
    xminus = get_analytic_minus_z(q, lnX)
    return np.exp(xminus + A * np.log((abs(k) - 1j * q)/(kappa - 1j * q)))


if False:
    pl.figure()
    yvals = np.linspace(-10.0, 10.0, 1000)
    for k in [0.1, 0.3, 1.0, 3.0]:
       print("******** k = ", k)
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

def Xrho_star(k):
    def f(x):
        return np.arctan(x) / x
    I, eps = integrate.quad(f, 0.0, gamma/np.abs(k), limit=500)
    lnXstar = I/np.pi + 0.5 * np.log((np.abs(k) + np.sqrt(k**2 + gamma**2))/2.0/np.abs(k))
    return np.exp(lnXstar)

def Xomega_star(k):
    kappa = np.sqrt(k**2 + gamma**2)
    return Xrho_star(k)**2 * (2.0 * k / (k + kappa)) 

def Xomega_prime(k):
    res = - 0.5 / absk + 1.0/ np.pi / absk * np.arctan (absk/gamma)
    res += - absk * 0.5 / gamma**2 + k / np.pi /gamma**2 * np.arctan(absk/gamma)
    res += 1.0/np.pi/gamma
    return res

def Xomega_reg(k, Xo_0 = None):
    if not Xo_0:
       Xo_0 = Xomega_star(k)
    return gamma**2 / k**2 / Xo_0 + 2.0/Xo_0 + 2*gamma**2/k * Xomega_prime(k) / Xo_0
    
    
