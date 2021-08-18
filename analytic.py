import numpy as np
from scipy import integrate
import scipy

def get_analytic_plus_z(z, F):
    def f(x):
        return F(x) / (x - z)
    def f_re(x):
        return f(x).real
    def f_im(x):
        return f(x).imag
    if False:
       xvals = np.linspace(-100, 100, 1000)
       y_re = np.vectorize(lambda x: f_re(x))(xvals)
       y_im = np.vectorize(lambda x: f_im(x))(xvals)
       pl.figure()
       pl.plot(xvals, y_re, label='Re')
       pl.plot(xvals, y_im, label='Im')
       pl.legend()
       pl.title("z = %g + 1j %g" % (z.real, z.imag))
       pl.show()
    I_re, eps_re = integrate.quad(f_re, -scipy.infty, scipy.infty, limit=1000)
    I_im, eps_im = integrate.quad(f_im, -scipy.infty, scipy.infty, limit=1000)
    I = I_re + 1j * I_im
    return 1.0 / 2.0 / np.pi * 1j  * I

def get_analytic_minus_z(z, F):
    def f(x):
        return F(x) / (x - z)
    def f_re(x):
        return f(x).real
    def f_im(x):
        return f(x).imag
    I_re, eps_re = integrate.quad(f_re, -scipy.infty, scipy.infty, limit=1000)
    I_im, eps_im = integrate.quad(f_im, -scipy.infty, scipy.infty, limit=1000)
    I = I_re + 1j * I_im
    return - 1.0 / 2.0 / np.pi * 1j *  I

def get_analytic_minus(q, F):
    def f(x):
        if abs(x - q) < 1e-8:
           x = q + 1e-8
        return (F(x) - F(q))/(x - q)
    def f_re(x):
        return f(x).real
    def f_im(x):
        return f(x).imag
    I_re, eps_re = integrate.quad(f_re, -scipy.infty, scipy.infty, limit=1000)
    I_im, eps_im = integrate.quad(f_im, -scipy.infty, scipy.infty, limit=1000)
    I = I_re + 1j * I_im
    Fplus = 0.5 * F(q) - 1j / 2.0 / np.pi * I
    return Fplus

def get_analytic_plus(q, F):
    def f(x):
        if abs(x - q) < 1e-6:
           x = q + 1e-6
        return (F(x) - F(q))/(x - q)
    def f_re(x):
        return f(x).real
    def f_im(x):
        return f(x).imag
    I_re, eps_re = integrate.quad(f_re, -scipy.infty, scipy.infty, limit=1000)
    I_im, eps_im = integrate.quad(f_im, -scipy.infty, scipy.infty, limit=1000)
    I = I_re + 1j * I_im
    #I, eps = integrate.quad(f, -scipy.infty, scipy.infty, limit=1000)
    Fplus = 0.5 * F(q) + 1j / 2.0 / np.pi * I
    return Fplus

