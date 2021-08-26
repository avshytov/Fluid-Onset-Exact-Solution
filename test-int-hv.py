from scipy import integrate
import pylab as pl
import numpy as np

def complex_quad(f, a, b):
    def f_re(x):
        return f(x).real
    def f_im(x):
        return f(x).imag
    
    I_re, eps_re = integrate.quad(f_re, a, b)
    I_im, eps_im = integrate.quad(f_im, a, b)

    return I_re + 1j * I_im

def do_quad_up(func, k, q, gamma):
       
        def f(alpha):
            return func(alpha, k, q, gamma)
        I = complex_quad(f, 0, np.pi)
        return I / 2.0 / np.pi

def do_quad_full(func, k, q, gamma):
       
        def f(alpha):
            return func(alpha, k, q, gamma)
        I = complex_quad(f, 0, 2.0 * np.pi)
        return I / 2.0 / np.pi

def test_int(func, expr, gamma, kvals, qvals, title):
    pl.figure()
    pl.title(title)
    for k in kvals:
        Ivals  = np.vectorize(lambda q: do_quad_up(func, k, q, gamma))(qvals)
        Evals = np.vectorize(lambda q: expr(k, q, gamma))(qvals)
        pl.plot(qvals, Ivals.real, label='Re num I, k=%g' % k)
        pl.plot(qvals, Ivals.imag, label='Im num I')
        pl.plot(qvals, Evals.real, '--', label='Re expr')
        pl.plot(qvals, Evals.imag, '--', label='Im expr')
        pl.legend()

def test_int_full(func, expr, gamma, kvals, qvals, title):
    pl.figure()
    pl.title(title)
    for k in kvals:
        Ivals  = np.vectorize(lambda q: do_quad_full(func, k, q, gamma))(qvals)
        Evals = np.vectorize(lambda q: expr(k, q, gamma))(qvals)
        pl.plot(qvals, Ivals.real, label='Re num I, k=%g' % k)
        pl.plot(qvals, Ivals.imag, label='Im num I')
        pl.plot(qvals, Evals.real, '--', label='Re expr')
        pl.plot(qvals, Evals.imag, '--', label='Im expr')
        pl.legend()

def k_dot_v(alpha, k, q):
    return k * np.cos(alpha) + q * np.sin(alpha)

def k_cross_v(alpha, k, q):
    return k * np.sin(alpha) - q * np.cos(alpha)

def propagator(alpha, k, q, gamma):
    return gamma - 1j * k_dot_v(alpha, k, q)

def func_sin_x_3(alpha, k, q, gamma):
    return np.sin(alpha) * k_cross_v(alpha, k, q) / propagator(alpha, k, q, gamma)**3

def sqr(k, q, gamma):
    return np.sqrt(k**2 + q**2 + gamma**2)

def one_log(k, q, gamma):
    return 1.0 + 2.0 / np.pi * 1j * np.log((q + sqr(k, q, gamma))/sqr(k, 0, gamma))

def expr_sin_x_3a(k, q, gamma):
    return k / 4.0 * (1.0/sqr(k, q, gamma)**3 * one_log(k, q, gamma)
            + 2 * 1j / np.pi * q / sqr(k, 0, gamma)**2 / sqr(k, q, gamma)**2)

def expr_sin_x_3b(k, q, gamma):
    return k / 4.0 / sqr(k, q, gamma)**3 * (one_log(k, q, gamma)
             + 2 * 1j / np.pi * q * sqr(k, q, gamma) /(k**2 + gamma**2) )


def func_1(alpha, k, q, gamma):
    return 1.0/propagator(alpha, k, q, gamma)

def expr_1(k, q, gamma):
    return 0.5/sqr(k, q, gamma)*one_log(k, q, gamma)

def expr_sin_x_3c(k, q, gamma):
    eps = 1e-5
    return -0.5 * (expr_1 (k + eps, q, gamma) - expr_1(k - eps, q, gamma))/2.0/eps

#def expr_sin_omega(k, q, g)
def func_sin_omega(alpha, k, q, gamma):
    return (gamma**2 + k**2 + q**2) * func_sin_3(alpha, k, q, gamma) - gamma * func_sin_2(alpha, k, q, gamma)

def expr_sin_omega_a(k, q, gamma):
    return (gamma**2 + k**2 + q**2) * expr_sin_3b(k, q, gamma) - gamma * expr_sin_2b(k, q, gamma)

def expr_sin_omega_b(k, q, gamma):
    return 0.25j * gamma * q / sqr(k, q, gamma)**3 * (one_log(k, q, gamma)
            + 2j / np.pi * q / (k**2 + gamma**2) * sqr(k, q, gamma))

def func_sin_2(alpha, k, q, gamma):
    return np.sin(alpha) / propagator(alpha, k, q, gamma)**2

def expr_sin_2a(k, q, gamma):
    return 0.5j * (q/sqr(k, q, gamma)**3 * one_log(k, q, gamma)
                    - 2.0j/np.pi / sqr(k, q, gamma)**2)

def expr_sin_2b(k, q, gamma):
    eps = 1e-6
    return -1j * (expr_1(k, q + eps, gamma) - expr_1(k, q - eps, gamma))/2.0/eps

def func_sin_3(alpha, k, q, gamma):
    return np.sin(alpha) / propagator(alpha, k, q, gamma)**3

def expr_sin_3a(k, q, gamma):
    return 0.25j * gamma * (3 * q/sqr(k, q, gamma)**5 * one_log(k, q, gamma)
                    + 2j/np.pi * q**2/sqr(k, q, gamma)**4 / (k**2 + gamma**2)
                            - 4j / np.pi * 1.0/sqr(k, q, gamma)**4)

def expr_sin_3b(k, q, gamma):
    return 0.25j * gamma/sqr(k, q, gamma)**4 * (
        3 * q/sqr(k, q, gamma) * one_log(k, q, gamma)
                    + 2j/np.pi * q**2 / (k**2 + gamma**2)
                            - 4j / np.pi)

def expr_sin_3c(k, q, gamma):
    eps = 1e-5
    return -0.5 * (expr_sin_2b(k, q, gamma + eps) - expr_sin_2b(k, q, gamma - eps)) /2.0/eps

def func_x_3(alpha, k, q, gamma):
    return k_cross_v(alpha, k, q) / propagator(alpha, k, q, gamma)**3

def expr_x_3(k, q, gamma):
    return k * gamma / np.pi / (k**2 + gamma**2)**2

def func_1_2(alpha, k, q, gamma):
    return 1 / propagator(alpha, k, q, gamma)**2

def expr_1_2a(k, q, gamma):
    return gamma / 2.0 * (1.0/sqr(k, q, gamma)**3 * one_log(k, q, gamma)
                + 2j / np.pi * q / (k**2 + gamma**2) / sqr(k, q, gamma)**2)

def expr_1_2b(k, q, gamma):
    eps = 1e-5
    return - (expr_1(k, q, gamma + eps) - expr_1(k, q, gamma - eps))/2.0/eps

def func_1_3(alpha, k, q, gamma):
    return 1 / propagator(alpha, k, q, gamma)**3

def func_1_omega(alpha, k, q, gamma):
    return (gamma**2 + k**2 + q**2) / propagator(alpha, k, q, gamma)**3 - gamma / propagator(alpha, k, q, gamma)**2

def expr_1_omega_a(k, q, gamma):
    return (gamma**2 + k**2 + q**2) * expr_1_3c(k, q, gamma) \
        - gamma * expr_1_2a(k, q, gamma)

def expr_1_omega_b(k, q, gamma):
    return 0.25 * (-(k**2 + q**2)/sqr(k, q, gamma)**3 * one_log(k, q, gamma)
                   + 2j / np.pi * q  * (
                       (q**2 + k**2) * (gamma**2 - k**2) + 2 * gamma**4
                   )
                   / (k**2 + gamma**2)**2 / sqr(k, q, gamma)**2)

def expr_1_3a(k, q, gamma):
    return 0.25 * (
      (2 * gamma**2 - k**2 - q**2)/sqr(k, q, gamma)**5 * one_log(k, q, gamma)
    + 2j / np.pi * q * gamma**2 / (k**2 + gamma**2) / sqr(k, q, gamma)**4
    - 2j / np.pi * q / (k**2 + gamma**2) / sqr(k, q, gamma)**2
    + 4j * gamma**2 / np.pi / (q + 1e-6) * (1/(k**2 + gamma**2)**2
                                 - 1./sqr(k, q, gamma)**4)
    )

def expr_1_3b(k, q, gamma):
    return 0.25 * (
      (2 * gamma**2 - k**2 - q**2)/sqr(k, q, gamma)**5 * one_log(k, q, gamma)
    - 2j / np.pi * q * (k**2 + q**2)/sqr(k, q, gamma)**4 / (k**2 + gamma**2) 
    + 4j * gamma**2 / np.pi * q * (q**2 + 2.0* (k**2 + gamma**2))
         / (k**2 + gamma**2)**2 / sqr(k, q, gamma)**4)

def expr_1_3c(k, q, gamma):
    return 0.25 * (
      (2 * gamma**2 - k**2 - q**2)/sqr(k, q, gamma)**5 * one_log(k, q, gamma)
    +2j / np.pi * q * (q**2 * gamma**2 + 3 * k**2 * gamma**2 + 4 * gamma**4
                       - k**4 - q**2 * k**2)
        / (k**2 + gamma**2)**2 / sqr(k, q, gamma)**4) 
    #- 2j / np.pi * q * (k**2 + q**2)/sqr(k, q, gamma)**4 / (k**2 + gamma**2) 
    #+ 4j * gamma**2 / np.pi * q * (q**2 + 2.0* (k**2 + gamma**2))
    #     / (k**2 + gamma**2)**2 / sqr(k, q, gamma)**4)

def expr_1_3d(k, q, gamma):
    eps = 1e-5
    Ip = expr_1(k, q, gamma + eps)
    Im = expr_1(k, q, gamma - eps)
    Io = expr_1(k, q, gamma)
    return 0.5 * (Ip + Im - 2*Io)/eps**2
    
gamma = 1.2
kvals = [0.5, 1.0, 1.5]
qvals = np.linspace(0, 10.0, 201)

#test_int(func_1, expr_1, gamma, kvals, qvals, "1")
#test_int(func_sin_x_3, expr_sin_x_3a, gamma, kvals, qvals, "sin-x-3a")
#test_int(func_sin_x_3, expr_sin_x_3b, gamma, kvals, qvals, "sin-x-3b")
#test_int(func_sin_x_3, expr_sin_x_3c, gamma, kvals, qvals, "sin-x-3c")
#test_int(func_sin_2, expr_sin_2a, gamma, kvals, qvals, "sin-2a")
#test_int(func_sin_2, expr_sin_2b, gamma, kvals, qvals, "sin-2b")
#test_int(func_sin_3, expr_sin_3a, gamma, kvals, qvals, "sin-3a")
#test_int(func_sin_3, expr_sin_3b, gamma, kvals, qvals, "sin-3b")
#test_int(func_sin_3, expr_sin_3c, gamma, kvals, qvals, "sin-3c")
#test_int(func_sin_omega, expr_sin_omega_a, gamma, kvals, qvals, "sin-o-a")
#test_int(func_sin_omega, expr_sin_omega_b, gamma, kvals, qvals, "sin-o-b")
#test_int(func_x_3, expr_x_3, gamma, kvals, qvals, "x-3")
#test_int(func_1_2, expr_1_2a, gamma, kvals, qvals, "1-2a")
#test_int(func_1_2, expr_1_2b, gamma, kvals, qvals, "1-2b")
#test_int(func_1_3, expr_1_3a, gamma, kvals, qvals, "1-3a")
#test_int(func_1_3, expr_1_3b, gamma, kvals, qvals, "1-3b")
#test_int(func_1_3, expr_1_3c, gamma, kvals, qvals, "1-3c")
#test_int(func_1_3, expr_1_3d, gamma, kvals, qvals, "1-3d")
#test_int(func_1_omega, expr_1_omega_a, gamma, kvals, qvals, "1-o-a")
#test_int(func_1_omega, expr_1_omega_b, gamma, kvals, qvals, "1-o-b")
def src_I(alpha, k, q, gamma):
    return 2.0 / propagator(alpha, k, q, gamma)

def src_s(alpha, k, q, gamma):
    return np.sin(alpha) / propagator(alpha, k, q, gamma)

def make_rho(src):
    def f(alpha, k, q, gamma):
        return k_cross_v(alpha, k, q) / propagator(alpha, k, q, gamma)**2 * src(alpha, k, q, gamma)
    return f

def make_omega(src):
    def f(alpha, k, q, gamma):
        s = src(alpha, k, q, gamma)
        gk = gamma**2 + k**2 + q**2
        p1 = gk / propagator(alpha, k, q, gamma)**2
        p2 = gamma / propagator(alpha, k, q, gamma)**1
        return s * (p1 - p2)
    return f

def expr_F_rho_I_up(k, q, gamma):
    return k * gamma / (k**2 + gamma**2)**2 * 2.0 / np.pi

def expr_F_rho_I_full(k, q, gamma):
    return 0.0

def expr_F_omega_I_full(k, q, gamma):
    k2 = k**2 + q**2
    kqgamma = np.sqrt(k2 + gamma**2)
    return -k2 / kqgamma**3 

def expr_F_omega_I_up(k, q, gamma):
    k2 = k**2 + q**2
    kqgamma = np.sqrt(k2 + gamma**2)
    kgamma = np.sqrt(k**2 + gamma**2)
    log_q = np.log((q + kqgamma) / kgamma)
    f1 = (1.0 + 2.0j / np.pi * log_q) * k2 / kqgamma
    f2a = (k2 * (gamma**2 - k**2) + 2.0 * gamma**4) / kgamma**4
    f2 = 2.0j * q / np.pi * f2a
    return -(f1 - f2) / 2.0 / kqgamma**2 

def expr_F_rho_s_full(k, q, gamma):
    k2 = k**2 + q**2
    kqgamma = np.sqrt(k2 + gamma**2)
    return k / 2.0 / kqgamma**3
    #kgamma = np.sqrt(k**2 + gamma**2)
    #log_q = np.log((q + kqgamma) / kgamma)
    #return -1j / np.pi * k / kqgamma**3 * (log_q + q * kqgamma / kgamma**2)

def expr_F_rho_s_up(k, q, gamma):
    k2 = q**2 + k**2
    kqgamma = np.sqrt(k2 + gamma**2)
    kgamma = np.sqrt(k**2 + gamma**2)
    log_q = np.log((q + kqgamma) / kgamma)
    f2 = 2.0j / np.pi * q * kqgamma / kgamma**2
    return k / 4.0 / kqgamma**3 * (1.0 + 2j / np.pi * log_q + f2)

def expr_F_omega_s_up(k, q, gamma):
    k2 = q**2 + k**2
    kqgamma = np.sqrt(k2 + gamma**2)
    kgamma = np.sqrt(k**2 + gamma**2)
    log_q = np.log((q + kqgamma) / kgamma)
    f2 = 2.0j / np.pi * q * kqgamma / kgamma**2
    return 1j * q * gamma/ 4.0 / kqgamma**3 * (1.0 + 2j / np.pi * log_q + f2)

def expr_F_omega_s_full(k, q, gamma):
    k2 = k**2 +q**2
    kqgamma = np.sqrt(k2 + gamma**2)
    return 0.5 * 1j * q * gamma / kqgamma**3
    

#test_int(make_rho(src_I), expr_F_rho_I_up, gamma, kvals, qvals,
#         "I-rho-up")
#test_int_full(make_rho(src_I), expr_F_rho_I_full, gamma, kvals, qvals,
#              "I-rho-full")
#test_int(make_omega(src_I), expr_F_omega_I_up, gamma, kvals, qvals,
#         "I-omega-up")
#test_int_full(make_omega(src_I), expr_F_omega_I_full, gamma, kvals, qvals,
#              "s-omega-full")
#test_int(make_rho(src_s), expr_F_rho_s_up, gamma, kvals, qvals,
#         "s-rho-up")
#test_int_full(make_rho(src_s), expr_F_rho_s_full, gamma, kvals, qvals,
#              "s-rho-full")
#test_int(make_omega(src_s), expr_F_omega_s_up, gamma, kvals, qvals,
#         "s-rho-up")
#test_int_full(make_omega(src_s), expr_F_omega_s_full, gamma, kvals, qvals,
#              "s-rho-full")

def src_o(alpha, k, q, gamma):
    return 2 * k_cross_v(alpha, k, q) / (k**2 + q**2)  / propagator(alpha, k, q, gamma)

def src_rho(alpha, k, q, gamma):
    return 1.0 / propagator(alpha, k, q, gamma)

def zero(k, q, gamma): return 0.0

def drho_dct(k, q, gamma): # correct
    return 1.0 / (k**2 + q**2 + gamma**2)**1.5

def dO_dct(k, q, gamma): 
    return - 0.5 * (k**2 + q**2) / (k**2 + q**2 + gamma**2)**1.5

#correct belo
test_int_full(make_rho(src_o), drho_dct, gamma, kvals, qvals, "drho-dct")
# wrong
test_int_full(make_omega(src_rho), dO_dct, gamma, kvals, qvals, "dO-dct")
test_int_full(make_rho(src_rho), zero, gamma, kvals, qvals, "zero in drho")
test_int_full(make_omega(src_o), zero, gamma, kvals, qvals, "zero in dO")

pl.show()
