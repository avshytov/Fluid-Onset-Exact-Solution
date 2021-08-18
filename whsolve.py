import numpy as np
import pylab as pl
from scipy import integrate
import scipy
import random
from scipy import linalg

gamma = 1.0

def im_plus(q, a):
    u = np.zeros((len(q)), dtype=complex)
    print("do implus", a)
    for i in range(len(q) - 1):
        q1 = q[i]
        q2 = q[i + 1]
        Ia = 0.5 * np.log((q2**2 + a**2)/(q1**2 + a**2))
        Ia += - 1j*(np.arctan(q2/a) - np.arctan(q1/a))
        
        Ib = (q2 - q1)
        Ib += - 0.5*(q1 + 1j*a)* np.log( (q2**2 + a**2) / (q1**2 + a**2) ) 
        Ib += (1j * q1 - a) * (np.arctan(q2/a) - np.arctan(q1/a))

        h = q2 - q1

        u[i    ] += (Ia - Ib / h)
        u[i + 1] += Ib / h
    q0 = q[-1]
    Iinf =  1.0 / a * np.arctan(a/q0) - 1j / a * np.log(np.sqrt(q0**2 + a**2)/np.abs(q0))
    u[-1] += Iinf * q0
    q0 = -q[0]
    Iinf =  1.0 / a * np.arctan(a/q0) + 1j / a * np.log(np.sqrt(q0**2 + a**2)/np.abs(q0))    
    u[0] += Iinf * (-q0)
    return u / 2.0 / np.pi *  1j

if False:
   xt = np.linspace(-1.0, 1.0, 1001)
   qt = 50.0 * (xt * np.abs(xt))
   #qt = np.linspace(-50.0, 50.0, 1001)
   #f1 = 1.0 / (qt - 1j)
   def f1(q):
       return 1.0 / (q - 1j)**2
   f1t = f1(qt)
   for a in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
       u = im_plus(qt, a)
       f_true = f1(-1j*a)
       f_eval = np.dot(u, f1t)
       err = f_eval - f_true
       print("a = ", a, "true val", f_true, "found:", f_eval, "err = ", err)
       pl.figure()
       pl.plot(qt, u.real, label='Re')
       pl.plot(qt, u.imag, label='Im')
       pl.title("@ a = %g" % a)
   pl.show()   

def Fourier(q, x):
    F = np.zeros((len(x), len(q)), dtype=complex)
    for i in range(len(q) - 1):
        q1 = q[i]
        q2 = q[i + 1]
        e1 = np.exp(1j * q1 * x)
        e2 = np.exp(1j * q2 * x)

        h = q2 - q1
        Ia = (e2 - e1)/(1j * x)
        Ib = ((q2 - q1) * e2  - Ia)/(1j * x)
        F[:,   i] += Ia - Ib/h
        F[:, i+1] += Ib/h
        
    F /= 2.0 * np.pi
    return F

if False:
    #qa = np.linspace(-10.0, 10.0, 1001)
    ta = np.linspace(-1.0, 1.0, 1001)
    qa = 10.0 * ta * np.abs(ta)
    xv = np.linspace(-10.0, 10.0, 500)
    fq1 = np.exp(-qa**2)
    F = Fourier(qa, xv)
    fx1 = np.dot(F, fq1)
    ft1 = np.exp(-xv**2 / 4.0) * np.sqrt(np.pi)/2.0/np.pi
    fq2 = np.exp(-np.abs(qa))
    fx2 = np.dot(F, fq2)
    ft2 = 1.0 / np.pi / (xv**2 + 1.0)
    pl.figure()
    pl.plot(xv, fx1, 'r-')
    pl.plot(xv, ft1, 'k--')
    pl.plot(xv, fx2, 'g-')
    pl.plot(xv, ft2, 'm--')
    pl.show()

def Hilbert(q):
    N = len(q)
    H = np.zeros((N, N), dtype=complex)
    for i in range(0, N):
        qi = q[i]
        for j in range(0, i - 1):
            q1 = q[j]
            q2 = q[j + 1]
            h = q2 - q1
            Ia = np.log((q2 - qi)/(q1 - qi))
            #Ib = (1.0 + Ia) * (qi - q1)
            #H[i, j]     += Ia - Ib / h
            #H[i, j + 1] += Ib / h
            H[i, j]     += Ia * (1.0  - (qi - q1)/h) - 1.0 #Ia - Ib / h
            H[i, j + 1] += Ia * (qi - q1)/h + 1.0 #(qi - q1)/h#Ib / h
        for j in range(i + 1, N - 1):
            q1 = q[j]
            q2 = q[j + 1]
            h = q2 - q1
            Ia = np.log((q2 - qi)/(q1 - qi))
            #Ib = 1.0 #(1.0 + Ia) * (qi - q1)
            H[i, j]     += Ia * (1.0  - (qi - q1)/h) - 1.0 #Ia - Ib / h
            H[i, j + 1] += Ia * (qi - q1)/h + 1.0 #(qi - q1)/h#Ib / h
        if i > 0:
           q1 = q[i - 1]
           q2 = qi
           H[i, i] += 1.0
           H[i, i - 1] -= 1.0
        if i < N - 1:
           q1 = q[i]
           q2 = q[i + 1]
           H[i, i] += -1.0
           H[i, i + 1] += 1.0
        if i < N - 1: # integrate to +inf
           q0 = q[-1]
           #print qi, q0
           if np.abs(qi) > 1e-6:
              Ia = 1.0 / qi * np.log(np.abs(q0 / (q0 - qi)))
           else:
              Ia = 1.0 / q0; 
           H[i, -1] += Ia * q0
        if i > 0:
           q0 = -q[0]
           #print
           if np.abs(qi) > 1e-6: 
              Ia = 1.0 / qi * np.log(np.abs( (q0 + qi)/q0 ))
           else:
              Ia = 1.0 / q0
           H[i, 0] += Ia * (-q0)
        H[i, i] -= np.sum(H[i, :])
        if i % 100 == 0: print(i, np.sum(H[i, :]))
        
    H *= 1.0 / np.pi
    return H

if __name__ == '__main__':
  try:
    d = np.load('Hilbert.npz')
    q = d['q']
    H = d['H']
  except:
    import traceback
    traceback.print_exc()
    #q = np.linspace(-100.0, 100.0, 10001)
    #q = np.linspace(-30.0, 30.0, 10001)
    #q = np.linspace(-30.0, 30.0, 5001)
    xx = np.linspace(-1.0, 1.0, 15001)
    q = xx * np.abs(xx) * 100.0
    H = Hilbert(q)
    np.savez("Hilbert", q=q, H=H)
if False:
    y1 = 1.0 / (1.0 + q**2)
    o = np.ones((len(q)))
    print("ones: ", linalg.norm(np.dot(H, o)[1:-1]))
    pl.figure()
    pl.plot(np.dot(H, o))
    print(np.shape(H), np.shape(y1))
    y2 = np.dot(H, y1)
    #y2 -= (y2[-1] + y2[0])/2.0
    y3 = - np.dot(H, y2)
    y2t = -q / (1.0 + q**2)
    y3x = - np.dot(H, y2t)
    pl.figure()
    pl.plot(q, y1, label='y1')
    pl.plot(q, y2, label='y2 = H y1')
    pl.plot(q, y2t, 'k--')
    pl.plot(q, y3, 'r--', label='y3 = -H y2')
    pl.plot(q, y3x, 'g--', label='y3 = -H y2t')
    pl.legend()
    pl.figure()
    pl.plot(q, y2 - y2t, label='y2 err')
    pl.plot(q, y3 - y1,  label='y3 err')
    pl.plot(q, y3 - y3x,  label='y3 err with y2t')
    pl.legend()
    pl.show()

def analytic_minus(F):
    return 0.5 * F - 0.5j * np.dot(H, F)

def analytic_plus(F):
    return 0.5 * F + 0.5j * np.dot(H, F)

def get_analytic_minus(q, F):
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

def Xrho(k, q):
    return 1.0 - gamma / np.sqrt(gamma**2 + k**2 + q**2)

def Xomega(k, q):
    sq = np.sqrt(gamma**2 + k**2 + q**2)
    absk2 = q**2 + k**2
    return (sq - gamma)**2 / absk2 

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

def Frho_direct(k, q):
    def f(theta):
        sn = np.sin(theta)
        cs = np.cos(theta)
        k_v = k * cs + q * sn
        return sn / (gamma + 1j * k_v)        
    def f_re(theta):
        return f(theta).real
    def f_im(theta):
        return f(theta).imag
    I_re, eps_re = integrate.quad(f_re, 0.0, 1.0 * np.pi, limit=1000)
    I_im, eps_im = integrate.quad(f_im, 0.0, 1.0 * np.pi, limit=1000)
    return (I_re + 1j * I_im)

def Fomega_direct(k, q):
    def f(theta):
        sn = np.sin(theta)
        cs = np.cos(theta)
        k_v = k * cs + q * sn
        return sn * (k * sn - q * cs) / (gamma + 1j * k_v)
    def f_re(theta):
        return f(theta).real
    def f_im(theta):
        return f(theta).imag
    I_re, eps_re = integrate.quad(f_re, 0.0, 1.0 * np.pi, limit=1000)
    I_im, eps_im = integrate.quad(f_im, 0.0, 1.0 * np.pi, limit=1000)
    return (I_re + 1j * I_im)

def Fm(k, q):
    sq = np.sqrt(gamma**2 + k**2 + q**2)
    sq0 = np.sqrt(k*k + gamma * gamma)
    return (np.pi - 2.0j * np.log((q + sq)/sq0))/sq


def Frho_analytic(k, q):
    sq = np.sqrt(q**2 + k**2 + gamma**2)
    return 1.0/(k**2 + q**2) * (2.0 * k * np.arctan(k/gamma) - 1j * np.pi * q + 1j * gamma * q * Fm(k, q))
    #return 4.0 / (k**2 + q**2)  * (k * np.arctan(k/gamma) + q * gamma / 2.0 / sq * np.log((q + sq) / (sq - q)))


    
    
def phi_rho_plus(k, q):
    print(k, q)
    def f(x):
        return Frho_analytic(k, x) / Xrho_minus(k, x)
    return get_analytic_plus(q, f)

def phi_rho_minus(k, q):
    def f(x):
        return Frho_analytic(k, x) / Xrho_minus(k, x)
    return get_analytic_minus(q, f)

if False:
  for k in [0.01, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]:
    print("k = ", k)
    #qvals = np.linspace(-10.0, 10.0, 501)
    #phi_p_vals = np.vectorize(lambda q: phi_rho_plus(k, q))(qvals)
    Frho_vals = np.vectorize(lambda qt: Frho_analytic(k, qt))(q)
    Xrho_vals = np.vectorize(lambda qt: Xrho(k, qt))(q)
    #lnX = np.log(Xrho_vals)
    #lnX_m = 0.5 * lnX - 0.5j * np.dot(H, lnX)
    #Xrho_m = np.exp(lnX_m)
    Xrho_m = np.exp(analytic_minus(np.log(Xrho_vals)))
    Xrho_p = np.exp(analytic_plus(-np.log(Xrho_vals)))
    phi_p_vals = analytic_plus(Frho_vals / Xrho_m)
    phi_m_vals = analytic_minus(Frho_vals / Xrho_m)
    rho_p_vals = Xrho_p * phi_p_vals
    rho_m_vals = Xrho_m * phi_m_vals
    pl.figure()
    pl.title("phi+ @ k = %g" % k)
    pl.plot(q, phi_p_vals.real, label='Re phi+')
    pl.plot(q, phi_p_vals.imag, label='Im phi+')
    pl.plot(q, rho_p_vals.real, label='Re rho+')
    pl.plot(q, rho_p_vals.imag, label='Im rho+')
    pl.legend()
    y = np.linspace(-5.0, 5.0, 1001)
    dq = q[1] - q[0]
    rho_y = np.dot(np.exp(1j * np.outer(y, q)), rho_p_vals) * dq / 2.0 / np.pi
    rho_y_m = np.dot(np.exp(1j * np.outer(y, q)), rho_m_vals) * dq / 2.0 / np.pi
    pl.figure()
    pl.plot(y, rho_y.real, label='Re+')
    pl.plot(y, rho_y.imag, label='Im+')
    pl.plot(y, rho_y_m.real, label='Re-')
    pl.plot(y, rho_y_m.imag, label='Im-')
    pl.legend()
    pl.title("rho(y) @ k = %g" % k)
    np.savez("phi_p_vals-k=%g" % k, q=q, phi_p=phi_p_vals, rho_p = rho_p_vals, y=y, rho_y=rho_y)
  pl.show()  


def Fomega_analytic(k, q):
    fm = Fm(k, q)
    absk2 = k**2 + q**2
    #sq = np.sqrt(k**2 + q**2 + gamma**2)
    f1 = k * (1 + gamma**2 / absk2) * fm
    f2 = - np.pi * gamma * k / absk2
    f3 = + 2.0 * 1j * gamma * q / absk2 * np.arctan(k/gamma)
    return f1 + f2 + f3

#def phi_rho_minus(k, q):

if False:
  for k in [0.0, 0.1, -0.1, 0.2, -0.2, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0]:
    pl.figure()
    pl.title("k = %g" % k)
    qvals = np.linspace(-10.0, 10.0, 1000)
    #yvals = np.vectorize(lambda k: Frho_direct(k, q))(kvals)
    #y0vals = np.vectorize(lambda k: Frho_analytic(k, q))(kvals)
    yvals = np.vectorize(lambda qt: Fomega_direct(k, qt))(qvals)
    y0vals = np.vectorize(lambda qt: Fomega_analytic(k, qt))(qvals)
    pl.plot(qvals, yvals.real, 'r-', label='Re I')
    pl.plot(qvals, y0vals.real, 'b--', label='Re F')
    pl.plot(qvals, yvals.imag, 'g-', label='Im I')
    pl.plot(qvals, y0vals.imag, 'k--', label='Im F')
    pl.legend()
  pl.show()


def wh_solve(k):

    im_p = im_plus(q, np.abs(k))
    
    Xrho_vals   = np.vectorize(lambda qt: Xrho(k, qt))(q)
    Xomega_vals = np.vectorize(lambda qt: Xomega(k, qt))(q)

    Xrho_m = np.exp(analytic_minus(np.log(Xrho_vals)))
    Xrho_p = np.exp(analytic_plus(-np.log(Xrho_vals)))

    Xomega_m = np.exp(analytic_minus(np.log(Xomega_vals)))
    Xomega_p = np.exp(analytic_plus(-np.log(Xomega_vals)))
    
    absk2 = k**2 + q**2

    Frho_vals   = np.vectorize(lambda qt: Frho_analytic(k, qt))(q) / 2.0 / np.pi
    Fomega_vals = np.vectorize(lambda qt: Fomega_analytic(k, qt))(q) / 2.0 / np.pi
    Fj = -2.0 * gamma / absk2 * Xrho_vals
    if False:
        pl.figure()
        pl.title("Xrho")
        pl.plot(q, Xrho_vals.real, label='Re Xo')
        pl.plot(q, Xrho_vals.imag, label='Im Xo')
        pl.plot(q, Xrho_m.real, label='Re X-')
        pl.plot(q, Xrho_m.imag, label='Im X-')
        pl.plot(q, Xrho_p.real, label='Re X+')
        pl.plot(q, Xrho_p.imag, label='Im X+')
        pl.legend()
        pl.figure()
        pl.title("Xo")
        pl.plot(q, Xomega_vals.real, label='Re Xo')
        pl.plot(q, Xomega_vals.imag, label='Im Xo')
        pl.plot(q, Xomega_m.real, label='Re X-')
        pl.plot(q, Xomega_m.imag, label='Im X-')
        pl.plot(q, Xomega_p.real, label='Re X+')
        pl.plot(q, Xomega_p.imag, label='Im X+')
        #pl.plot(q, (Xomega_m/Xomega_p).real, 'r--', label='Re X-/X+')
        #pl.plot(q, (Xomega_m/Xomega_p).imag, 'k--', label='Im X-/X+')
        pl.legend()

        pl.figure()
        pl.plot(q, Fomega_vals.real, label='Re Fo')
        pl.plot(q, Fomega_vals.imag, label='Im Fo')
        pl.plot(q, Frho_vals.real, label='Re Frho')
        pl.plot(q, Frho_vals.imag, label='Im Frho')

        pl.legend()
        #pl.show()
    
    phi = Frho_vals / Xrho_m 
    chi = Fj / Xrho_m 
    psi = Fomega_vals / Xomega_m
    
    phi_p = analytic_plus(phi)
    chi_p = analytic_plus(chi)
    psi_p = analytic_plus(psi)
    psi_p0 = np.dot(im_p, psi_p)
    psi_p -= psi_p0

    phi_m = analytic_minus(phi)
    chi_m = analytic_minus(chi)
    psi_m = analytic_minus(psi)



    Xrho_0 = np.exp(np.dot(im_p, np.log(Xrho_p)))
    print("Xrho_0 = ", Xrho_0)
    i_q0 = np.argmin(np.abs(q))
    print("Xrho_p @ ", q[i_q0], " = ", Xrho_p[i_q0])
    # testing analytical expression for chi_p1
    # unfortunately, the coefficient in front of the pole
    # is not always done properly # 0.984
    chi_p1 = 1.0 / absk2 / Xrho_p - 0.9925*1j / 2.0 / abs(k)/(q + 1j * abs(k))/Xrho_0
    chi_p1 *= -2.0 * gamma 
    
    if False:
        pl.figure()
        pl.plot(q, chi_p.real, label='Re chi+')
        pl.plot(q, chi_p.imag, label='Im chi+')
        pl.plot(q, chi_p1.real,'r--', label='Re chi1')
        pl.plot(q, chi_p1.imag,'g--', label='Im chi1')
        pl.plot(q, chi_p1.real - chi_p.real, label='Re chi - chi1')
        pl.plot(q, chi_p1.imag - chi_p.imag, label='Im chi - chi1')

        pl.legend()
        pl.show()
    if False:
        #pl.figure()
        #pl.plot(q, phi_p.real + phi_m.real, label='phi+ + phi-')
        #pl.plot(q, phi.real, label='phi')
        #pl.legend()
        pl.figure()
        pl.title("phi, chi, psi + ")
        pl.plot(q, phi_p.real, label='Re phi p')
        pl.plot(q, phi_p.imag, label='Im phi p')
        pl.plot(q, chi_p.real, label='Re chi p')
        pl.plot(q, chi_p.imag, label='Im chi p')
        pl.plot(q, psi_p.real, label='Re psi p')
        pl.plot(q, psi_p.imag, label='Im psi p')
        #pl.plot(q, phi_p.real + phi_m.real, label='phi+ + phi-')
        #pl.plot(q, phi_p.real, label='phi')
        pl.legend()
        pl.figure()
        pl.title("phi, chi, psi - ")
        pl.plot(q, phi_m.real, label='Re phi -')
        pl.plot(q, phi_m.imag, label='Im phi -')
        pl.plot(q, chi_m.real, label='Re chi -')
        pl.plot(q, chi_m.imag, label='Im chi -')
        pl.plot(q, psi_m.real, label='Re psi -')
        pl.plot(q, psi_m.imag, label='Im psi -')
        pl.legend()
        
    rho_f = phi_p * Xrho_p
    rho_j = chi_p * Xrho_p
    rho_f_full = phi_p * Xrho_p + phi_m * Xrho_m
    rho_j_full = chi_p * Xrho_p + chi_m * Xrho_m
    omega_p = Xomega_p * psi_p
    omega_f = omega_p
    Xo0 = np.exp(np.dot(im_p, np.log(Xomega_p)))
    sgn_k = 0.0
    if k < 0:
       sgn_k = -1.0
    if k > 0:
       sgn_k = 1.0
    omega_j = Xomega_p / Xo0 * sgn_k
    omega_full = Xomega_p * psi_p + Xomega_m * psi_m
    
    if False:
        pl.figure()
        pl.plot(q, rho_f.real, label='Re rho_f')
        pl.plot(q, rho_f.imag, label='Im rho_f')
        pl.plot(q, rho_j.real, label='Re rho_j')
        pl.plot(q, rho_j.imag, label='Im rho_j')
        pl.legend()
        
    Frho1   = np.vectorize(lambda qt: Frho_analytic(k, -qt))(q) / 2.0 / np.pi
    Fomega1 = np.vectorize(lambda qt: Fomega_analytic(k, -qt))(q) / 2.0 / np.pi

    flux_f = rho_f * gamma * Frho1 + omega_p * 2.0 * gamma / absk2 * Fomega1
    flux_j = rho_j * gamma * Frho1 - 2.0 * gamma / absk2 * 2.0 / (2.0 * np.pi)
    flux_j += 2.0 * gamma**2 / absk2 * Frho1
    flux_j += - omega_j * 2.0 * gamma / absk2 * Fomega1

    yv = np.linspace(-10.0, 50.0, 2001)
    dq = q[1] - q[0]
    i0 = np.argmin(np.abs(yv))
    fourier = Fourier(q, yv)
    #fourier = np.exp(1j * np.outer(yv, q)) * dq / 2.0 / np.pi
    flux_f_y = np.dot(fourier, flux_f)
    flux_j_y = np.dot(fourier, flux_j)
    rho_f_y = np.dot(fourier, rho_f)
    rho_j_y = np.dot(fourier, rho_j)
    omega_f_y = np.dot(fourier, omega_f)
    omega_j_y = np.dot(fourier, omega_j)

    rho_f_y_full = np.dot(fourier, rho_f_full)
    rho_j_y_full = np.dot(fourier, rho_j_full)
    omega_f_y_full = np.dot(fourier, omega_full)
    omega_j_y_full = np.dot(fourier, omega_j)

    if False:
        pl.figure()
        phi_y = np.dot(fourier, phi)
        chi_y = np.dot(fourier, chi)
        chi_y_p = np.dot(fourier, chi_p)
        chi_y_m = np.dot(fourier, chi_m)
        psi_y = np.dot(fourier, psi)
        
        pl.figure()
        pl.title("phi(y), chi(y), psi(y)")
        pl.plot(yv, phi_y.real, label='Re phi')
        #pl.plot(yv, phi_y.imag, label='Im phi p')
        pl.plot(yv, chi_y.real, label='Re chi')
        #pl.plot(yv, chi_y.imag, label='Im chi p')
        pl.plot(yv, psi_y.real, label='Re psi')
        #pl.plot(yv, psi_y.imag, label='Im psi p')
        pl.plot(yv, chi_y_p.real, label='Re chi+')
        pl.plot(yv, chi_y_m.real, label='Re chi-')
        pl.legend()
    
    if  False:
        pl.figure()
        pl.title(" @ k = %g" % k)
        pl.plot(yv, flux_f_y.real, label='Re Flux f ')
        pl.plot(yv, flux_f_y.imag, label='Im Flux f ')
        pl.plot(yv, flux_j_y.real, label='Re Flux j ')
        pl.plot(yv, flux_j_y.imag, label='Im Flux j ')
        pl.legend()
        pl.figure()
        pl.plot(yv, rho_f_y.real, label='Re rho f')
        pl.plot(yv, rho_f_y.imag, label='Im rho f')
        pl.plot(yv, rho_j_y.real, label='Re rho j')
        pl.plot(yv, rho_j_y.imag, label='Im rho j')
        pl.plot(yv, omega_f_y.real, label='Re omega f')
        pl.plot(yv, omega_f_y.imag, label='Im omega f')
        pl.plot(yv, omega_j_y.real, label='Re omega j')
        pl.plot(yv, omega_j_y.imag, label='Im omega j')
        pl.legend()
        pl.figure()
        pl.title("Full quantities")
        pl.plot(yv, rho_f_y_full.real, label='Re rho f')
        pl.plot(yv, rho_f_y_full.imag, label='Im rho f')
        pl.plot(yv, rho_j_y_full.real, label='Re rho j')
        pl.plot(yv, rho_j_y_full.imag, label='Im rho j')
        pl.plot(yv, omega_f_y_full.real, label='Re omega f')
        pl.plot(yv, omega_f_y_full.imag, label='Im omega f')
        pl.plot(yv, omega_j_y_full.real, label='Re omega f')
        pl.plot(yv, omega_j_y_full.imag, label='Im omega f')
        pl.legend()
        pl.show()
    #f0 = Flux_j / (2.0 - Flux_f)

    #pl.figure()
    #pl.title("f0 @ k = %g" % k)
    #pl.plot(k, f0)

    if False:
        theta = np.linspace(0.0, 2.0 * np.pi, 301)
        #fv = 0.0 * thvals
        yi = np.array([-1.0, -0.5, -0.1, -0.01, -0.001, 0.001, 0.01, 0.1, 0.5, 1.0])
        fi = Fourier(q, yi)
        f = np.zeros((len(yi), len(theta)), dtype=complex)
        for i, th in enumerate(theta):
            kv = k * np.cos(th) + q * np.sin(th)
            kxv = k * np.sin(th) - q * np.cos(th)
            src  = gamma * rho_j - 2.0j * gamma /absk2 * kv
            src += - 2.0 * gamma / absk2 * omega_j * kxv
            fq = src / (gamma + 1j * kv)
            fy = np.dot(fi, fq)
            f[:, i] = fy
        for i, yi in enumerate(yi):
            pl.figure()
            pl.title("y = %g" % yi)
            pl.plot(theta, f[i, :].real, label='Re f for j')
            pl.plot(theta, f[i, :].imag, label='Im f for j')
            dtheta = theta[1] - theta[0]
            jx = np.sum(f[i, :] * np.cos(theta)) * dtheta / 2.0 / np.pi
            jy = np.sum(f[i, :] * np.sin(theta)) * dtheta / 2.0 / np.pi
            print("y = ", yi, "current", jx, jy)
            pl.legend()
        pl.show()
            
        #for yi in [-0.1, 0.0, 0.1, 0.5, 1.0]:
        #    yy = np.array([yi])
        #    fi = fourier(q, )
        
    
    flux_f_0 = flux_f_y[i0]
    flux_j_0 = flux_j_y[i0]
    rho_f_0 = rho_f_y[i0] 
    rho_j_0 = rho_j_y[i0]
    omega_f_0 = omega_f_y[i0]
    omega_j_0 = omega_j_y[i0]
    print("fluxes: ", flux_f_0, flux_j_0)
    return flux_f_0, flux_j_0, rho_f_0, rho_j_0, omega_f_0, omega_j_0

if __name__ == '__main__':
    #kvals = np.linspace(-10.0, 10.0, 1000)
    #kvals = np.linspace(-10.0, 10.0, 50)
    #kvals0 = np.linspace(0.003, 0.1, 51)
    #kvals1 = np.linspace(0.1, 10.0, 101)
    #kvals = list(kvals0)
    #kvals.extend(list(kvals1))
    #kvals = np.array(kvals)
    #kvals = np.linspace(-20.0, 20.0, 4000)
    kvals = np.linspace(0.01, 30.0, 3000)
    #kvals = np.linspace(-10.0, 10.0, 500)
    #kvals = np.linspace(0.01, 10.0, 20)
    #kvals = np.array([0.15, 0.2, 0.3])
    f_f = []
    f_j = []
    rho_f = []
    rho_j = []
    omega_f = []
    omega_j = []
    for k in kvals:
        print("k = ", k)
        ff_k, fj_k, rho_fk, rho_jk, omega_fk, omega_jk = wh_solve(k)
        f_f.append(ff_k)
        f_j.append(fj_k)
        rho_f.append(rho_fk)
        rho_j.append(rho_jk)
        omega_f.append(omega_fk)
        omega_j.append(omega_jk)

    f_f = np.array(f_f)
    f_j = np.array(f_j)
    rho_f = np.array(rho_f)
    rho_j = np.array(rho_j)
    omega_f = np.array(omega_f)
    omega_j = np.array(omega_j)

    fj =  (1.0 + f_j)/(1.0/np.pi - f_f)
    net_flux = fj * f_f + f_j
    net_density = rho_f * fj + rho_j

    r0 = 0.02
    src = np.exp( - r0**2 / 2.0 * kvals**2)

    flux_k = src * net_flux
    density_k = src * net_density
    xvals = np.linspace(-10.0, 10.0, 1000)

    fourier_x = Fourier(kvals, xvals)
    # doubling since only positive k's are included. 
    flux_x = np.dot(fourier_x, flux_k).real * 2.0
    density_x = np.dot(fourier_x, density_k).real * 2.0


    np.savez("f-data-kmax=%g" % np.max(kvals), k=kvals, f_f=f_f, f_j=f_j, rho_f=rho_f, rho_j=rho_j, omega_f=omega_f, omega_j=omega_j, fj=fj, net_flux=net_flux, net_density=net_density, r0=r0, x=xvals, flux_x=flux_x, density_x=density_x, flux_k=flux_k, density_k=density_k)

    pl.figure()
    pl.plot(kvals, f_f.real, label='Re ff')
    pl.plot(kvals, f_f.imag, label='Im ff')
    pl.plot(kvals, f_j.real, label='Re fj')
    pl.plot(kvals, f_j.imag, label='Im fj')
    pl.plot(kvals, rho_f, label='rho f')
    pl.plot(kvals, rho_j, label='rho j')
    pl.plot(kvals, omega_f, label='omega_f')
    pl.plot(kvals, omega_j, label='omega_j')
    pl.legend()

    pl.figure()
    pl.title("f0/j0")
    pl.plot(kvals, fj, label='f0/j0')
    pl.plot(kvals, net_flux, label='Flux')
    pl.legend()


    pl.figure()
    pl.title("Flux(x)")
    pl.plot(xvals, flux_x, label='Flux')
    pl.plot(xvals, density_x, label='Density')
    pl.show()




    #for k in [0.1, 0.3, 1.0, 3.0, 10.0]:
    #    wh_solve(k)

    #pl.show()




