import numpy as np


class Integrator:
    def __init__ (self, x, func_const, func_lin):
        self.x = x
        self.func_const = func_const
        self.func_lin = func_lin
        self.wt = 0.0 * x + 0.0
        self.setup()

    def __call__(self, y):
        return self.integrate(y)
    
    def setup(self):
        self.wt = np.zeros((len(self.x)))
        for i in range(len(self.x) - 1):
            x1 = self.x[i]
            x2 = self.x[i + 1]
            I_const = self.func_const(x2) - self.func_const(x1)
            I_lin   = self.func_lin  (x2) - self.func_lin(x1)
            I_a = I_const
            I_b = I_lin - (x1 + x2) / 2.0 * I_const
            dx = x2 - x1
            wt1 = I_a / 2.0 - I_b / dx
            wt2 = I_a / 2.0 + I_b / dx
            self.wt[i] += wt1
            self.wt[i + 1] += wt2
            
    def integrate(self, y):
        return np.dot(y, self.wt)

def sqrt_integrator(a, x):
    def const_func(x):
        return 0.5 * x * np.sqrt(x**2 - a**2) - 0.5 * a**2 * np.log(x + np.sqrt(x**2 - a**2))
    def lin_func(x):
        return 1.0/3.0 * (x**2 - a**2)**1.5
    return Integrator(x, const_func, lin_func)

def invsqrt_integrator(a, x):
    def const_func(x):
        return np.log(x + np.sqrt(x**2 - a**2))
    def lin_func(x):
        return np.sqrt(x**2 - a**2)
    return Integrator(x, const_func, lin_func)

if False:
    a = 1.5
    x = np.linspace(a, 100*a, 3001)
    sqrt_int = sqrt_integrator(a, x)
    invsqrt_int = invsqrt_integrator(a, x)
    funcs = [
        lambda x: 1.0/(1.0 + x**2)**2,
        lambda x: np.exp(-0.2*x) 
    ]
    for f in funcs:
        y = f(x)
        I1 = sqrt_int.integrate(y)
        I2 = invsqrt_int.integrate(y)
        I1_exact, eps1 = integrate.quad(lambda x: np.sqrt(x**2 - a**2) * f(x),
                                        x[0], x[-1], limit=500)
        I2_exact, eps2 = integrate.quad(lambda x: 1.0/np.sqrt(x**2 - a**2) * f(x),
                                        x[0], x[-1], limit=500)
        print("Int sqrt(): found: ", I1, "exact: ", I1_exact, "err = ", (I1_exact - I1), "rel = ", (I1 - I1_exact) / I1)
        print("Int sqrt^-1(): found: ", I2, "exact: ", I2_exact, "err = ", (I2_exact - I2), "rel = ", (I2 - I2_exact) / I2)
    import sys
    sys.exit(0)
