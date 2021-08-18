import numpy as np

def get_poly_weights(z1, z2, z3, z0):
    z10 = z1 - z0
    z20 = z2 - z0
    z30 = z3 - z0
    M = np.zeros((3, 3), dtype=complex)
    M[0, 0] = z20 * z30 * (z30 - z20)
    M[1, 0] = (z20 - z30) * (z20 + z30)
    M[2, 0] = z30 - z20
    M[0, 1] = z10 * z30 * (z10 - z30)
    M[1, 1] = (z30 - z10) * (z30 + z10)
    M[2, 1] = z10 - z30
    M[0, 2] = z10 * z20 * (z20 - z10)
    M[1, 2] = (z10 - z20) * (z10 + z20)
    M[2, 2] = z20 - z10
    W = (z30 - z20) * (z30 - z10) * (z20 - z10)
    #print z20, z30, z10, W
    return np.transpose(M) / W

def get_quad_weights(z0, za, zb):
    I = np.zeros((3), dtype=complex)
    I[0] = zb - za
    I[1] = 0.5 * (zb - za) * ((zb - z0) - (z0 - za))
    #I[2] = 1.0/3.0 * ((zb - z0)**3 - (za - z0)**3)
    I[2] = 1.0/3.0 * (zb - za) * ((zb - z0)**2 + (za - z0)**2 + (zb - z0)*(za - z0))
    return I

class PathIntegrator:
    def __init__(self, z, wt):
        self.z = z
        self.wt = wt
    def __call__(self, f):
        return self.integrate(f)
    def integrate(self, f):
        return np.dot(f, self.wt)
    def eval(self, f):
        return np.vectorize(lambda z: f(z))(self.z)
        

class SinglePathIntegrator(PathIntegrator):
    def __init__(self, z):
        #ZIntetrator.__init__(self)
        #self.z = z
        wt = self.setup(z)
        PathIntegrator.__init__(self, z, wt)
    def setup(self, z):
        wt = 0.0 * z + 0.0j
        N = len(z)
        for i in range(1, N-1):
            z1 = z[i - 1]
            z2 = z[i]
            z3 = z[i + 1]
            z0 = z2
            M = get_poly_weights(z1, z2, z3, z0)
            I = get_quad_weights(z0, z1, z3)
            wt[i-1:i+2] += 0.5 * np.dot(M, I)
        M = get_poly_weights(z[0], z[1], z[2], z[0])
        I = get_quad_weights(z[0], z[0], z[1])
        wt[0:3] += 0.5 * np.dot(M, I)
        M = get_poly_weights(z[-3], z[-2], z[-1], z[-2])
        I = get_quad_weights(z[-2], z[-2], z[-1])
        wt[-3:] += 0.5 * np.dot(M, I)
        return wt

class JointPathIntegrator(PathIntegrator):
    def __init__(self, parts):
        z = []
        wt = []
        for part in parts:
            z.extend(list(part.z))
            wt.extend(list(part.wt))
        z = np.array(self.z)
        wt = np.array(self.z)
        PathIntegratoor.__init__(z, wt) 
    
if __name__ == '__main__':
    import pylab as pl
    t = np.linspace(0.0, 2.0*np.pi, 1001)
    Rx = 1.3
    Ry = 0.7
    dx = 0.1
    dy = 0.2
    def zc(t):
       return   (dx + Rx * np.cos(t)) + 1j * (dy + Ry * np.sin(t))
    z = zc(t)
    C_int = ContourIntegrator(z)
    for f in [ lambda z: 1.0,
               lambda z: 1.0 / (z**2 + 5.0),
               lambda z: 1.0/(z - 10j),
               lambda z: np.log(z + 10j)
            ]:
            z0 = 0.02 - 0.03j
            fz0 = f(z) / (z - z0)
            I = C_int(fz0) / 2.0 / np.pi / 1j
            print("I = ", I, "f(z0) = ", f(z0), "diff = ", I - f(z0))
            if False:
               pl.figure()
               pl.plot(t, z.real)
               pl.plot(t, z.imag)
               pl.plot(t, fz0.real)
               pl.plot(t, fz0.imag)
               pl.show()
    import sys
    sys.exit(0)
    
