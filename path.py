import numpy as np
from contourint import SinglePathIntegrator

class Path:
    def __init__ (self, z):
        self.t = np.linspace(0, 1, len(z))
        self.z = np.array(z)
        self.update()
        self.invalidate()
        self.integrator = None

    def update(self):
        self.find_arc_lengths()

    def begins_at(self):
        return self.z[0]
    
    def ends_at(self):
        return self.z[-1]

    def arc_lengths(self): return self.arc_len

    def find_arc_lengths(self):
        z = self.z
        dz = z[1:] - z[:-1]
        self.arc_len = np.zeros((len(z)))
        self.arc_len[1:] = np.cumsum(np.abs(dz))        
        
    def points(self):
        return self.z
    
    def copy(self):
        path_copy = Path(self.z)
        path_copy.t = self.t
        return path_copy
        
    def setup(self):
        if not self.integrator:
           self.integrator = SinglePathIntegrator(self.z)

    def invalidate(self):
        self.integrator = None

    def reverse(self):
        z_new = list(self.z)
        #dt = self.t[1:] - self.t[:-1]
        t_a = self.t[0]
        t_b = self.t[-1]
        t_new = list(t_b + t_a - self.t)
        t_new.reverse()
        z_new.reverse()
        self.z = np.array(z_new)
        self.t = np.array(t_new)
        self.invalidate()
        self.update()
        
    def transform(self, f):
        z_new = self.eval(f)
        self.z = z_new
        self.invalidate()
        self.update()
        
    def append(self, path):
        t_append = (path.t - path.t[0]) + self.t[-1]
        z_new = list(self.z)
        t_new = list(self.t)
        z_append = path.z
        # Remove to avoid the duplication
        if abs(z_append[0] - z_new[-1]) < 1e-10:
            z_append = z_append[1:]
            t_append = t_append[1:]
        z_new.extend(z_append)
        t_new.extend(t_append)
        self.z = np.array(z_new)
        self.t = np.array(t_new)
        self.invalidate()
        self.update()

    def eval(self, f):
       return np.vectorize(f)(self.z)
        
    def integrate(self, f):
        return self.integrate_array(self.eval(f))
    
    def integrate_array(self, f_arr):
        self.setup()
        return self.integrator(f_arr)
       
def append_paths(*path_list):
    if len(path_list) < 1: return None
    path_ret = path_list[0].copy()
    for path in path_list[1:]:
        path_ret.append(path)
    return path_ret

def transform(path, f):
    path_copy = path.copy()
    path_copy.transform(f)
    return path_copy

def reverse(path):
    path_copy = path.copy()
    path_copy.reverse()
    return path_copy
   
class JointPath:
    def __init__ (self):
        self.paths = []
        
    def add(self, path):
        self.paths.append(path)

    def integrate(self, f):
        ret = 0.0 + 0.0j
        for path in self.paths:
            ret += path.integrate(f)
            
def join_paths(*path_list):
    joint_path = JointPath()
    for path in path_list:
        joint_path.join(path)
    return joint_path

def scale_grid(tmin, tmax, N, scaling):
    tau = np.linspace(0.0, 1.0, N + 1)
    tau_min = scaling(0.0)
    tau_max = scaling(1.0)
    tau = (scaling(tau) - tau_min) / (tau_max - tau_min)
    return tmin * (1.0 - tau) + tmax * tau

class StraightPath(Path):
    def __init__ (self, a, b, N, scaling = lambda t: t):
        #t = np.linspace(0.0, 1.0, N + 1)
        t = scale_grid(0.0, 1.0, N, scaling)
        z = a * (1.0 - t) + b * t
        Path.__init__ (self, z)

class CirclePath(Path):
    def __init__ (self, z0, R, N, scaling = lambda t: t):
        #theta = np.linspace(0.0, 2.0 * np.pi, N + 1)
        theta = scale_grid(0.0, 2.0 * np.pi, N, scaling)
        z = z0 + R * np.exp(1j * theta)
        Path.__init__ (self, z)

class ArcPath(Path):
    def __init__ (self, z0, a, b, th_0, th_1, N, scaling=lambda t: t):
        #theta = np.linspace(th_0, th_1, N + 1)
        theta = scale_grid(th_0, th_1, N, scaling)
        x = z0.real + a * np.cos(theta)
        y = z0.imag + b * np.sin(theta)
        Path.__init__ (self, x + 1j * y)

if __name__ == '__main__':
    #path_up = StraightPath(1000j - 0.1, 0.5j - 1)
    #path_up.append(ArcPath(0.5j, 1.0, 0.25, np.pi, 1.5*np.pi))
    path_up_left = append_paths(StraightPath(1000j - 0.001, 100j - 0.01, 1000),
                                StraightPath(100j  - 0.01, 0.5j - 1, 1000),
                                ArcPath(0.5j, 1.0, 0.25, np.pi, 1.5*np.pi, 50))
    path_up_right = transform(reverse(path_up_left),
                              lambda z: complex(-z.real, z.imag))
    path_up = append_paths(path_up_left, path_up_right)
    I = path_up.integrate(lambda z: 1.0 / (z - 1j))
    print ("integral = ", I, "err = ", I - 2.0 * np.pi * 1j)
    import pylab as pl
    pl.figure()
    pl.plot(path_up.z.real, path_up.z.imag)
    pl.plot(path_up_left.z.real, path_up_left.z.imag)
    pl.plot(path_up_right.z.real, path_up_right.z.imag)
    pl.show()
    

        
