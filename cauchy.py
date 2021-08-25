import numpy as np
import path

def cauchy_integral_array(path, f_p, z):
    z_p = path.points()
    # matrix with entries (z - z')
    diff_z  = np.outer(np.ones(np.shape(z)), z_p)
    diff_z -= np.outer(z,                    np.ones(np.shape(z_p)))
    F_z = f_p / diff_z
    result =  path.integrate_array(F_z) / (2.0 * np.pi * 1j)
    #print ("np.shape(z) = ", np.shape(z))
    if np.shape(z) == ():
        result = result[0]
    return result

def cauchy_integral(path, f, z):
    z_p = path.points()
    f_p = f(z_p)
    return cauchy_integral_array(path, f_p, z)

