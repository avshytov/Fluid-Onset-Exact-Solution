import numpy as np
#import pylab as pl

def sgn(x):
    sgn_x = np.zeros(np.shape(x))
    sgn_x[x >  1e-6] =  1.0
    sgn_x[x < -1e-6] = -1.0
    return sgn_x


#
# Borrowed from Simon, including the comment
#
# I took this from a forum on the internet (http://stackoverflow.com/questions/17044052/mathplotlib-imshow-complex-2d-array).
# It's very useful for plotting complex valued functions (brightness gives magnitude, color phase)

def colorize(z):
    from colorsys import hls_to_rgb
    r = np.abs(z)
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1.0 - 1.0/(1.0 + r**0.5)
    s = 0.8

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2)
    c = c.swapaxes(0,1)
    return c

def display_complex(Z, F, scale=1.0):
    import pylab as pl
    re_min = np.min(Z.real)
    re_max = np.max(Z.real)
    im_min = np.min(Z.imag)
    im_max = np.max(Z.imag)
    pl.figure()
    pl.imshow(colorize(scale*F), origin='lower',
              extent=(re_min, re_max, im_min, im_max))
