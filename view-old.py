import numpy as np
import pylab as pl

def readData(fname):
    d = np.load(fname)
    for k in d.keys(): print (k)
    y = d['y']
    rho = d['rho']
    drho = d['drho']
    rho_s = d['rho_s']
    rho_j = d['bare_rho_j']
    drho_j = d['bare_rho_j']
    print ("rho_s = ", rho_s)
    print ("f_s = ", d['f_s'])
    pl.figure()
    pl.plot(y, rho[0, :].real, label='Re rho')
    pl.plot(y, rho[0, :].imag, label='Im rho')
    pl.plot(y, drho[0, :].real, label='Re rho')
    pl.plot(y, drho[0, :].imag, label='Im rho')
    pl.plot(y, rho_s.real, label='Re rho_s')
    pl.plot(y, rho_s.imag, label='Im rho_s')
    pl.plot(y, rho_j[:].real, label='Re rho_j')
    pl.plot(y, rho_j[:].imag, label='Im rho_j')
    pl.legend()
    
    jx = d['bare_jx']
    jy = d['bare_jy']
    pl.figure()
    pl.plot(y, jx[:].real, label='Re j_x bare')
    pl.plot(y, jx[:].imag, label='Im j_x bare')
    pl.plot(y, jy[:].real, label='Re j_y bare')
    pl.plot(y, jy[:].imag, label='Im j_y bare')
    pl.legend()

    rho_sin = d['bare_rho_sin']
    jx_sin = d['bare_jx_sin']
    jy_sin = d['bare_jy_sin']
    pl.figure()
    pl.plot(y, rho_sin[:].real, label='Re rho sin')
    pl.plot(y, rho_sin[:].imag, label='Im rho sin')
    pl.plot(y, jx_sin[:].real, label='Re j_x sin')
    pl.plot(y, jx_sin[:].imag, label='Im j_x sin')
    pl.plot(y, jy_sin[:].real, label='Re j_y sin')
    pl.plot(y, jy_sin[:].imag, label='Im j_y sin')
    pl.legend()
    
    rho_cos = d['bare_rho_cos']
    jx_cos = d['bare_jx_cos']
    jy_cos = d['bare_jy_cos']
    pl.figure()
    pl.plot(y, rho_cos[:].real, label='Re rho cos')
    pl.plot(y, rho_cos[:].imag, label='Im rho cos')
    pl.plot(y, jx_cos[:].real, label='Re j_x cos')
    pl.plot(y, jx_cos[:].imag, label='Im j_x cos')
    pl.plot(y, jy_cos[:].real, label='Re j_y cos')
    pl.plot(y, jy_cos[:].imag, label='Im j_y cos')
    pl.legend()
    
    rho_j = d['bare_rho_j']
    jx_j  = d['bare_jx']
    jy_j  = d['bare_jy']
    pl.figure()
    pl.plot(y, rho_j[:].real, label='Re rho j')
    pl.plot(y, rho_j[:].imag, label='Im rho j')
    pl.plot(y, jx_j[:].real, label='Re j_x j')
    pl.plot(y, jx_j[:].imag, label='Im j_x j')
    pl.plot(y, jy_j[:].real, label='Re j_y j')
    pl.plot(y, jy_j[:].imag, label='Im j_y j')
    pl.legend()
    
    rho_s = d['rho_s']
    jx_s  = d['jx_s']
    jy_s  = d['jy_s']
    pl.figure()
    pl.plot(y, rho_s[:].real, label='Re rho s')
    pl.plot(y, rho_s[:].imag, label='Im rho s')
    pl.plot(y, jx_s[:].real, label='Re j_x s')
    pl.plot(y, jx_s[:].imag, label='Im j_x s')
    pl.plot(y, jy_s[:].real, label='Re j_y s')
    pl.plot(y, jy_s[:].imag, label='Im j_y s')
    pl.legend()

    
    jx_sin_d = d['jx_sin']
    jy_sin_d = d['jy_sin']
    rho_sin_d = d['rho_sin']
    pl.figure()
    pl.plot(y, rho_sin_d[0, :].real, label='Re rho sin+d')
    pl.plot(y, rho_sin_d[0, :].imag, label='Im rho sin+d')
    pl.plot(y, jx_sin_d[0, :].real, label='Re j_x sin + d')
    pl.plot(y, jx_sin_d[0, :].imag, label='Im j_x sin + d')
    pl.plot(y, jy_sin_d[0, :].real, label='Re j_y sin + d')
    pl.plot(y, jy_sin_d[0, :].imag, label='Im j_y sin + d')
    pl.legend()
    
    jx_cos_d = d['jx_cos']
    jy_cos_d = d['jy_cos']
    rho_cos_d = d['rho_cos']
    pl.figure()
    pl.plot(y, rho_cos_d[0, :].real, label='Re rho cos+d')
    pl.plot(y, rho_cos_d[0, :].imag, label='Im rho cos+d')
    pl.plot(y, jx_cos_d[0, :].real, label='Re j_x cos + d')
    pl.plot(y, jx_cos_d[0, :].imag, label='Im j_x cos + d')
    pl.plot(y, jy_cos_d[0, :].real, label='Re j_y cos + d')
    pl.plot(y, jy_cos_d[0, :].imag, label='Im j_y cos + d')
    pl.legend()
    
    jx_j_d = d['jx']
    jy_j_d = d['jy']
    rho_j_d = d['rho']
    pl.figure()
    pl.plot(y, rho_j_d[0, :].real, label='Re rho j+d')
    pl.plot(y, rho_j_d[0, :].imag, label='Im rho j+d')
    pl.plot(y, jx_j_d[0, :].real, label='Re j_x j + d')
    pl.plot(y, jx_j_d[0, :].imag, label='Im j_x j + d')
    pl.plot(y, jy_j_d[0, :].real, label='Re j_y j + d')
    pl.plot(y, jy_j_d[0, :].imag, label='Im j_y j + d')
    pl.legend()
    
    pl.show()
    


import sys
for f in sys.argv[1:]:
    readData(f)
