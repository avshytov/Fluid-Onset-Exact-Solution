import pylab as pl
import numpy as np

def make_diff(x_new, y_new, x_old, y_old):
    x_diff = []
    y_diff = []
    for i_new, x in enumerate(x_new):
        i_old = np.argmin(np.abs(x_old - x))
        x1 = x_old[i_old]
        if abs(x1 - x) > 1e-5: continue
        x_diff.append(x)
        y_diff.append(y_new[i_new] - y_old[i_old])
    x_diff = np.array(x_diff)
    y_diff = np.array(y_diff)
    return x_diff, y_diff

def compare(data_new, data_old, x_new_key, x_old_key, y_new_key, y_old_key):
    x_new = data_new[x_new_key]
    y_new = data_new[y_new_key]
    x_old = data_old[x_old_key]
    y_old = data_old[y_old_key]

    pl.figure()
    pl.plot(x_new, y_new.real, label='Re %s' % y_new_key)
    pl.plot(x_new, y_new.imag, label='Im%s' % y_new_key)
    pl.plot(x_old, y_old.real, '--', label='Re %s' % y_old_key)
    pl.plot(x_old, y_old.imag, '--', label='Im %s' % y_old_key)
    pl.legend()

    x_diff, y_diff = make_diff(x_new, y_new, x_old, y_old)
    pl.figure()
    pl.plot(x_diff, y_diff.real, label='Re diff y')
    pl.plot(x_diff, y_diff.imag, label='Im diff y')
    pl.legend()

data_old = np.load("wh-data16-h=0-delta=0.0001-kmin=0.001-kmax=30.npz")
data_new = np.load('whnew-data-ver01a-h=0-gamma1=1.npz')

print ("new keyw")
print (list(data_new.keys()))
print ("old keys")
print (list(data_old.keys()))
compare(data_new, data_old, 'k', 'k', 'I:f', 'f_s')

pl.show()
