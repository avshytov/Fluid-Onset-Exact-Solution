import pylab as pl
import numpy as np

def make_diff(x_new, y_new, x_old, y_old, factor = 1):
    x_diff = []
    y_diff = []
    for i_new, x in enumerate(x_new):
        i_old = np.argmin(np.abs(x_old - x))
        x1 = x_old[i_old]
        if abs(x1 - x) > 1e-5: continue
        x_diff.append(x)
        y_diff.append(y_new[i_new] - y_old[i_old].conj() * factor)
    x_diff = np.array(x_diff)
    y_diff = np.array(y_diff)
    return x_diff, y_diff

def compare(data_new, data_old, x_new_key, x_old_key, y_new_key, y_old_key,
            factor = 1):
    x_new = data_new[x_new_key]
    y_new = data_new[y_new_key]
    x_old = data_old[x_old_key]
    y_old = data_old[y_old_key].conj() * factor

    pl.figure()
    pl.plot(x_new, y_new.real, label='Re %s' % y_new_key)
    pl.plot(x_new, y_new.imag, label='Im%s' % y_new_key)
    pl.plot(x_old, y_old.real, '--', label='Re %s' % y_old_key)
    pl.plot(x_old, y_old.imag, '--', label='Im %s' % y_old_key)
    pl.legend()

    x_diff, y_diff = make_diff(x_new, y_new, x_old, y_old, factor)
    pl.figure()
    pl.plot(x_diff, y_diff.real, label='Re diff y')
    pl.plot(x_diff, y_diff.imag, label='Im diff y')
    pl.legend()

def compare_arr_row(data_new, data_old, x_new_key, x_old_key, y_new_key,
                y_old_key, x0, factor = 1):
    x_new = data_new[x_new_key]
    y_new_arr = data_new[y_new_key]
    x_old = data_old[x_old_key].conj() * factor
    y_old_arr = data_old[y_old_key]

    i_new = np.argmin(np.abs(x_new - x0))
    i_old = np.argmin(np.abs(x_old - x0))
    print ("compare @ old: x = ", x_new[i_new], x_old[i_old])
    y_new = y_new_arr[:, i_new]
    y_old = y_old_arr[:, i_old]

    pl.figure()
    pl.plot(x_new, y_new.real, label='Re %s' % y_new_key)
    pl.plot(x_new, y_new.imag, label='Im%s' % y_new_key)
    pl.plot(x_old, y_old.real, '--', label='Re %s' % y_old_key)
    pl.plot(x_old, y_old.imag, '--', label='Im %s' % y_old_key)
    pl.legend()

    x_diff, y_diff = make_diff(x_new, y_new, x_old, y_old, factor)
    pl.figure()
    pl.plot(x_diff, y_diff.real, label='Re diff y')
    pl.plot(x_diff, y_diff.imag, label='Im diff y')
    pl.legend()

def compare_arr_col(data_new, data_old, x_new_key, x_old_key, y_new_key,
                y_old_key, x0, factor = 1):
    x_new = data_new[x_new_key]
    y_new_arr = data_new[y_new_key]
    x_old = data_old[x_old_key].conj() * factor
    y_old_arr = data_old[y_old_key]

    i_new = np.argmin(np.abs(x_new - x0))
    i_old = np.argmin(np.abs(x_old - x0))
    print ("compare @ old: x = ", x_new[i_new], x_old[i_old])
    y_new = y_new_arr[i_new, :]
    y_old = y_old_arr[i_old, :]

    pl.figure()
    pl.plot(x_new, y_new.real, label='Re %s' % y_new_key)
    pl.plot(x_new, y_new.imag, label='Im%s' % y_new_key)
    pl.plot(x_old, y_old.real, '--', label='Re %s' % y_old_key)
    pl.plot(x_old, y_old.imag, '--', label='Im %s' % y_old_key)
    pl.legend()

    x_diff, y_diff = make_diff(x_new, y_new, x_old, y_old, factor)
    pl.figure()
    pl.plot(x_diff, y_diff.real, label='Re diff y')
    pl.plot(x_diff, y_diff.imag, label='Im diff y')
    pl.legend()

fname_old = "wh-data16-h=0.5-delta=0.0001-kmin=0.001-kmax=30.npz"
data_old = np.load(fname_old)
fname_new = 'whnew-data-ver01b-h=0.5-gamma1=1.npz'
data_new = np.load(fname_new)

print ("new keys", fname_old)
print (list(data_new.keys()))
print ("old keys", fname_new)
print (list(data_old.keys()))
#compare(data_new, data_old, 'k', 'k', 'I:f', 'f_s')
compare(data_new, data_old, 'k', 'k', 'Fx:f', 'f_cos', 1.0)
compare(data_new, data_old, 'k', 'k', 'Fy:f', 'f_sin', 1.0)
compare_arr_row(data_new, data_old, 'k', 'k', 'Fx:rho', 'rho_cos', 0.01, 1.0)
compare_arr_col(data_new, data_old, 'y', 'y', 'Fy:rho', 'rho_sin', 0.1, 1.0)
pl.show()

compare_arr_row(data_new, data_old, 'k', 'k', 'Fx:jx', 'jx_cos', 0.01, 1.0)
compare_arr_row(data_new, data_old, 'k', 'k', 'Fx:jy', 'jy_cos', 0.01, 1.0)
compare_arr_row(data_new, data_old, 'k', 'k', 'Fy:jx', 'jx_sin', 0.01, 1.0)
compare_arr_row(data_new, data_old, 'k', 'k', 'Fy:jy', 'jy_sin', 0.01, 1.0)

compare_arr_col(data_new, data_old, 'y', 'y', 'Fx:jx', 'jx_cos', 0.1, 1.0)
compare_arr_col(data_new, data_old, 'y', 'y', 'Fx:jy', 'jy_cos', 0.1, 1.0)
compare_arr_col(data_new, data_old, 'y', 'y', 'Fy:jx', 'jx_sin', 0.1, 1.0)
compare_arr_col(data_new, data_old, 'y', 'y', 'Fy:jy', 'jy_sin', 0.1, 1.0)

pl.show()
