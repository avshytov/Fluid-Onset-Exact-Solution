import numpy as np
from whsolver import WHSolver

def run(h, gamma, gamma1, kvals, yvals, fname):
    solver = WHSolver(h, gamma, gamma1, kvals, yvals)
    solver.run(fname)

def join_arrays(*arrays):
    res_list = []
    for a in arrays:
        res_list.extend(list(a))
    return np.array(res_list)

if __name__ == '__main__':
    kvals = join_arrays( np.linspace(0.001, 0.009, 5),
                         np.linspace(0.01, 0.99, 99),
                         np.linspace(1.0, 10.0, 361),
                         np.linspace(10.1, 30.0, 200))
    yvals = np.linspace(-1.0, 10.0, 1101)

    gamma  = 1.0
    gamma1 = 1.0
    ver = "01e"

    for h in [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
    #for h in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
    #for h in [0.1,  0.2, 0.3,  0.4, 4.0, 4.5, 8.5,  9.0, 13.0]:
    #for h in [0.6,  0.7, 0.8,  0.9, 3.5, 5.5, 7.5,  9.5, 12.0]:
    #for h in [1.25, 1.5, 1.75, 2.5, 6.0, 6.5, 7.0, 10.0, 11.0]:
        fname = "whnew-data-ver%s-h=%g-gamma1=%g" % (ver, h, gamma1)
        run(h, gamma, gamma1, kvals, yvals, fname)

