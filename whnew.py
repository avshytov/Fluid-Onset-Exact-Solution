import xkernel_new as xkernel
import path
import flows
import numpy as np
from diffuse import DiffuseFlow
from stokeslet import Stokeslet
from bulk import InjectedFlow
from edge import EdgeInjectedFlow
from flows import CombinedFlow
import time

def make_arc(k, kappa):
    abs_k = np.abs(k)
    a = np.sqrt(abs_k * kappa)
    b = kappa - abs_k / 2.0
    coarse = False
    if not coarse:
       path_vert = path.StraightPath(100.0j*kappa   - 0.1*a,
                                         1j * kappa -     a, 4000)
    else:
       path_vert = path.StraightPath(100.0j*kappa   - 0.1*a,
                                         1j * kappa -     a, 400)
    #path_vert = path.StraightPath(100.0j*kappa - 0.1*a, -a + 1j * kappa, 40)
    def scaling_func(t):
        return t * (2.0 - t)
        #xi  = 2 * t - 1.0      # (0, 1) -> (-1, 1)
        #eta = xi * np.abs(xi)  # more points near the middle
        #return (eta + 1.0) / 2.0 # (-1, 1) -> (0, 1)
    if not coarse:
       path_arc  = path.ArcPath(1j * kappa, a, b, np.pi, 1.5*np.pi, 251,
                                scaling_func)
    else:
       path_arc  = path.ArcPath(1j * kappa, a, b, np.pi, 1.5*np.pi, 25,
                               scaling_func)
    path_up_left  = path.append_paths(path_vert, path_arc)
    #path_up_right = path.transform(path.reverse(path_up_left),
    #                               lambda z: complex(-z.real, z.imag))
    #path_dn_left  = path.transform(path_up_left, lambda z: z.conj())
    #path_dn_right = path.transform(path_up_right, lambda z: z.conj())
    return path_up_left

def append_paths_and_kernels(path_a, K_a, path_b, K_b):

    if abs(K_a.k - K_b.k) > 1e-6:
        raise Exception("k do not match")
    if abs(K_a.gamma - K_b.gamma) > 1e-6:
        raise Exception("gammas do not match")
    if abs(K_a.gamma1 - K_b.gamma1) > 1e-6:
        raise Exception("gammas do not match")
    
    path_joint = path.append_paths(path_a, path_b)
    q_joint = path_joint.points()
    Krho_p   = 0.0 * q_joint + 0.0j
    Komega_p = 0.0 * q_joint + 0.0j
    Krho_p[0:len(path_a.points())] = K_a.Krho_p
    Krho_p[-len(path_b.points()):] = K_b.Krho_p
    Komega_p[0:len(path_a.points())] = K_a.Komega_p
    Komega_p[-len(path_b.points()):] = K_b.Komega_p
    K_joint = xkernel.TabulatedKernels(K_a.K, K_a.k, q_joint,
                                       Krho_p, Komega_p)
    return path_joint, K_joint

def reverse_path_and_kernel(pth, K):
    path_new = path.reverse(pth)
    Krho_p = list(K.Krho_p)
    Komega_p = list(K.Komega_p)
    Krho_p.reverse()
    Komega_p.reverse()
    Krho_p = np.array(Krho_p)
    Komega_p = np.array(Komega_p)
    Knew = xkernel.TabulatedKernels(K.K, K.k,
                                  path_new.points(), Krho_p, Komega_p)
    return path_new, Knew

def conjugate_left_right(pth, K):
    path_c = path.transform(pth, lambda z: complex(-z.real, z.imag))
    Krho_p = K.Krho_p.conj()
    Komega_p = K.Komega_p.conj()
    Knew = xkernel.TabulatedKernels(K.K, K.k,
                                    path_c.points(), Krho_p, Komega_p)
    return path_c, Knew

def conjugate_up_down(pth, K):
    path_c     = path.transform(pth, lambda z: complex(z.real, -z.imag))

    # When z is changed to z_bar, X+ becomes X_- conjugate
    Krho_c       = K.K.rho(K.k,   path_c.points())
    Komega_c     = K.K.omega(K.k, path_c.points())

    Krho_p     = K.Krho_p
    Komega_p   = K.Komega_p

    Krho_m_new   =  1.0/Krho_p.conj()
    Komega_m_new = 1.0 / Komega_p.conj()
    Krho_p_new   = Krho_m_new / Krho_c
    Komega_p_new = Komega_m_new / Komega_c
    #Krho_m     = Krho * Krho_p
    #Komega_m   = Komega * Komega_p

    #Krho_new   = Krho_m.conj()
    #Komega_new = Komega_m.conj()

    Knew = xkernel.TabulatedKernels(K.K, K.k, path_c.points(),
                                    Krho_p_new, Komega_p_new)
    return path_c, Knew


def make_contours_and_kernels(k, gamma, gamma1):
    kappa = np.sqrt(gamma**2 + k**2)
    #path_up, path_dn = make_contours_im(abs(k), kappa)
    path_ul = make_arc(abs(k), kappa)
    # upper left segment, used to construct the rest 
    

    #show_paths(path_up, path_dn, q_m)
    #import pylab as pl; pl.show()
    
    K = xkernel.WHKernels(gamma, gamma1)
    print ("tabulate kernels up")
    import time; now = time.time()
    K_ul = xkernel.tabulate_kernel(K, k, path_ul.points())
    print ("tabulation done, ", time.time() - now)
    #import sys; sys.exit(0)
    #K_ul = xkernel.load_kernel(K, k, path_ul.points(), "ul", False)
    # get upper-right segment
    path_ru, K_ru = conjugate_left_right(path_ul, K_ul) # backward
    path_ur, K_ur = reverse_path_and_kernel(path_ru, K_ru)
    
    path_up, K_up = append_paths_and_kernels(path_ul, K_ul, path_ur, K_ur)
    path_dn, K_dn = conjugate_up_down(path_up, K_up)

    return path_up, K_up, path_dn, K_dn

class WHSolver:
    def __init__ (self, h, gamma, gamma1, kvals, yvals):
        self.gamma  = gamma
        self.gamma1 = gamma1
        self.kvals  = kvals
        self.h = h
        self.yvals = yvals
        self.data = []

    def run(self, fname):
        save_interval = 10
        for i_k, k in enumerate(self.kvals):
            result = self.solve(k)
            self.data.append((k, result))
            if  i_k  % save_interval == 0:
                self.save(fname)
        self.save(fname)
            
    def solve(self, k):
        print ("*** Solve for k = ", k, "h = ", h)
        gamma = self.gamma
        gammma1 = self.gamma1

        # Prepare the integration contours and the tabulated kernels
        path_up, K_up, path_dn, K_dn = make_contours_and_kernels(k, gamma,
                                                                 gamma1)

        # Prepare the building blocks
        diffuse_flow   = DiffuseFlow(k, K_up, path_up, K_dn, path_dn)
        if (h < 0.001):
            injected_flow = EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn)
        else:
            injected_flow = InjectedFlow(h, k, K_up, path_up, K_dn, path_dn)
        stokeslet_x = Stokeslet(1, 0, h, k, K_up, path_up, K_dn, path_dn)
        stokeslet_y = Stokeslet(0, 1, h, k, K_up, path_up, K_dn, path_dn)
        bare_flows = {
            "I"    : injected_flow,
            "Fx"   : stokeslet_x,
            "Fy"   : stokeslet_y
        }

        results = dict()
        # Solve the problem for flows of all types
        for key, bare_flow in bare_flows.items():
            field = dict()

            # Determine the diffuse flux from self-consistency
            bare_flux = bare_flow.wall_flux()
            diff_flux = diffuse_flow.wall_flux()
            f_s = bare_flux / (1.0/np.pi - diff_flux)
            field['f'] = f_s
            print (key, "f_s = ", f_s)

            # Form a superposition of the bare flow and the diffuse flow
            total_flow = CombinedFlow(K_up, path_up, K_dn, path_dn)
            total_flow.add(bare_flow, 1.0)
            total_flow.add(diffuse_flow, f_s)

            # Evaluate the fields and store the results
            yvals = self.yvals
            field['rho_bare']   = bare_flow.rho_y(yvals)
            field['drho_bare']  = bare_flow.drho_y(yvals)
            field['jx_bare']    = bare_flow.jx_y(yvals)
            field['jy_bare']    = bare_flow.jy_y(yvals)
            field['flux_bare']  = bare_flow.wall_flux()
            field['flux_diff']  = diffuse_flow.wall_flux()
            field['rho']        = total_flow.rho_y(yvals)
            field['drho']       = total_flow.drho_y(yvals)
            field['jx']         = total_flow.jx_y(yvals)
            field['jy']         = total_flow.jy_y(yvals)
            field['flux']       = total_flow.wall_flux()
            results[key] = field
            
        return results
    
    def save(self, fname):
        if not len(self.data): return 
        results = dict()

        # Store the parameters
        results['h']      = self.h
        results['gamma']  = self.gamma
        results['gamma1'] = self.gamma1
        results['y']      = self.yvals
        
        # first, record k values
        k_done = np.array([t[0] for t in self.data])
        results['k'] = k_done
        # Parse the first item to determine the names
        # of flows and fields
        k0, data0 = self.data[0]
        flows = list(data0.keys())
        fields = data0[flows[0]].keys()
        res_keys = []

        #
        # Use these keys in the .npz file
        #
        def make_key(flow, field):
                return '%s:%s' % (flow, field)

        # Now make empty arrays to sort the data items into
        for flow in flows:
            for field in fields:
                results[make_key(flow, field)] = []

        # Scan the data
        for k, result_k in self.data:
            # For each flow, extract individual fields: rho, jx, jy, etc
            # and append to the data already collected
            for flow, flow_fields in result_k.items():
                for field, data in flow_fields.items():
                    results[make_key(flow, field)].append(data)

        # Convert lists to numpy arrays
        for key in results.keys():
            results[key] = np.array(results[key])

        # Save the data
        np.savez(fname, **results)

def run(h, gamma, gamma1, kvals, yvals, fname):
    solver = WHSolver(h, gamma, gamma1, kvals, yvals)
    solver.run(fname)

def join_arrays(*arrays):
    res_list = []
    for a in arrays:
        res_list.extend(list(a))
    return np.array(res_list)
    
kvals = join_arrays( np.linspace(0.001, 0.009, 5),
                     np.linspace(0.01, 0.99, 99),
                     np.linspace(1.0, 10.0, 361),
                     np.linspace(10.1, 30.0, 200))
yvals = np.linspace(-1.0, 10.0, 1101)

gamma  = 1.0
gamma1 = 1.0
ver = "01a"

for h in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
    fname = "whnew-data-ver%s-h=%g-gamma1=%g" % (ver, h, gamma1)
    run(h, gamma, gamma1, kvals, yvals, fname)

