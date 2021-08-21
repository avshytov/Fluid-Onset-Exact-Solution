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
import util
#from cauchy import cauchy_integral
from contours import make_paths_and_kernels

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
        print ("*** Solve for k = ", k, "h = ", self.h)
        gamma = self.gamma
        gammma1 = self.gamma1

        # Prepare the integration contours and the tabulated kernels
        K = xkernel.WHKernels(self.gamma, self.gamma1)
        path_up, K_up, path_dn, K_dn = make_paths_and_kernels(K, k)
        if False:
           path_up_old, K_up_old, path_dn_old, K_dn_old \
               = make_contours_and_kernels(k, gamma, gamma1)
           import pylab as pl
           x_arc_old = path_up_old.arc_lengths()
           x_arc_new = path_up.arc_lengths()
           pl.figure()
           pl.plot(x_arc_new, np.abs(K_up.Krho_p - K_up_old.Krho_p),
                   label='diff Krho+ up')
           pl.plot(x_arc_new, np.abs(K_up.Komega_p - K_up_old.Komega_p),
                   label='diff Komega+ up')
           pl.plot(x_arc_new, np.abs(K_dn.Krho_p - K_dn_old.Krho_p),
                   label='diff Krho+ dn')
           pl.plot(x_arc_new, np.abs(K_dn.Komega_p - K_dn_old.Komega_p),
                   label='diff Komega+ dn')
           pl.legend()
           pl.figure()
           pl.plot(x_arc_new, K_up.Krho_p.real, label='Re Krho new up')
           pl.plot(x_arc_new, K_up.Krho_p.imag, label='Im Krho new up')
           pl.plot(x_arc_old, K_up_old.Krho_p.real,
                   '--', label='Re Krho old up')
           pl.plot(x_arc_old, K_up_old.Krho_p.imag,
                   '--', label='Im Krho old up')

           pl.plot(x_arc_new, K_up.Komega_p.real, label='Re Komega new up')
           pl.plot(x_arc_new, K_up.Komega_p.imag, label='Im Komega new up')
           pl.plot(x_arc_old, K_up_old.Komega_p.real,
                   '--', label='Re Komega old up')
           pl.plot(x_arc_old, K_up_old.Komega_p.imag,
                   '--', label='Im Komega old up')
           pl.legend()

           pl.figure()
           pl.plot(x_arc_new, K_dn.Krho_p.real, label='Re Krho new dn')
           pl.plot(x_arc_new, K_dn.Krho_p.imag, label='Im Krho new dn')
           pl.plot(x_arc_old, K_dn_old.Krho_p.real,
                   '--', label='Re Krho old dn')
           pl.plot(x_arc_old, K_dn_old.Krho_p.imag,
                   '--', label='Im Krho old dn')

           pl.plot(x_arc_new, K_dn.Komega_p.real, label='Re Komega new dn')
           pl.plot(x_arc_new, K_dn.Komega_p.imag, label='Im Komega new dn')
           pl.plot(x_arc_old, K_dn_old.Komega_p.real,
                   '--', label='Re Komega old dn')
           pl.plot(x_arc_old, K_dn_old.Komega_p.imag,
                   '--', label='Im Komega old dn')
           pl.legend()
           pl.show()
 
        # Prepare the building blocks
        diffuse_flow   = DiffuseFlow(k, K_up, path_up, K_dn, path_dn)
        if (self.h < 0.001):
            injected_flow = EdgeInjectedFlow(k, K_up, path_up, K_dn, path_dn)
        else:
            injected_flow = InjectedFlow(h, k, K_up, path_up, K_dn, path_dn)
        stokeslet_x = Stokeslet(1, 0, self.h, k, K_up, path_up, K_dn, path_dn)
        stokeslet_y = Stokeslet(0, 1, self.h, k, K_up, path_up, K_dn, path_dn)
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

