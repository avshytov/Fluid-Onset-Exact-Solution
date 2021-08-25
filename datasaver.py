import numpy as np

class DataSaver:
    def __init__(self,  **kwargs):
        self.fname = None
        self.data = []
        self.vals = kwargs

    def set_filename(self, fname):
        self.fname = fname
        
    def set_values(self, **kwargs):
        self.vals.update(kwargs)
        
    def append_result(self, k, results):
        self.data.append((k, results))

    def save(self, fname = None):
        if fname == None: fname = self.fname
        if self.fname == None: self.fname = fname
        if not len(self.data): return 
        results = dict()
        results.update(self.vals)
        
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
        pass
