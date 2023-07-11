import numpy as np
import quimb
import quimb.tensor
import pandas

######################################################################################
######################################################################################
# TODO Move to a separate file
# TODO Implement saving and indexing functionality
class QuimbTEBD1DSolver:
    '''
    Performs a TEBD simulation
    '''

    def __init__ (self,
        initial_mps,
        n_sites,
        t_initial,
        t_final,
        n_steps,
        hamiltonian_builder=None,
        hamiltonian_local=None,
        trotter_opts=None,
        split_opts=None):

        self.initial_mps = initial_mps
        self.n_sites = n_sites
        
        self.t_initial = t_initial
        self.t_final = t_final
        self.n_steps = n_steps

        self.split_opts = split_opts
        self.trotter_opts = trotter_opts
        
        assert (hamiltonian_builder != None) or (hamiltonian_local != None)
        
        if hamiltonian_builder != None:
            self.hamiltonian_builder = hamiltonian_builder
            self.hamiltonian_local = hamiltonian_builder.build_local_ham(n_sites)
            
        if hamiltonian_local != None:
            self.hamiltonian_local = hamiltonian_local

    def run (self):

        self.t_list =  np.linspace(self.t_initial, self.t_final, self.n_steps)

        self.tebd = quimb.tensor.TEBD(self.initial_mps, self.hamiltonian_local)
        self.tebd.split_opts = self.split_opts

        self.states = list(self.tebd.at_times(self.t_list, **self.trotter_opts))

    def evolve(self):
        if not hasattr(self, "states"):
            self.run()

        self.rows = []
        for ix_time in range(len(self.t_list)):
            row = {
                "ix_time": ix_time,
                "time": self.tlist[ix_time],
                "mps": self.states[ix_time],
            }
            self.rows += [row]


    def get_mps_history_df(self):

        if not hasattr(self, "df"):
            self.df = pandas.DataFrame(self.rows)

        return self.df
#
#
######################################################################################
######################################################################################

