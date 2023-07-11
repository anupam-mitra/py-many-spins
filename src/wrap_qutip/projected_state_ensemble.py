import numpy as np
import qutip

from . import ensembles

def generate_basis_vectors(psi:qutip.qobj.Qobj):
    """Generates a basis for the pure state
    space of `psi`"""
    
    if not psi.isbra:
        identity = qutip.identity(psi.dims[0])
    else:
        identity = qutip.identity(psi.dims[1])
        
    basis_list = identity.eigenstates()[1]
    
    return basis_list

class ProjectedStateEnsemble(ensembles.StateEnsemble):
    """Represents a projected state ensemble"""

    def __init__(self, 
        state:qutip.qobj.Qobj,
        selected_sites:tuple, 
        all_sites:tuple=None):
        
        self.state = state
        self.selected_sites = selected_sites

        if all_sites != None:
            self.all_sites = all_sites
        else:
            self.all_sites = \
                [l for l in range(len(state.dims[0]))]

    def generate_povm(self):
        """Generates the POVM elements for projective
        measurements"""

        self.project_meas_sites = \
            tuple(set(self.all_sites) - set(self.selected_sites))

        project_meas_sample = qutip.ptrace(self.state, self.project_meas_sites)

        self.project_meas_basis = generate_basis_vectors(project_meas_sample)
        self.project_meas_povm = \
            [qutip.ket2dm(ket) for ket in self.project_meas_basis]
    
    def generate_ensemble(self):
        """Generates the projected state ensemble"""

        projected_states, probabilities = \
            qutip.measurement.measurement_statistics_povm(\
                state=self.state, ops=self.project_meas_povm,\
                targets=self.project_meas_sites)

        self.probabilities = probabilities
        self.projected_states = []

        # Project states now contains the ket for all sites conditioned
        # on the povm element.
        # Extract the ket on selected sites by partial tracing
        # to obtain the reduce density matrix and selecting its
        # eigenvector with highest eigenvalue
        for ket in projected_states:
            reduced_dm = qutip.ptrace(ket, self.selected_sites)
            projected_ket = reduced_dm.eigenstates()[1][-1]

            self.projected_states.append(projected_ket)