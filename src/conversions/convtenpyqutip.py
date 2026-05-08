import itertools

import numpy as np
import tenpy
import tenpy.linalg.np_conserved, tenpy.networks.mps


def tenpy_mps_to_probamp(mps):
    '''
    Converts a matrix product state represented
    using a `tenpy` implementation in `tenpy.networks.mps.MPS`
    to a ket represented as `qutip.Qobj`
    by calculating each probability amplitude


    '''

    L = mps.L
    dimensions = mps.dim

    ket_labels = list(itertools.product(*[tuple(range(d)) for d in dimensions]))
    dim_manybody_state = 2**L

    amps = np.empty((dim_manybody_state), dtype=complex)

    for j in range(dim_manybody_state):
        label = ket_labels[j]
        basis_mps = tenpy.networks.mps.MPS.from_product_state(\
                        mps.sites, label, "finite")

        amps[j] = basis_mps.overlap(mps)

    return amps


tenpyMPSToProbAmps = tenpy_mps_to_probamp
