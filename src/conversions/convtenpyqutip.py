import numpy as np

import tenpy
import tenpy.linalg.np_conserved, tenpy.networks.mps
import qutip


tenpyMPSToProbAmps = tenpy_mps_to_probamp

def tenpy_mps_to_probamp (mps):
    '''
    Converts a matrix product state represented
    using a `tenpy` implementation in `tenpy.networks.mps.MPS`
    to a ket represented as `qutip.qobj.Qobj`
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
    #
    # qutip_qobj = qutip.qobj.Qobj(amp, dims=[mps.dim, [1]*len(mps.dim)])
    #
    # return qutip_qobj


if __name__ == '__main__':
    n_sites = 8
    psi = qutip.rand_ket(2 ** n_sites, dims=[[2]*n_sites, [1]*n_sites])

    ''' I do not understand the concept of `LegCharge`, `ChargeInfo`.'''
    chinfo = tenpy.linalg.np_conserved.ChargeInfo([1], ['2*Sz'])
    p  = tenpy.linalg.np_conserved.LegCharge.from_qflat(chinfo, [1, -1], qconj=+1)

    #chinfo = tenpy.linalg.np_conserved.ChargeInfo([], [])
    p  = tenpy.linalg.np_conserved.LegCharge.from_trivial(2, qconj=+1)

    psi_npconserved = tenpy.linalg.np_conserved.Array.from_ndarray(\
        psi.data.toarray().reshape([2 for n in range(n_sites)]), [p for n in range(n_sites)])
