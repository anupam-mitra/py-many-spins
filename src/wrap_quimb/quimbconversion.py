import qutip
import quimb.tensor as qtn
import numpy as np


def quimb_mp_to_qutip_qobj (quimb_mp):
    '''
    Converts a qutip ket to a quimb
    Matrix Product State

    Parameters
    ----------
    quimb_mps:
        Representation as a quimb matrix product state
        or matrix product operator

    Returns
    -------
    qutip_ket:
        Representation as a qutip ket
    '''
    data = quimb_mp.to_dense()

    if isinstance(quimb_mp, qtn.MatrixProductState):
        dims = [[quimb_mp.phys_dim()] * quimb_mp.L, [1]]
    elif isinstance(quimb_mp, qtn.MatrixProductOperator):
        dims = [
            [quimb_mp.phys_dim()] * quimb_mp.L,
            [quimb_mp.phys_dim()] * quimb_mp.L,
        ]
        data = np.asarray(data)
    else:
        dims=None

    qutip_qobj = qutip.Qobj(data, dims=dims)

    return qutip_qobj


quimb_mp_to_ndarray = quimb_mp_to_qutip_qobj

def qutip_ket_to_quimb_mps (qutip_ket, cutoff=None, cutoff_mode='sum2', max_bond=None):
    '''
    Converts a qutip ket to a quimb
    Matrix Product State

    Parameters
    ----------
    qutip_ket:
        Representation as a qutip ket

    Returns
    -------
    quimb_mps:
        Representation as a quimb mps
    '''

    split_opts = {}

    if cutoff != None:
        split_opts['cutoff'] = cutoff
        split_opts['cutoff_mode'] = 'rsum2'

    if max_bond != None:
        split_opts['max_bond'] = max_bond


    quimb_mps = qtn.MatrixProductState.from_dense(\
                    qutip_ket.full(), qutip_ket.dims[0], \
                    **split_opts)

    return quimb_mps


convert_quimb_mp_to_qutip_qobj = quimb_mp_to_qutip_qobj
convert_qutip_ket_to_quimb_mps = qutip_ket_to_quimb_mps

if __name__ == '__main__':
    n_spins = 10

    dimensions = [[2]*n_spins]

    qutip_ket = qutip.rand_ket(dimensions[0])

    quimb_mps = qutip_ket_to_quimb_mps(qutip_ket)

    qutip_ket_reconstruct = quimb_mp_to_qutip_qobj(quimb_mps)

    fidelity = qutip.fidelity (qutip_ket, qutip_ket_reconstruct)
