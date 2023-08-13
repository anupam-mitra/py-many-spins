import qutip
import quimb
import quimb.tensor
import numpy as np


def convert_quimb_mp_to_qutip_qobj (quimb_mp):
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

    if isinstance(quimb_mp, quimb.tensor.tensor_1d.MatrixProductState):
        dims = [[quimb_mp.phys_dim()] * quimb_mp.L, \
                [1] * quimb_mp.L]
        shape = (quimb_mp.phys_dim()** quimb_mp.L, 1)
    elif isinstance(quimb_mp, quimb.tensor.tensor_1d.MatrixProductOperator):
        dims = [[quimb_mp.phys_dim()] * quimb_mp.L, \
                [quimb_mp.phys_dim()] * quimb_mp.L]
        shape = (quimb_mp.phys_dim()** quimb_mp.L, quimb_mp.phys_dim()** quimb_mp.L)
        data = np.asarray(data)
    else:
        dims=None
        shape=None

    qutip_qobj = qutip.qobj.Qobj(data, dims=dims, shape=shape)

    return qutip_qobj

def convert_qutip_ket_to_quimb_mps (qutip_ket, cutoff=None, cutoff_mode='sum2', max_bond=None):
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

    quimb_mps = quimb.tensor.tensor_1d.MatrixProductState.from_dense(\
                    qutip_ket.data, qutip_ket.dims[0], \
                    **split_opts)

    return quimb_mps
