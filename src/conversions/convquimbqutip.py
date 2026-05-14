import qutip
import quimb.tensor as qtn
import numpy as np
from typing import Any

 
def convert_quimb_mp_to_qutip_qobj(quimb_mp: Any) -> Any:
    """
    Convert a Quimb matrix-product state or operator to a QuTiP ``Qobj``.

    Parameters
    ----------
    quimb_mp : Any
        Representation as a quimb matrix product state
        or matrix product operator.

    Returns
    -------
    Any
        Representation as a QuTiP object.
    """
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
        dims = None

    qutip_qobj = qutip.Qobj(data, dims=dims)

    return qutip_qobj

 
def convert_qutip_ket_to_quimb_mps(
    qutip_ket: Any,
    cutoff: float | None = None,
    cutoff_mode: str = "sum2",
    max_bond: int | None = None,
) -> Any:
    """
    Convert a QuTiP ket to a Quimb matrix-product state.

    Parameters
    ----------
    qutip_ket : Any
        Representation as a QuTiP ket.
    cutoff : float | None, optional
        SVD cutoff, by default None.
    cutoff_mode : str, optional
        SVD cutoff mode, by default "sum2".
    max_bond : int | None, optional
        Maximum bond dimension, by default None.

    Returns
    -------
    Any
        Representation as a Quimb MPS.
    """

    split_opts = {}

    if cutoff is not None:
        split_opts["cutoff"] = cutoff
        split_opts["cutoff_mode"] = cutoff_mode

    if max_bond is not None:
        split_opts["max_bond"] = max_bond

    quimb_mps = qtn.MatrixProductState.from_dense(
        qutip_ket.full(),
        qutip_ket.dims[0],
        **split_opts,
    )

    return quimb_mps

