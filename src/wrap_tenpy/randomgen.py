import numpy as np

import tenpy
import tenpy.networks.site
import tenpy.linalg.np_conserved

def gen_random_spinhalf_MPS(n_spins:int, dtype=float):
    """
    Generate a random MPS for a 1d array
    of spin-half degrees of freedom
    Parameters
    ----------
    n_spins: int
        Number of spins
    dtype: type
        Type of the entries of the matrix product state.
        Currently only `float` is supported.
    Returns
    -------
    mps:
        Randomly generated matrix product state
    """
    logging.info("Creating sites")
    site:tenpy.networks.site.SpinHalfSite \
        = tenpy.networks.site.SpinHalfSite(conserve=None)
    
    logging.info("Generating random vector using numpy")
    psi:np.ndarray \
        = np.random.randn(2**n_spins)
    
    logging.info("Normalizing the random vector to a 2-norm of 1")
    psi /= np.linalg.norm(psi)

    psi = psi.reshape((2,)*n_spins)
    logging.info("Creating charge info")
    chargeinfo:tenpy.linalg.np_conserved.charges \
        = tenpy.linalg.np_conserved.charges.ChargeInfo([1], ["2*Sz"])
    
    logging.info("Creating physical leg from charge info")
    p_leg:tenpy.linalg.np_conserved.LegCharge \
        = tenpy.linalg.np_conserved.LegCharge.from_qflat(chargeinfo, [[1],␣ ↪[-1]])
    
    logging.info("Creating npc.Array object with physical legs")
    psi_npc:tenpy.linalg.np_conserved.Array \
        = tenpy.linalg.np_conserved.Array.from_ndarray_trivial(psi, labels=['p%d' % l for l in range(n_spins)])
    
    logging.info("Creating MPS from the npc.Array object")

    mps:tenpy.networks.mps.MPS \
        = tenpy.networks.mps.MPS.from_full([site]*n_spins, psi_npc)
    return mps
