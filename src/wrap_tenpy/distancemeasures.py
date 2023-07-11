import numpy as np

import tenpy
import tenpy.linalg.np_conserved

import logging

def hilbertschmidt_distance(
        a:tenpy.linalg.np_conserved.Array, 
        b:tenpy.linalg.np_conserved.Array):
    """
    Calculates the Hilbert Schmidt distance
    between operators
    
    Parameters
    ----------
    a, b: tenpy.linalg.np_conserved.Array
        operators which are used to calculate the
        Hilbert Schmidt distance
    
    Returns
    -------
    hsd: float
        Hilbert Schmidt distance between the two
        input operators
    
    """
    logging.debug("squared Hilbert Schmidt distance for operators `a` and `b`")
    
    leg_labels_tensor_a:list = a.get_leg_labels()
    leg_labels_tensor_b:list = b.get_leg_labels()
    
    leg_labels_tensor_a.sort()
    leg_labels_tensor_b.sort()

    logging.debug("leg labels for `a`: %s" % (leg_labels_tensor_a,))
    logging.debug("leg labels for `b`: %s" % (leg_labels_tensor_b,))

    assert(leg_labels_tensor_a == leg_labels_tensor_b)
    
    leg_labels_tensor:list = leg_labels_tensor_a

    ket_leg_labels:list = [l for l in leg_labels_tensor if not l.endswith("*")]
    bra_leg_labels:list = [l for l in leg_labels_tensor if l.endswith("*")]

    ket_leg_labels.sort()
    bra_leg_labels.sort()

    logging.debug("ket leg labels: %s" % (ket_leg_labels,))
    logging.debug("bra leg labels: %s" % (bra_leg_labels,))
    
    n_sites:int = len(ket_leg_labels)

    logging.debug("n_sites = %d" % (n_sites))
  
    op_diff:tenpy.linalg.np_conserved.Array = (a - b)
    
    op_diff_sq:tenpy.linalg.np_conserved.Array = \
        tenpy.linalg.np_conserved.tensordot(
            op_diff, 
            op_diff, 
            axes=(bra_leg_labels, ket_leg_labels))

    tensor_traced = op_diff_sq

    for ix_site in range(n_sites):
        logging.debug("Contracting ket and bra indices for site %d" % (ix_site,))
        logging.debug("Before contraction, the tensor is of type: %s" % (type(tensor_traced),))
        logging.debug("Before contraction, the tensor has legs: %s" % (tensor_traced.get_leg_labels(),))
        
        tensor_traced = tenpy.linalg.np_conserved.trace(
            tensor_traced,
            leg1=ket_leg_labels[ix_site],
            leg2=bra_leg_labels[ix_site])

        logging.debug("After contraction, the tensor is of type: %s" % (type(tensor_traced),))
        logging.debug("After contraction, the tensor has legs: %s" %
                      (tensor_traced.get_leg_labels() if isinstance(tensor_traced, tenpy.linalg.np_conserved.Array) else None,)
                      )
        
    logging.debug("Taking real part remove the imaginary part")
    hsd_sq = np.real(tensor_traced)
    return hsd_sq
