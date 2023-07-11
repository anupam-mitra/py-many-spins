import quimb
import numpy as np

def hs_distance_mpo(rho1mpo, rho2mpo):
    
    return float("nan")
    

def hs_distance(rho1, rho2):
    rho1_normalized = rho1 / quimb.tr(rho1)
    rho2_normalized = rho2 / quimb.tr(rho2)
    
    rho_diff = rho1_normalized - rho2_normalized
    
    hsd = quimb.vdot(rho_diff, rho_diff)
    
    return hsd

def trace_distance(rho1, rho2):
    assert quimb.isherm(rho1)
    assert quimb.isherm(rho2)
    
    assert quimb.ispos(rho1)
    assert quimb.ispos(rho2)
    
    rho1_normalized = rho1 / quimb.tr(rho1)
    rho2_normalized = rho2 / quimb.tr(rho2)
      
    trd = quimb.trace_distance(rho1_normalized, rho2_normalized)
    
    return trd

def one_minus_sqrtfidelity (rho1, rho2):
    rho1_normalized = rho1 / quimb.tr(rho1)
    rho2_normalized = rho2 / quimb.tr(rho2)
    
    f = np.real(quimb.fidelity(rho1_normalized, rho2_normalized, squared=False))
    
    if f >= 1:
        f = 1
    return 1 - f

def sqrt_one_minus_fidelity (rho1, rho2):
    
    rho1_normalized = rho1 / quimb.tr(rho1)
    rho2_normalized = rho2 / quimb.tr(rho2)
    
    f = quimb.fidelity(rho1_normalized, rho2_normalized, squared=True)
    if f >= 1:
        f = 1
    return np.sqrt(1 - f)