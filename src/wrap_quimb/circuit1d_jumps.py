import quimb as qu
import quimb.tensor as qtn

import numpy as np
import pickle
import uuid
import time
import scipy
import scipy.linalg

import itertools


from numpy import pi, cos, sin, sqrt, exp

import sys
sys.path.append('.')

import circuit1d

class TrotterizedJumps1D (circuit1d.AlternateOneQubitTwoQubit1D):
    '''
    Represents a circuit with alternate layers
    of one qubit gates and two qubit gates with quantum jumps
    described by one qubit jumps
    
    Parameters
    ----------
    
    onequbitgate: one qubit gate
    
    twoqubitgate: two qubit gate

    jump_ops: jump operators

    

    '''
    pass