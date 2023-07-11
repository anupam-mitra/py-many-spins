import quimb
import quimb.tensor

"""
The class `QuantumTrajectoriesMPSResult` stores the result of a quantum trajectories
calculation with matrix product states.
"""

class QuantumTrajectoriesResult:
    """
    Represents the results of an quantum trajectories
    calculation with Matrix Product States

    Parameters
    ----------
    nDof: number of degrees of freedom

    dims: dimensions of each degree of freedom

    errorTrotter: Trotter error

    errorDecimation: Decimation error

    numTrajectories: number of trajectories

    timesList: list of times at which the state is represented
    """

    pass


class SpinModelQuimb:
    '''
    Represents a spin model with quimb
    
    '''
    
    pass

'''
Loop over parameter values for a model family, and perform the calculation on the models

The parameters 
    1. Hamitonian parameters
    2. Decoherence parameters
    3. Initial condition parameters
    4. System size
    5. Temporal parameters

For each parameter value,
    - construct the model
    - solve the model
    - save the results
    - index the results
'''