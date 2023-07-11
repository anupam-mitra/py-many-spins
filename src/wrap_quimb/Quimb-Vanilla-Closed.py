import quimb as qu
import numpy as np

import time


class Dynamics:
    '''
    Represents a quantum dynamics problem. Different
    backend solvers can be used to calculate the dynamics.

    Parameters
    ----------

    hamiltonian: hamiltonian

    jumpOps: jump operators

    timeInitial: initial time

    timeFinal: final time

    stateInitial: initial state

    timeList: list of times at which the state is to be calculated
    '''
    pass


n_spins = 14

ket_initial = qu.ket(data=[[1] + [0]*(2**(n_spins)-1)])

hamiltonian = \
    sum([1/2 * \
         qu.ikron(qu.pauli('X'), dims=[2]*n_spins, inds=[n, n+1]) 
         for n in range(n_spins-1)])

evolver = qu.Evolution(ket_initial, hamiltonian)

n_steps = 1024
t_initial = 0
t_final = np.pi * 32

t_list = np.arange(t_initial, t_final, n_steps)

while True:

    time_start = time.time()
    ket_t = evolver.at_times(t_list)
    ket_t = list(ket_t)
    time_finish = time.time()
    print('# Time taken for ket tdse with %d spins = %g s' % (n_spins, (time_finish - time_start)))

    if n_spins <= 12:
        time_start = time.time()
        ket_t = evolver.at_times(t_list)
        ket_t = list(ket_t)
        time_finish = time.time()
        print('# Time taken for dm tdse %d spins = %g s' % (n_spins, (time_finish - time_start)))
