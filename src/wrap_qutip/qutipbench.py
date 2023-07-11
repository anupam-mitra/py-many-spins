import qutip
import numpy as np

import itertools
import time

from numpy import pi

import spinsystems

# System parameters
nSpins = 18

# Solution parameters
nSteps = 256
nTraj = 1

# Hamiltonian model parameters
jInteraction = 1.0
bField = 1.0
axisInteraction = 'z'
axisMagneticField = 'x'

# Decoherence model parameters
gammaX = 0.01
gammaY = 0.01
gammaZ = 0.01
gammaPlus = 0.01
gammaMinus = 0.01

# Initial condition parameters
polarAngle = 0
azimuthalAngle = 0

# Duration parameters
timeInitial = 0
timeFinal = 2 * nSpins * np.pi

h = spinsystems.TransverseFieldIsingModelNearestNeighbor1D(
    jInteraction, bField, axisInteraction, axisMagneticField,
    boundarycondition='periodic')

d = spinsystems.SpinModelDecoherence(gammaPlus, gammaMinus, gammaZ)

sim = spinsystems.SpinDynamicsSimulation(timeInitial, timeFinal,
                                         nSpins, polarAngle, azimuthalAngle, nSteps, h, d)

print('@', time.strftime('%Y-%m-%d %T'))
print('Constructing initial state')
sim.construct_initial_state_qutip()

FLAG_SOLVE_SE = True
FLAG_SOLVE_MC = True
FLAG_SOLVE_ME = False

if FLAG_SOLVE_SE:
    print('Evolving the time dependent Schrodinger equation')
    timeStart = time.time()
    sim.solve_schroedingerequation(looped=False)
    timeEnd = time.time()
    print(
        'Qutip: Time dependent Schrodinger equation for %d spins took %g seconds'
        % (nSpins, timeEnd - timeStart))

if FLAG_SOLVE_MC:
    print('Evolving the monte carlo wavefunction')
    timeStart = time.time()
    sim.solve_montecarlo(n_trajectories=nTraj)
    timeEnd = time.time()
    print('Evolving Quantum Trajectories took %gs' % (timeEnd - timeStart))

if FLAG_SOLVE_ME:
    print('Evolving the master equation wavefunction')
    timeStart = time.time()
    sim.solve_masterequation()
    timeEnd = time.time()
    print('Evolving Master Equation took %gs' % (timeEnd - timeStart))

print('@', time.strftime('%Y-%m-%d %T'))

psi = qutip.rand_ket(2**nSpins, dims=[[2]*nSpins, [1]*nSpins])

# import sys
# sys.path.append('../conversions/')

# import convquimbqutip

# for threshold in np.logspace(-nSpins, 0, 2*nSpins+1, base=2):
#     print(threshold)
#     psi_mps = convquimbqutip.convert_qutip_ket_to_quimb_mps(psi, threshold)
#     psi_trunc = convquimbqutip.convert_quimb_mp_to_qutip_qobj(psi_mps)
#     print('\t', np.abs(psi_trunc.dag() * psi)**2)

# def ket_truncate(psi, threshold):
#     psi_mps = convquimbqutip.convert_qutip_ket_to_quimb_mps(psi, threshold)
#     psi_trunc = convquimbqutip.convert_quimb_mp_to_qutip_qobj(psi_mps)

#     return psi_trunc
