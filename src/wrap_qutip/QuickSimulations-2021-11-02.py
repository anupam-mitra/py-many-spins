#!/usr/bin/env python
# coding: utf-8



import qutip
import numpy as np
import scipy

import itertools
import time

from numpy import pi

import os


import spinsystems
import manybodystateevolve


# System parameters
nSpins = 8

# Solution parameters
nSteps = 256
nTraj = 256

# Hamiltonian model parameters
jzz = +1.0
bx = +1.0 
bz = +1.0

# Decoherence model parameters
gammaX = 1e-2
gammaY = 1e-2
gammaZ = 1e-2 

# Initial condition parameters
polarAngle = pi/2
azimuthalAngle = pi/2

initial_ket_onebody = qutip.spin_coherent(j=1/2, theta=polarAngle, phi=azimuthalAngle, type='ket')
initial_ket = qutip.tensor([initial_ket_onebody] * nSpins)

#initial_ket = qutip.ket('01' * (nSpins//2))

# Duration parameters
timeInitial = 0
timeFinal = nSpins**2 / jzz

interact_graph = manybodystateevolve.Interaction1DNearest(nSpins, bc='open')

h = spinsystems.UniformTwoBodyInteraction(
    [(qutip.sigmaz(), qutip.sigmaz())],
    [qutip.sigmax(), qutip.sigmaz()],
    [jzz], [bx, bz],
    interact_graph)

d = spinsystems.UniformOneBodyDecoherence(
    [qutip.sigmaz(), qutip.sigmax(), qutip.sigmay()],
    [gammaZ, gammaX, gammaY])


sim = spinsystems.SpinDynamicsSimulation(timeInitial, timeFinal, nSpins, polarAngle, azimuthalAngle, nSteps, h, d)


print(time.strftime("%Y-%m-%d_%H:%M:%S"))
print("Constructing Hamiltonian")
hamiltonian_list = h.construct_hamiltonian_qutip(nSpins)
hamiltonian = sum(hamiltonian_list)

print(time.strftime("%Y-%m-%d_%H:%M:%S"))
print("Constructing jump operators")

jump_ops = d.construct_jumps_qutip(nSpins)

print(time.strftime("%Y-%m-%d_%H:%M:%S"))
print("Diagonalizing Hamiltonian")
hamiltonian_eigenvalues, hamittonian_eigenvectors \
    = hamiltonian.eigenstates()

print(time.strftime("%Y-%m-%d_%H:%M:%S"))
print("Constructing Liouvillians")
liouvillian = qutip.superoperator.liouvillian(hamiltonian, jump_ops)

print(time.strftime("%Y-%m-%d_%H:%M:%S"))
print("Diagonalizing Liouvillian")
liouvillian_eigenvalues, liouvillian_eigenvectors \
    = liouvillian.eigenstates(sparse=True,sort="high")

print(time.strftime("%Y-%m-%d_%H:%M:%S"))
print("Finished")


FLAG_SOLVE_SE = True
FLAG_SOLVE_MC = True
FLAG_SOLVE_ME = (nSpins <= 8)


# expm_states = [
#     (-1j * t * hamiltonian).expm() * initial_ket for t in np.linspace(timeInitial, timeFinal, nSteps)]

