#!/usr/bin/env python
# coding: utf-8

# This notebook is used for quick simulations using `qutip`. The data generated from the simulations is used for many purposes.
# 
# 1. [DONE] Create plots to describe the difference between open and closed quantum systems
# 2. [DONE] Refactor classes to allow operators to be used for Hamiltonian and jump operators to be supplied as arguments. This would allow generalization to several model, in place of hard coding the model in a class.
# 4. [ ] Write a class for correlation length analysis.
# 5. [ ] Write a class that iterates over different model parameters.
# 5. [DONE] Moment calculation only supports a one-dimensional array of states. Change it to support a multi dimensional array of states.
# 
# 
# ## Test cases
# 1. Test cases for classes

# ## Quantifying trajectory sampling uncertainty

# ### Standard error of the mean method
# This method involves considering a marginal $\rho$ to be a vector of expectation values of a basis of operators.
# $$
# \rho \doteq
# \begin{pmatrix}
# \langle w_1 \rangle \\ \vdots \\ \langle w_{d^2 -1} \rangle
# \end{pmatrix}
# $$
# 
# 1. Construct the sample mean
# $$
# \overline{\rho} \doteq
# \begin{pmatrix}
# \overline{\langle w_1} \rangle \\ \vdots \\ \overline{\langle w_{d^2 -1}} \rangle
# \end{pmatrix},
# $$ 
# where the $j$-th component $\overline{\rho}_j = \overline{\langle w_j \rangle}$ is the sample mean of the expectation value of operator $w_j$
# $$
# \overline{\rho}_j 
# = \overline{\langle w_j \rangle}
# = \frac{1}{S} \sum_{s=1}^{S}
# \langle w_j \rangle_s
# $$
# 
# 2. Construct the sample covariance matrix whose $j,k$-th element is
# $$
# \left(\rho_{\mathrm{cov}}^2\right)_{j,k}
# = \frac{1}{S-1}
# \sum_{s=1}^{S} \left(
# \overline{\langle w_j \rangle} - \langle w_j \rangle_s
# \right) \left(
# \overline{\langle w_k \rangle} - \langle w_k \rangle_s
# \right)^\top.
# $$
# 
# 3. The quantifier for sampling uncertainty is
# $$
# \text{Tr} \left(\rho_{\mathrm{cov}}^2 \right)
# = \left\Vert\rho_{\mathrm{cov}}\right\Vert_2^2.
# $$

# ## Convergence by adding a sample method
# This method involves evaluating the sampling mean using $S$ samples and comparing it with the sample mean using $S+1$ samples. It is useful use the samples used to estimate a sample mean $\overline{\rho}$ as argument, for example, $\overline{\left(\rho\right)_{1, \cdots, S}}$.
# 
# 1. Construct sample means $\overline{\rho_{1, \cdots, S}}$ and $\overline{\rho_{1, \cdots, S+1}}$ whose $j$-th components read
# $$
# \left(\overline{\rho_{1, \cdots, S}}\right)_j 
# = \overline{\langle w_j \rangle}
# = \frac{1}{S} \sum_{s=1}^{S}
# \langle w_j \rangle_s
# $$
# and
# $$
# \left(\overline{\rho_{1, \cdots, S+1}}\right)_j 
# = \overline{\langle w_j \rangle}
# = \frac{1}{S+1} \sum_{s=1}^{S+1}
# \langle w_j \rangle_s
# $$
# respectively.
# 
# 2. The quantifier for sampling uncertainty is the component-wise squared difference between  $\overline{\rho_{1, \cdots, S}}$ and $\overline{\rho_{1, \cdots, S+1}}$
# $$
# \sum_{j=1}^{d^2-1} 
# \left(
# \left(\overline{\rho_{1, \cdots, S+1}}\right)_j -
# \left(\overline{\rho_{1, \cdots, S}}\right)_j 
# \right)^2
# = \mathcal{D}_{\text{HS}}^2
# \left(
# \overline{\rho_{1, \cdots, S}},
# \overline{\rho_{1, \cdots, S+1}}
# \right)
# $$
# which also happens to be the squared Hilbert-Schmidt distance between density operators $\overline{\rho_{1, \cdots, S}}$ and $\overline{\rho_{1, \cdots, S+1}}$.

# Schema for storing marginals over trajectories
# 
# | ixTrajectory | ixTime | ixMarginal | spinsSelected | marginalMatrix | marginalVector |
# | --- | --- | --- | --- | --- | --- |
# |. |. |. |. |.  |.  |
# 
# Schema for storing marginals over time
# 
# | ixTrajectory | ixTime | ixMarginal | spinsSelected | marginalMatrix | marginalVector |
# | --- | --- | --- | --- | --- | --- |
# |. |. |. |. |.  |.  |
# 
# 
# 

# In[58]:


import qutip
import numpy as np
import scipy

import itertools
import time

from numpy import pi

import os
import autoreload


# In[2]:


import spinsystems
import manybodystateevolve

# In[3]:

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
if bx/jzz >= 4.0:
    timeFinal = 10 * 2 * np.pi / bx
else:
    timeFinal = 10 * 2 * np.pi / jzz


# In[4]:


interact_graph = manybodystateevolve.Interaction1DNearest(nSpins, bc='open')

# In[5]:
FLAG_SOLVE_SE = True
FLAG_SOLVE_MC = True
FLAG_SOLVE_ME = (nSpins <= 8)


# # Calculations

# ## Setting up the simulation

# In[6]:


h = spinsystems.UniformTwoBodyInteraction(
    [(qutip.sigmaz(), qutip.sigmaz())],
    [qutip.sigmax(), qutip.sigmaz()],
    [jzz], [bx, bz],
    interact_graph)

d = spinsystems.UniformOneBodyDecoherence(
    [qutip.sigmaz(), qutip.sigmax(), qutip.sigmay()],
    [gammaZ, gammaX, gammaY])

sim = spinsystems.SpinDynamicsSimulation(timeInitial, timeFinal, nSpins, polarAngle, azimuthalAngle, nSteps, h, d)


# In[7]:


hamiltonian_list = h.construct_hamiltonian_qutip(nSpins)
hamiltonian = sum(hamiltonian_list)
energies = hamiltonian.eigenenergies()


# In[8]:


energy_gaps = np.diff(energies)
energy_gaps_sorted = np.sort(energy_gaps)

np.min(energy_gaps)


# ## Solving for the state
# 1. closed quantum system using `qutip.scsolve`
# 2. open quantum system using `qutip.mcsolve`
# 3. open quantum system using `qutip.mesolve`

# In[9]:


expm_states = [
    (-1j * t * hamiltonian).expm() * initial_ket for t in np.linspace(timeInitial, timeFinal, nSteps)]


# In[10]:


if FLAG_SOLVE_SE:
    sesolver = manybodystateevolve.QutipSESolve(timeInitial, timeFinal, nSteps, h, nSpins, initial_ket)
    sesolver.run()


# In[11]:


if FLAG_SOLVE_MC:
    mcsolver = manybodystateevolve.QutipMCSolve(timeInitial, timeFinal, nSteps, h, d, nSpins, initial_ket, nTraj)
    mcsolver.run()


# In[12]:


if FLAG_SOLVE_ME:
    initial_dm = qutip.ket2dm(initial_ket)

    mesolver = manybodystateevolve.QutipMESolve(timeInitial, timeFinal, nSteps, h, d, nSpins, initial_dm)

    mesolver.run()


# In[13]:


overlaps = np.empty((nSteps))
fidelities = np.empty((nSteps))

for s in range(nSteps):
    overlaps[s] = np.abs((expm_states[s].dag() * sesolver.states[s])[0, 0])
    fidelities[s] = qutip.fidelity(expm_states[s], sesolver.states[s])


# In[139]:


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 10, 'text.usetex': False})


# In[135]:


def select_central_sites(n, m):
    '''
    Selects the `m` central contiguous sites
    from `n sites labeled `0, 1, ..., (n-1)`.

    In an even system size, that is when `n % 2 == 0`,
    the sites chosen are the smaller ones, that is closer
    to the left edge at `0` than to the right edge at `n-1`
    '''

    locations = tuple(range((n - m + 1)//2, (n + m + 1)//2))

    return locations


# In[17]:


# It maybe useful to use a class for this.
# Using an object of a class will enable bookkeeping
# about which operator represented as `qutip.qobj.Qobj`
# using a string to describe it
def gen_pauli_string_operators(m, pauli_operators=None):
    """
    Generates Pauli string operators for `m`
    spin-1/2 degrees of freedom
    """
    
    if pauli_operators == None:
        sigma0 = qutip.identity(2)
        sigmax = qutip.sigmax()
        sigmay = qutip.sigmay()
        sigmaz = qutip.sigmaz()
        
        
    pauli_string_operators = [qutip.tensor(*string) 
           for string in itertools.product(*[[sigma0, sigmax, sigmay, sigmaz]]*m)]
    
    return pauli_string_operators
    


# In[18]:


marginal_size = 4
pauli_string_operators = gen_pauli_string_operators(marginal_size)
locations = select_central_sites(nSpins, marginal_size)


# In[19]:


pauli_string_expectations_se = np.empty((nSteps, len(pauli_string_operators)))

for ixTime in range(nSteps):

    ρ = qutip.ptrace(sesolver.states[ixTime], locations)

    for ixPauli in range(len(pauli_string_operators)):
        w = pauli_string_operators[ixPauli]
        pauli_string_expectations_se[ixTime, ixPauli] = qutip.expect(w, ρ)


# In[20]:


pauli_string_expectations_me = np.empty((nSteps, len(pauli_string_operators)))

for ixTime in range(nSteps):

    ρ = qutip.ptrace(mesolver.states[ixTime], locations)

    for ixPauli in range(len(pauli_string_operators)):
        w = pauli_string_operators[ixPauli]
        pauli_string_expectations_me[ixTime, ixPauli] = qutip.expect(w, ρ)


# In[21]:


pauli_string_expectations_mc = np.empty((nTraj, nSteps, len(pauli_string_operators)))

for ixTime in range(nSteps):
    for ixTraj in range(nTraj):

        ρ = qutip.ptrace(mcsolver.states[ixTraj][ixTime], locations)
                                        
        for ixPauli in range(len(pauli_string_operators)):
            w = pauli_string_operators[ixPauli]
            pauli_string_expectations_mc[ixTraj, ixTime, ixPauli] = qutip.expect(w, ρ)

        
pauli_string_expectations_mc_mean = np.mean(pauli_string_expectations_mc, axis=0)
pauli_string_expectations_mc_sem = np.std(pauli_string_expectations_mc, axis=0) / (np.sqrt(nTraj-1))


# In[22]:


# In[24]:


def expectations_sem (samples):
    
    nExpectations, nSamples = samples.shape
    expectations_cov = np.cov(samples)
    
    sem = np.trace(expectations_cov) / (nSamples - 1)

    return sem


# In[25]:


hsdistance_mc_me_time =     np.sum((pauli_string_expectations_me - pauli_string_expectations_mc_mean)**2, axis=1)



# In[26]:


mc_sem_time = np.asarray([    expectations_sem(pauli_string_expectations_mc[:, ixTime, :].T)         for ixTime in range(nSteps)
    ])


# In[28]:


mc_sem_traj_time = np.empty((nTraj-1, nSteps))

for nTrajUsed in range(nTraj):
    for ixTime in range(nSteps):
        mc_sem_traj_time[nTrajUsed - 1, ixTime] =             expectations_sem(                pauli_string_expectations_mc[:nTrajUsed+2, ixTime, :].T)
        


# In[29]:


pauli_string_expectations_mc_cummean =     np.empty((nTraj, nSteps, len(pauli_string_operators)))

for ixTraj in range(nTraj):
    pauli_string_expectations_mc_cummean[ixTraj, :, :] =         np.mean(pauli_string_expectations_mc[:ixTraj+1, :, :], axis=0)

pauli_string_expectations_mc_cummean_squared_difference =     np.empty((nTraj-1, nSteps, len(pauli_string_operators)))

for ixTraj in range(nTraj-1):
    pauli_string_expectations_mc_cummean_squared_difference[ixTraj, :, :] =         (pauli_string_expectations_mc_cummean[ixTraj, :] -          pauli_string_expectations_mc_cummean[ixTraj+1, :, :])**2

pauli_string_expectations_me_mc_cummean_squared_difference =     np.empty((nTraj, nSteps, len(pauli_string_operators)))

for ixTraj in range(nTraj):
    pauli_string_expectations_me_mc_cummean_squared_difference[ixTraj, :, :] =         (pauli_string_expectations_me[:, :] -          pauli_string_expectations_mc_cummean[ixTraj, :, :])**2


# In[31]:


DATE_STRING = time.strftime('%Y-%m-%d_')
DATE_STRING


# In[141]:


hsd_latex = r'$\mathcal{D}_{\mathrm{HS}}^2$ '

for ixTime in [6, 8, 10, 12, 16, 32, 64, 128, 255]:

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    ax.plot(range(2, nTraj+1),
        np.sum(pauli_string_expectations_mc_cummean_squared_difference[:, ixTime, :], axis=-1),
           label=hsd_latex + \
            r'$(\overline{\rho_{1, \cdots, S-1}}, \overline{\rho_{1, \cdots, S}})$')

    ax.plot(range(2, nTraj+1), mc_sem_traj_time[:, ixTime],
           label=r'$\Vert \rho_{\mathrm{cov}} \Vert_2^2$')

    ax.plot(range(1, nTraj+1), 
            np.sum(pauli_string_expectations_me_mc_cummean_squared_difference[:, ixTime, :], axis=-1),
           label= hsd_latex + r'$(\overline{\rho_{1, \cdots, S}}, \rho_{\mathrm{master}} )$')


    ax.plot(range(2, nTraj+1), 1/np.arange(2, nTraj+1),
            label=r'$10^{-1} S^{-1}$', ls='dashed', color='gray')

    ax.plot(range(2, nTraj+1), 1/np.arange(2, nTraj+1)**2,
            label=r'$10^{-1} S^{-2}$', ls='dashdot', color='gray')

    ax.set_yscale('log')
    ax.legend(frameon=False, ncol=2)#loc=(1.05,0.5))

    ax.set_title(r'$\rho^{%s}$ of $\rho^{%s}$' % (locations, tuple(range(nSpins)),) +                  '\n' +                  'at $Jt = %g$ or $\Gamma t = %g$ or $3N\Gamma t = %g$' %
                   (sesolver.t_list[ixTime],
                    sesolver.t_list[ixTime] * gammaX,
                    sesolver.t_list[ixTime] * gammaX * 3 * nSpins,))

    ax.set_xlabel(r'Number of samples, $S$')
    ax.set_ylabel('Squared sampling error estimate')

    plt.tight_layout()
    plotfilename = DATE_STRING + '%d-SpinMarginalSamplingErrorEstimate_%03d' % \
        (marginal_size, ixTime)
    plt.savefig(plotfilename + '.pdf')
    print('Saved to %s' % plotfilename)




