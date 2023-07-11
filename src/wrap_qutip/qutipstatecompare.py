import qutip
import numpy as np
import math
import cmath
import itertools

import pandas as pd
import sqlite3
import pickle
import os
import time

from numpy import pi, sqrt

import quimb
import quimb.tensor

import sys
sys.path.append('.')
sys.path.append('..')


''' Creating the index '''
sql_create_table_hamiltonians = '''
    CREATE TABLE IF NOT EXISTS
    Hamiltonians
    (
        uuid TEXT,
        description TEXT
    )
'''


jInteraction = 1
bMagField = jInteraction
n_spins = 8

tInitial = 0
tFinal = 2 * pi  / jInteraction * n_spins

n_steps = 64

tlist = np.linspace(tInitial, tFinal, n_steps)

# Hamiltonian
sigmaz = qutip.sigmaz()
sigmax = qutip.sigmax()

qutip_ham_int_list = [jInteraction * \
    qutip.qip.operations.expand_operator(qutip.tensor(sigmaz, sigmaz), n_spins, targets=(n, n+1)) \
    for n in range(n_spins-1)]

# qutip_ham_int_list = [jInteraction * 1/(n1-n2)**2 * \
#                           qutip.qip.operations.expand_operator(qutip.tensor(sigmaz, sigmaz), n_spins, targets=(n2, n1)) \
#                           for n1, n2 in itertools.combinations(range(n_spins), 2)]

qutip_ham_int_list = [jInteraction * \
    qutip.qip.operations.expand_operator(qutip.tensor(sigmaz, sigmaz), n_spins, targets=(na, nb)) \
    for na, nb in itertools.combinations(range(n_spins), 2)]


qutip_ham_magfield_x_list = [bMagField * \
    qutip.qip.operations.expand_operator(sigmax, n_spins, targets=(n,)) \
        for n in range(n_spins)] 

qutip_ham_magfield_z_list = [bMagField * \
    qutip.qip.operations.expand_operator(sigmaz, n_spins, targets=(n,)) \
        for n in range(n_spins)] 

qutip_ham_magfield_list = qutip_ham_magfield_x_list

def qutip_dynamics_se_setup ():
    '''
    Prepares data structure for a simulation of closed system using `qutip.sesolve`,
    which solves the time dependent Schroedinger equation

    Parameters
    ----------

    Returns
    -------
    hamiltonian:
        Hamiltonian terms represented as `qutip.qobj.Qobj` or an iterable containing
        terms of the type `qutip.qobj.Qobj`
    


    '''
    pass

def qutip_dynamics_se_run (hamiltonians, tlist, state_initial, \
        callback=None, callback_args=None):
    '''
    Runs a simulation for a closed system using the time dependent Schroedinger equation

    Parameters
    ----------

    hamiltonians: 
        Hamiltonian terms represented as `qutip.qobj.Qobj` or an iterable containing
        terms of the type `qutip.qobj.Qobj`

    tlist:
        List of time instants at which to calculate the state or unitary

    state_initial:
        The initial state represented as an object of type `qutip.qobj.Qobj`

    callback:
        Function to be called on the state at every instant during the time evolution

    callback_args:
        Arguments to be passed to the callback function
    '''

    results_se = qutip.sesolve(hamiltonians, \
        state_initial, tlist=tlist)

    return results_se


# Initial state
qutip_ket_initial = qutip.ket('0'*n_spins)

# Calculate the dynamics for the closed quantum system
print('@ %s' % time.strftime('%Y-%m-%d %T'))
print('Solving Schroedinger equation with no truncation, using a single `qutip.sesolve`')
qutip_results_se = qutip.sesolve(qutip_ham_int_list + qutip_ham_magfield_list, \
    qutip_ket_initial, tlist=tlist) 

# Calculate the dynamics looped over time

import conversions
import conversions.convquimbqutip as convquimbqutip

def ket_truncate(psi, cutoff=None, cutoff_mode=None, max_bond=None):

    psi_mps = convquimbqutip.convert_qutip_ket_to_quimb_mps(psi, \
        cutoff=cutoff, cutoff_mode=cutoff_mode, max_bond=max_bond)
    psi_trunc = convquimbqutip.convert_quimb_mp_to_qutip_qobj(psi_mps)

    return psi_trunc

callback = ket_truncate

states_single_sesolve = np.empty_like(tlist, dtype=object)

for ix_t in range(n_steps):
    states_single_sesolve[ix_t] = qutip_results_se.states[ix_t]

hamiltonians = qutip_ham_int_list + qutip_ham_magfield_list
resultsc = None

print('@ %s' % time.strftime('%Y-%m-%d %T'))
print('Solving Schroedinger equation with truncation, with multiple calls to `qutip.sesolve')

bonddim_vals = np.logspace(0, n_spins//2, num=n_spins//2+1, base=2)

states_multi_sesolve = np.empty((len(bonddim_vals), len(tlist)), dtype=object)

for ix_bonddim, bonddim in enumerate(bonddim_vals):
    states_multi_sesolve[ix_bonddim, 0] = qutip_ket_initial

    for ix_t in range(1, n_steps):
        
        resultsc = qutip.sesolve(hamiltonians, \
            states_multi_sesolve[ix_bonddim, ix_t-1], \
            [tlist[ix_t-1], tlist[ix_t]])

        state_new_pre = resultsc.states[-1]

        if callback != None:
            state_new = callback(state_new_pre, max_bond=bonddim)

        else:
            state_new = state_new_pre

        states_multi_sesolve[ix_bonddim, ix_t] = state_new

FLAG_CALC_INFIDELITY = True

if FLAG_CALC_INFIDELITY:
    print('@ %s' % time.strftime('%Y-%m-%d %T'))
    print('Calculating inner products for fidelity')
    inner_products = np.ones_like(states_multi_sesolve)

    for ix_t in range(0, n_steps):
        psi_ref = states_single_sesolve[ix_t]
        #psi_ref = states_multi_sesolve[0, ix_t]
        
        for ix_bonddim, bonddim in enumerate(bonddim_vals):

            psi = states_multi_sesolve[ix_bonddim, ix_t]

            inner_products[ix_bonddim, ix_t] = \
                np.abs(psi.dag() * psi_ref)[0, 0]**2
                


def qutip_create_manybodyoperators (terms, n_sites):
    '''
    Creates many body operators for spin 1/2 as objects of type `qutip.qobj.Qobj`
    from terms describing the operator and the location

    NOTE: This calculation can be cached by making an object for this.

    Parameters
    ----------
    terms:
        Iterable of tuples of the form (operator, site). For example [('sigmaz', 0), ('sigmaz', 1)]

    n_sites:
        Number of sites in the manybody system

    Returns
    -------
    operator:
        Many body operator represented as an object of type `qutip.qobj.Qobj`
    '''

    operator_list = []
    site_list = []

    for term in terms:
        operator_name = term[0]
        site = term[1]

        site_list.append(site)

        if operator_name.lower() == 'sigmax' or operator_name.lower() == 'x':
            operator = qutip.sigmax()
        elif operator_name.lower() == 'sigmay' or operator_name.lower() == 'y':
            operator = qutip.sigmay()
        elif operator_name.lower() == 'sigmaz' or operator_name.lower() == 'z':
            operator = qutip.sigmaz()
        elif operator_name.lower() == 'sigmap' or operator_name.lower() == 'sigmaplus':
            operator = qutip.sigmap()
        elif operator_name.lower() == 'sigmam' or operator_name.lower() == 'sigmaminus':
            operator = qutip.sigmam()
        elif operator_name.lower() == 'identity' or operator_name.lower() == 'id':
            operator = qutip.qeye(2)
        else:
            raise ValueError('%s: operator is not implemented', operator_name)

        operator_list.append(operator)

    operator_nontrivialsites = qutip.tensor(operator_list)

    operator = qutip.qip.operations.expand_operator(operator_nontrivialsites, n_sites, \
        targets=site_list)

    return operator

def qutip_calc_moments (state, n_sites, terms):
    '''
    Calculates moments 

    Parameters
    ----------
    state:
        State used for calculation of moments

    n_sites:
        Number of sites in the manybody state

    terms:
        Iterable of tuples of the form (operator, site). For example [('sigmaz', 0), ('sigmaz'), 1]

    '''
    operator =  qutip_create_manybodyoperators(terms, n_sites)

    moment = qutip.expect(operator, state)

    return moment


print('@ %s' % time.strftime('%Y-%m-%d %T'))
print('Calculating moments')

FLAG_CALC_NN = False
FLAG_CALC_NNN = False

if FLAG_CALC_NN:
    xx_nn_ref = np.ones_like(states_single_sesolve)
    xy_nn_ref = np.ones_like(states_single_sesolve)
    xz_nn_ref = np.ones_like(states_single_sesolve)

    yx_nn_ref = np.ones_like(states_single_sesolve)
    yy_nn_ref = np.ones_like(states_single_sesolve)
    yz_nn_ref = np.ones_like(states_single_sesolve)

    zx_nn_ref = np.ones_like(states_single_sesolve)
    zy_nn_ref = np.ones_like(states_single_sesolve)
    zz_nn_ref = np.ones_like(states_single_sesolve)

    xx_nn = np.ones_like(states_multi_sesolve)
    xy_nn = np.ones_like(states_multi_sesolve)
    xz_nn = np.ones_like(states_multi_sesolve)

    yx_nn = np.ones_like(states_multi_sesolve)
    yy_nn = np.ones_like(states_multi_sesolve)
    yz_nn = np.ones_like(states_multi_sesolve)

    zx_nn = np.ones_like(states_multi_sesolve)
    zy_nn = np.ones_like(states_multi_sesolve)
    zz_nn = np.ones_like(states_multi_sesolve)

    for ix_t in range(1, n_steps):

        psi_ref = states_single_sesolve[ix_t]

        # Nearest neighbors

        xx_nn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('x', la), ('x', lb)])
        xy_nn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('x', la), ('y', lb)])
        xy_nn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('x', la), ('z', lb)])

        yx_nn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('y', la), ('x', lb)])
        yy_nn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('y', la), ('y', lb)])
        yy_nn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('y', la), ('z', lb)])

        zx_nn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('z', la), ('x', lb)])
        zy_nn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('z', la), ('y', lb)])
        zy_nn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('z', la), ('z', lb)])

if FLAG_CALC_NNN:
    xx_nnn_ref = np.ones_like(states_single_sesolve)
    xy_nnn_ref = np.ones_like(states_single_sesolve)
    xz_nnn_ref = np.ones_like(states_single_sesolve)

    yx_nnn_ref = np.ones_like(states_single_sesolve)
    yy_nnn_ref = np.ones_like(states_single_sesolve)
    yz_nnn_ref = np.ones_like(states_single_sesolve)

    zx_nnn_ref = np.ones_like(states_single_sesolve)
    zy_nnn_ref = np.ones_like(states_single_sesolve)
    zz_nnn_ref = np.ones_like(states_single_sesolve)

    xx_nnn = np.ones_like(states_multi_sesolve)
    xy_nnn = np.ones_like(states_multi_sesolve)
    xz_nnn = np.ones_like(states_multi_sesolve)

    yx_nnn = np.ones_like(states_multi_sesolve)
    yy_nnn = np.ones_like(states_multi_sesolve)
    yz_nnn = np.ones_like(states_multi_sesolve)

    zx_nnn = np.ones_like(states_multi_sesolve)
    zy_nnn = np.ones_like(states_multi_sesolve)
    zz_nnn = np.ones_like(states_multi_sesolve)

    la = n_spins//2-1
    lb = n_spins//2

    for ix_t in range(1, n_steps):

        psi_ref = states_single_sesolve[ix_t]

        # Next to nearest neighbors

        xx_nnn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('x', la), ('x', lb+1)])
        xy_nnn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('x', la), ('y', lb+1)])
        xy_nnn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('x', la), ('z', lb+1)])

        yx_nnn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('y', la), ('x', lb+1)])
        yy_nnn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('y', la), ('y', lb+1)])
        yy_nnn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('y', la), ('z', lb+1)])

        zx_nnn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('z', la), ('x', lb+1)])
        zy_nnn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('z', la), ('y', lb+1)])
        zy_nnn_ref[ix_t] = qutip_calc_moments(psi_ref, n_spins, \
            [('z', la), ('z', lb+1)])


        for ix_bonddim, bonddim in enumerate(bonddim_vals):

            psi = states_multi_sesolve[ix_bonddim, ix_t]

            xx_nnn[ix_bonddim, ix_t] = qutip_calc_moments(psi, n_spins, \
                [('x', la), ('x', lb+1)])
            xy_nnn[ix_bonddim, ix_t] = qutip_calc_moments(psi, n_spins, \
                [('x', la), ('y', lb+1)])
            xy_nnn[ix_bonddim, ix_t] = qutip_calc_moments(psi, n_spins, \
                [('x', la), ('z', lb+1)])

            yx_nnn[ix_bonddim, ix_t] = qutip_calc_moments(psi, n_spins, \
                [('y', la), ('x', lb+1)])
            yy_nnn[ix_bonddim, ix_t] = qutip_calc_moments(psi, n_spins, \
                [('y', la), ('y', lb+1)])
            yy_nnn[ix_bonddim, ix_t] = qutip_calc_moments(psi, n_spins, \
                [('y', la), ('z', lb+1)])

            zx_nnn[ix_bonddim, ix_t] = qutip_calc_moments(psi, n_spins, \
                [('z', la), ('x', lb+1)])
            zy_nnn[ix_bonddim, ix_t] = qutip_calc_moments(psi, n_spins, \
                [('z', la), ('y', lb+1)])
            zy_nnn[ix_bonddim, ix_t] = qutip_calc_moments(psi, n_spins, \
                [('z', la), ('z', lb+1)])

print('@ %s' % time.strftime('%Y-%m-%d %T'))
print('Plotting')
    
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 10


if FLAG_CALC_INFIDELITY:
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for ix_bonddim, bonddim in enumerate(bonddim_vals):
        ax.plot(tlist/2/pi, 1-inner_products[ix_bonddim, :], label=r'$\chi=%d$' % (bonddim))

    ax.grid()
    ax.legend()
    ax.set_xlabel(r'$Jt/(2\pi)$')
    ax.set_ylabel(r'InFidelity')

    prefix = os.path.join('../_data_/', time.strftime('%Y-%m-%d_%H-%M-%S'))
    plt.savefig(prefix + '_qutip_tfim_%d_spins.pdf' % (n_spins))


axes_ylabels = np.asarray([\
               (r'$\langle XX\rangle $', r'$\langle XY\rangle $', r'$\langle XZ\rangle $'), \
               (r'$\langle YX\rangle $', r'$\langle YY\rangle $', r'$\langle YZ\rangle $'), \
               (r'$\langle ZX\rangle $', r'$\langle ZY\rangle $', r'$\langle ZY\rangle $')])

# Nearest neighbor plots
if FLAG_CALC_NN:
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    ((ax_nn_xx, ax_nn_xy, ax_nn_xz), \
        (ax_nn_yx, ax_nn_yy, ax_nn_yz), \
        (ax_nn_zx, ax_nn_zy, ax_nn_zz)) = axes


    for ix_bonddim, bonddim in enumerate(bonddim_vals):
        ax_nn_xx.plot(tlist/2/pi, xx_nn[ix_bonddim, :] - xx_nn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nn_xy.plot(tlist/2/pi, xy_nn[ix_bonddim, :] - xy_nn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nn_xz.plot(tlist/2/pi, xz_nn[ix_bonddim, :] - xz_nn_ref, \
            label=r'$\chi=%d$' % (bonddim))

        ax_nn_yx.plot(tlist/2/pi, yx_nn[ix_bonddim, :] - yx_nn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nn_yy.plot(tlist/2/pi, yy_nn[ix_bonddim, :] - yy_nn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nn_yz.plot(tlist/2/pi, yz_nn[ix_bonddim, :] - yz_nn_ref, \
            label=r'$\chi=%d$' % (bonddim))

        ax_nn_zx.plot(tlist/2/pi, zx_nn[ix_bonddim, :] - zx_nn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nn_zy.plot(tlist/2/pi, zy_nn[ix_bonddim, :] - zy_nn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nn_zz.plot(tlist/2/pi, zz_nn[ix_bonddim, :] - zz_nn_ref, \
            label=r'$\chi=%d$' % (bonddim))


    for ax, ylabel_str in zip(axes.flatten(), axes_ylabels.flatten()):
        ax.grid()
        ax.legend()
        ax.set_xlabel(r'$Jt/(2\pi)$')
        ax.set_ylabel(ylabel_str)

    plt.tight_layout()

    prefix = os.path.join('../_data_/', time.strftime('%Y-%m-%d_%H-%M-%S'))
    plt.savefig(prefix + '_qutip_tfim_nn_corr_%d_spins.pdf' % (n_spins))

# Next to nearest neighbor plots
if FLAG_CALC_NNN:
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    ((ax_nnn_xx, ax_nnn_xy, ax_nnn_xz), \
        (ax_nnn_yx, ax_nnn_yy, ax_nnn_yz), \
        (ax_nnn_zx, ax_nnn_zy, ax_nnn_zz)) = axes


    for ix_bonddim, bonddim in enumerate(bonddim_vals):
        ax_nnn_xx.plot(tlist/2/pi, xx_nnn[ix_bonddim, :] - xx_nnn_ref, \
             label=r'$\chi=%d$' % (bonddim))
        ax_nnn_xy.plot(tlist/2/pi, xy_nnn[ix_bonddim, :] - xy_nnn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nnn_xz.plot(tlist/2/pi, xz_nnn[ix_bonddim, :] - xz_nnn_ref, \
            label=r'$\chi=%d$' % (bonddim))

        ax_nnn_yx.plot(tlist/2/pi, yx_nnn[ix_bonddim, :] - yx_nnn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nnn_yy.plot(tlist/2/pi, yy_nnn[ix_bonddim, :] - yy_nnn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nnn_yz.plot(tlist/2/pi, yz_nnn[ix_bonddim, :] - yz_nnn_ref, \
            label=r'$\chi=%d$' % (bonddim))

        ax_nnn_zx.plot(tlist/2/pi, zx_nnn[ix_bonddim, :] - zx_nnn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nnn_zy.plot(tlist/2/pi, zy_nnn[ix_bonddim, :] - zy_nnn_ref, \
            label=r'$\chi=%d$' % (bonddim))
        ax_nnn_zz.plot(tlist/2/pi, zz_nnn[ix_bonddim, :] - zz_nnn_ref, \
            label=r'$\chi=%d$' % (bonddim))


    for ax, ylabel_str in zip(axes.flatten(), axes_ylabels.flatten()):
        ax.grid()
        ax.legend()
        ax.set_xlabel(r'$Jt/(2\pi)$')
        ax.set_ylabel(ylabel_str)

    plt.tight_layout()

    prefix = os.path.join('../_data_/', time.strftime('%Y-%m-%d_%H-%M-%S'))
    plt.savefig(prefix + '_qutip_tfim_nnn_corr_%d_spins.pdf' % (n_spins))