#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 20:40:27 2019

@author: anupam
"""
import qutip
import itertools

import numpy as np

from numpy import sqrt, sin, cos, exp, pi

def calcLocalOp (n, op, l):
    """
    Calculates the operator `op` acting at location `l` in a system of
    `n` identical degrees of freedom

    Parameters
    ----------
    n: int
        Number of independent degrees of freedom

    op: qutip.Qobj
        Operator acting on

    l: int
        Location at which to compute the local operator
    """

    assert 0 <= l < n

    dimOp = op.shape[0]

    if l == 0:
        opMany = op
    else:
        opMany = qutip.tensor([qutip.qeye(dimOp)]*l)
        opMany = qutip.tensor(opMany, op)

    opMany = qutip.tensor([opMany] + [qutip.qeye(dimOp)]*(n-l-1))

    return opMany

def calc2LocalOp (n, opl, opm, l, m):
    """
    Calculates the operator `opl` acting at location `l` and operator `opm`
    acting at location `m` in a system, with `l != m` of `n` identical degrees
    of freedom

    Parameters
    ----------
    n: int
        Number of independent degrees of freedom

    opl: qutip.Qobj
        Operator acting at location l

    opm: qutip.Qobj
        Operator acting at location m

    l: int
        Location at which to compute  local operator opl

    m: int
        Location at which to compute  local operator opm
    """

    assert 0 <= l < n
    assert 0 <= m < n

    assert l != m

    dimOpl = opl.shape[0]
    dimOpm = opm.shape[0]

    assert dimOpl == dimOpm

    if l < m:
        ll = l
        mm = m
        opll = opl
        opmm = opm
    elif l > m:
        ll = m
        mm = l
        opll = opm
        opmm = opl

    opManyLeft = calcLocalOp(mm, opll, ll)

    if mm < n-1:
        opManyRight = calcLocalOp(n-mm-1, opll, 0)
        opMany = qutip.tensor([opManyLeft, opmm, opManyRight])

    else:
        opManyRight = 1

    opMany = qutip.tensor([opManyLeft, opmm, opManyRight])

    return opMany

def calcNearestNeighbor1D (n, opa, opb, pbc=False):
    """
    Computes nearest neighbor interactions in one dimension

    Parameters
    ----------
    n: int
        Number of independent degrees of freedom

    opa: qutip.Qobj
        Operator acting at location l

    opb: qutip.Qobj
        Operator acting at location m

    pbc: bool
        Flag to determine whether to use periodic boundary conditions

    """

    dimOpa = opa.shape[0]
    dimOpb = opb.shape[0]

    assert dimOpa == dimOpb

    opTotal = calc2LocalOp(n, opa, opb, 0, 1)

    for l in range(1, n-2):
        op = calc2LocalOp(n, opa, opb, l, l+1)
        opTotal = opTotal + op

    if pbc:
        op = calc2LocalOp(n, opa, opb, 0, n-1)
        opTotal = opTotal + op

    return opTotal

def calcCollectiveOp (n_bodies, op):
    """
    Calculates the collective operator

    Parameters
    ---------
    n_bodies: Number of bodies

    op: One body operator
    """

    n_onebody_dim = op.shape[0]

    op_manybody = None

    for n in range(n_bodies):

        # Note the python uses zero based indexing. If there was one based
        # indexing, the condition would be ``n > 1``
        if n == 0:
            op_term = op
        else:
            op_term_list = [qutip.identity(n_onebody_dim)] * (n)
            op_term = qutip.tensor(op_term_list)

            op_term = qutip.tensor(op_term, op)

        op_term_list = [qutip.identity(n_onebody_dim)] * (n_bodies - n - 1)
        op_term = qutip.tensor([op_term] + op_term_list)

        if op_manybody == None:
            op_manybody = op_term
        else:

            op_manybody += op_term

    return op_manybody



if __name__ == '__main__':
    hzz = calcNearestNeighbor1D(3, qutip.sigmaz(), qutip.sigmaz())
