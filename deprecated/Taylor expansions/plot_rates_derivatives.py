#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-03-22 16:04:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-03-29 18:17:52

''' Plot profiles of rate constants functions and derivatives '''

import numpy as np
import matplotlib.pyplot as plt
from utils import bilinearExp, stdExp, dualExp, symExp, sigmoid


# Define function parameters
am_params = (-43.2, -0.32, 0.25)
Bm_params = (-16.2, 0.28, -0.20)
ah_params = (-39.2, 0.128, 1 / 18)
Bh_params = (-16.2, 4, 0.20)
an_params = (-41.2, -0.032, 0.20)
Bn_params = (-46.2, 0.5, 0.025)
pinf_params = (-35.0, 1, 0.1)
Tp_params = (-35.0, 0.608, 3.3, 0.05)
invTp_params = (-35.0, 1 / 0.608, 3.3, 0.05)


# Define potential range and maximal derivation order
nVm = 100
Vm = np.linspace(-80.0, 50.0, nVm)  # mV
norder = 3


# Define vectors
dalpham = np.empty((norder + 1, nVm))
dbetam = np.empty((norder + 1, nVm))
dalphah = np.empty((norder + 1, nVm))
dbetah = np.empty((norder + 1, nVm))
dalphan = np.empty((norder + 1, nVm))
dbetan = np.empty((norder + 1, nVm))
dpinf = np.empty((norder + 1, nVm))
dtaup = np.empty((norder + 1, nVm))
dinvTp = np.empty((norder + 1, nVm))
dpinfoverTp = np.empty((norder + 1, nVm))


# Compute derivatives
for i in range(norder + 1):
    dalpham[i, :] = bilinearExp(Vm, am_params, i)
    dbetam[i, :] = bilinearExp(Vm, Bm_params, i)
    dalphah[i, :] = stdExp(Vm, ah_params, i)
    dbetah[i, :] = sigmoid(Vm, Bh_params, i)
    dalphan[i, :] = bilinearExp(Vm, an_params, i)
    dbetan[i, :] = stdExp(Vm, Bn_params, i)
    dpinf[i, :] = sigmoid(Vm, pinf_params, i)
    dtaup[i, :] = symExp(Vm, Tp_params, i) * 1e3
    dinvTp[i, :] = dualExp(Vm, invTp_params, i) * 1e-3


# Compute pinf/taup derivatives
dpinfoverTp[0, :] = dpinf[0, :] * dinvTp[0, :]
dpinfoverTp[1, :] = dpinf[1, :] * dinvTp[0, :] + dpinf[0, :] * dinvTp[1, :]
dpinfoverTp[2, :] = dpinf[2, :] * dinvTp[0, :] + dpinf[1, :] * dinvTp[1, :]\
    + dpinf[0, :] * dinvTp[2, :]
dpinfoverTp[3, :] = dpinf[3, :] * dinvTp[0, :] + 3 * dpinf[2, :] * dinvTp[1, :]\
    + 3 * dpinf[1, :] * dinvTp[2, :] + dpinf[0, :] * dinvTp[3, :]


# Define plot parameters
seqx = (0, 0, 1, 1)
seqy = (0, 1, 0, 1)
f_str1 = ('$[ms^{-1}]$', '$d\ [ms^{-1}.mV^{-1}]$', '$d^2\ [ms^{-1}.mV^{-2}]$',
          '$d^3\ [ms^{-1}.mV^{-3}]$')
f_str2 = ('$[-]$', '$d\ [mV^{-1}]$', '$d^2\ [mV^{-2}]$', '$d^3\ [mV^{-3}]$')
f_str3 = ('$[ms]$', '$d\ [ms.mV^{-1}]$', '$d^2\ [ms.mV^{-2}]$', '$d^3\ [ms.mV^{-3}]$')
titles1 = ('$\\alpha_m$', '$\\beta_m$', '$\\alpha_h$', '$\\beta_h$', '$\\alpha_n$', '$\\beta_n$')
titles2 = ('$\\frac{1}{\\tau_p}$', '$\\frac{p_{\\infty}}{\\tau_p}$')
vectors1 = (dalpham, dbetam, dalphah, dbetah, dalphan, dbetan)
vectors2 = (dinvTp, dpinfoverTp)

# Plot alpha and beta functions
for j in range(len(vectors1)):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 10))
    for i in range(4):
        ax = axes[seqx[i], seqy[i]]
        ax.set_xlabel('$V_m \ [mV]$', fontsize=24)
        ax.set_ylabel(f_str1[i], fontsize=24)
        ax.plot(Vm, vectors1[j][i, :])
    fig.suptitle(titles1[j], fontsize=30)


# Plot p_inf functions
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 10))
for i in range(4):
    ax = axes[seqx[i], seqy[i]]
    ax.set_xlabel('$V_m \ [mV]$', fontsize=24)
    ax.set_ylabel(f_str2[i], fontsize=24)
    ax.plot(Vm, dpinf[i, :])
fig.suptitle('$p_{\\infty}$', fontsize=30)


# Plot tau_p functions
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 10))
for i in range(4):
    ax = axes[seqx[i], seqy[i]]
    ax.set_xlabel('$V_m \ [mV]$', fontsize=24)
    ax.set_ylabel(f_str3[i], fontsize=24)
    ax.plot(Vm, dtaup[i, :])
fig.suptitle('$\\tau_p$', fontsize=30)


# Plot invTaup and pinf/Taup functions
for j in range(len(vectors2)):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 10))
    for i in range(4):
        ax = axes[seqx[i], seqy[i]]
        ax.set_xlabel('$V_m \ [mV]$', fontsize=24)
        ax.set_ylabel(f_str1[i], fontsize=24)
        ax.plot(Vm, vectors2[j][i, :])
    fig.suptitle(titles2[j], fontsize=30)


plt.show()
