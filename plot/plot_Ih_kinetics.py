#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-11 20:35:38
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-12 19:45:05

""" Plot the voltage-dependent kinetics of the hyperpolarization-activated
    cationic current found in thalamo-cortical neurons. """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

# from PointNICE.solvers import SolverElec
from PointNICE.neurons import ThalamoCortical
from PointNICE.utils import rescale

# -------------- SIMULATION -----------------


# Create channels mechanism
neuron = ThalamoCortical()


# Input vectors
nV = 100
nC = 10
CCa_min = 0.01  # uM
CCa_max = 10  # um
Vm = np.linspace(-100, 50, nV)  # mV
CCa = np.logspace(np.log10(CCa_min), np.log10(CCa_max), nC)  # uM


# Output matrix: relative activation (0-2)
BA = neuron.betao(Vm) / neuron.alphao(Vm)
P0 = neuron.k2 / (neuron.k2 + neuron.k1 * (CCa * 1e-6)**4)
gH_rel = np.empty((nV, nC))
for i in range(nC):
    O_form = neuron.k4 / (neuron.k3 * (1 - P0[i]) + neuron.k4 * (1 + BA))
    OL_form = (1 - O_form * (1 + BA))
    gH_rel[:, i] = O_form + 2 * OL_form


mymap = cm.get_cmap('viridis')
sm = plt.cm.ScalarMappable(cmap=mymap, norm=LogNorm(CCa_min, CCa_max))
sm._A = []

fs = 18

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title('global activation', fontsize=fs)
ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$(O + 2O_L)_{\infty}$', fontsize=fs)
ax.set_yticks([0, 1, 2])
for i in range(nC):
    ax.plot(Vm, gH_rel[:, i], linewidth=2,
            c=mymap(rescale(np.log10(CCa[i]), np.log10(CCa_min), np.log10(CCa_max))))
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
fig.add_axes()
fig.colorbar(sm, cax=cbar_ax)
cbar_ax.set_ylabel('$[Ca^{2+}_i]\ (uM)$', fontsize=fs)


plt.show()
