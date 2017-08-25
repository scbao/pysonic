#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-11 20:35:38
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-25 10:42:24

""" Run simulations of the HH system with injected electric current,
and plot resulting dynamics. """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as patches

from PointNICE.solvers import SolverElec
from PointNICE.channels import *


# -------------- SIMULATION -----------------


# Create channels mechanism
neuron = ThalamoCortical()
for i in range(len(neuron.states_names)):
    print('{}0 = {:.2f}'.format(neuron.states_names[i], neuron.states0[i]))


# Set pulse parameters
tstim = 500e-3  # s
toffset = 300e-3  # s
Amin = -20.0
Amax = 20.0
amps = np.arange(Amin, Amax + 0.5, 1.0)
nA = len(amps)


root = 'C:/Users/admin/Desktop/test anim'
mymap = cm.get_cmap('coolwarm')
sm = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Amin, Amax))
sm._A = []


i = 0
for Astim in amps:

    i += 1

    # Run simulation
    print('sim {}/{} ({:.2f} mA/m2, {:.0f} ms)'.format(i, nA, Astim, tstim * 1e3))
    solver = SolverElec()
    (t, y) = solver.run(neuron, Astim, tstim, toffset)
    Vm = y[:, 0]

    # Plot membrane potential profile
    fs = 12
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlabel('$time\ (ms)$', fontsize=fs)
    ax.set_ylabel('$V_m\ (mV)$', fontsize=fs)
    ax.set_ylim(-150.0, 60.0)
    ax.set_xticks([0.0, 500.0])
    ax.set_yticks([-100, 50.0])
    ax.locator_params(axis='y', nbins=2)
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)
    for item in ax.get_xticklabels():
        item.set_fontsize(fs)
    (ybottom, ytop) = ax.get_ylim()
    ax.add_patch(patches.Rectangle((0.0, ybottom), tstim * 1e3, ytop - ybottom,
                                   facecolor='gold', alpha=0.2))

    ax.plot(t * 1e3, Vm, linewidth=2)
    plt.tight_layout()

    fig.subplots_adjust(right=0.80)
    cbar_ax = fig.add_axes([0.82, 0.2, 0.02, 0.75])
    fig.add_axes()
    fig.colorbar(sm, cax=cbar_ax, ticks=[Astim])
    cbar_ax.set_yticklabels(['{:.2f} mA/m2'.format(Astim)], fontsize=fs)

    fig.savefig('{}/fig{:03d}.png'.format(root, i))
    plt.close(fig)
