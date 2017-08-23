#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-11 20:35:38
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-22 18:08:12

""" Run simulations of the HH system with injected electric current,
and plot resulting dynamics. """

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PointNICE.solvers import SolverElec
from PointNICE.channels import *


# -------------- SIMULATION -----------------


# Create channels mechanism
neuron = LeechTouch()

print('initial states:')
print('Vm0 = {:.2f}'.format(neuron.Vm0))
for i in range(len(neuron.states_names)):
    if neuron.states_names[i] == 'C_Ca':
        print('{}0 = {:.3f} uM'.format(neuron.states_names[i], neuron.states0[i] * 1e6))
    else:
        print('{}0 = {:.2f}'.format(neuron.states_names[i], neuron.states0[i]))


# Set pulse parameters
tstim = 1.5  # s
toffset = 0.5  # s
Astim = 4.1  # mA/m2

show_currents = False

# Run simulation
print('stimulating {} neuron ({:.2f} mA/m2, {:.0f} ms)'.format(neuron.name, Astim, tstim * 1e3))
solver = SolverElec()
(t, y) = solver.runSim(neuron, Astim, tstim, toffset)


# -------------- VARIABLES SEPARATION -----------------


# Membrane potential and states
Vm = y[:, 0]
states = y[:, 1:].T

# Final states
statesf = y[-1, 1:]
print('final states:')
print('Vmf = {:.2f}'.format(Vm[-1]))
for i in range(len(neuron.states_names)):
    if len(neuron.states_names[i]) == 1:  # channel state
        print('{}f = {:.2f}'.format(neuron.states_names[i], statesf[i]))
    else:  # other state
        print('{}f = {:f}'.format(neuron.states_names[i], statesf[i]))


# Leakage current and net current
iL = neuron.currL(Vm)
iNet = neuron.currNet(Vm, states)

# Sodium and Potassium gating dynamics and currents
m = y[:, 1]
h = y[:, 2]
n = y[:, 3]
iNa = neuron.currNa(m, h, Vm)
iK = neuron.currK(n, Vm)


corticals = ['RS', 'FS', 'LTS']
thalamics = ['RE', 'TC']
leeches = ['LeechT']

# Cortical neurons
if neuron.name in corticals:
    p = y[:, 4]
    iM = neuron.currM(p, Vm)

    # Special case: LTS neuron
    if neuron.name == 'LTS':
        s = y[:, 5]
        u = y[:, 6]
        iCa = neuron.currCa(s, u, Vm)


# Thalamic neurons
if neuron.name in thalamics:
    s = y[:, 4]
    u = y[:, 5]
    iCa = neuron.currCa(s, u, Vm)

    # Special case: TC neuron
    if neuron.name == 'TC':
        O = y[:, 6]
        C = y[:, 7]
        P0 = y[:, 8]
        C_Ca = y[:, 9]
        OL = 1 - O - C
        P1 = 1 - P0
        Ih = neuron.currH(O, C, Vm)
        IKL = neuron.currKL(Vm)

# Leech neurons
if neuron.name in leeches:
    s = y[:, 4]
    C_Na = y[:, 5]
    A_Na = y[:, 6]
    C_Ca = y[:, 7]
    A_Ca = y[:, 8]
    iCa = neuron.currCa(s, Vm)
    iPumpNa = neuron.currPumpNa(C_Na, Vm)
    iKCa = neuron.currKCa(C_Ca, Vm)


# -------------- PLOTTING -----------------

fs = 12
if neuron.name == 'TC':
    naxes = 7
if neuron.name in ['LTS', 'RE']:
    naxes = 5
if neuron.name in ['RS', 'FS']:
    naxes = 4
if neuron.name in leeches:
    naxes = 7

if not show_currents:
    naxes -= 1

height = 5.5
if neuron.name == 'TC':
    height = 7
fig, axes = plt.subplots(naxes, 1, figsize=(10, height))

# Membrane potential
i = 0
ax = axes[i]
ax.plot(t * 1e3, Vm, linewidth=2)
ax.set_ylabel('$V_m\ (mV)$', fontsize=fs)
if i < naxes - 1:
    ax.get_xaxis().set_ticklabels([])
ax.locator_params(axis='y', nbins=2)
for item in ax.get_yticklabels():
    item.set_fontsize(fs)
(ybottom, ytop) = ax.get_ylim()
ax.add_patch(patches.Rectangle((0.0, ybottom), tstim * 1e3, ytop - ybottom,
                               color='#8A8A8A', alpha=0.1))

# iNa dynamics
i += 1
ax = axes[i]
ax.set_ylim([-0.1, 1.1])
ax.set_ylabel('$Na^+ \ kin.$', fontsize=fs)
ax.plot(t * 1e3, m, color='blue', linewidth=2, label='$m$')
ax.plot(t * 1e3, h, color='red', linewidth=2, label='$h$')
ax.plot(t * 1e3, m**2 * h, '--', color='black', linewidth=2, label='$m^2h$')
(ybottom, ytop) = ax.get_ylim()
ax.add_patch(patches.Rectangle((0.0, ybottom), tstim * 1e3, ytop - ybottom,
                               color='#8A8A8A', alpha=0.1))
ax.legend(fontsize=fs, loc=7)
if i < naxes - 1:
    ax.get_xaxis().set_ticklabels([])
ax.locator_params(axis='y', nbins=2)
for item in ax.get_yticklabels():
    item.set_fontsize(fs)


# iK & iM dynamics
i += 1
ax = axes[i]
ax.set_ylim([-0.1, 1.1])
ax.set_ylabel('$K^+ \ kin.$', fontsize=fs)
ax.plot(t * 1e3, n, color='#734d26', linewidth=2, label='$n$')
if neuron.name in ['RS', 'FS', 'LTS']:
    ax.plot(t * 1e3, p, color='#660099', linewidth=2, label='$p$')
(ybottom, ytop) = ax.get_ylim()
ax.add_patch(patches.Rectangle((0.0, ybottom), tstim * 1e3, ytop - ybottom,
                               color='#8A8A8A', alpha=0.1))
ax.legend(fontsize=fs, loc=7)
if i < naxes - 1:
    ax.get_xaxis().set_ticklabels([])
ax.locator_params(axis='y', nbins=2)
for item in ax.get_yticklabels():
    item.set_fontsize(fs)

# iCa dynamics
if neuron.name in ['LTS', 'RE', 'TC', 'LeechT']:
    i += 1
    ax = axes[i]
    ax.set_ylim([-0.1, 1.1])
    ax.set_ylabel('$Ca^{2+} \ kin.$', fontsize=fs)
    ax.plot(t * 1e3, s, color='#2d862d', linewidth=2, label='$s$')
    if neuron.name in ['LTS', 'RE', 'TC']:
        ax.plot(t * 1e3, u, color='#e68a00', linewidth=2, label='$u$')
        ax.plot(t * 1e3, s**2 * u, '--', color='black', linewidth=2, label='$s^2u$')
    (ybottom, ytop) = ax.get_ylim()
    ax.add_patch(patches.Rectangle((0.0, ybottom), tstim * 1e3, ytop - ybottom,
                                   color='#8A8A8A', alpha=0.1))
    ax.legend(fontsize=fs, loc=7)
    if i < naxes - 1:
        ax.get_xaxis().set_ticklabels([])
    ax.locator_params(axis='y', nbins=2)
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)


# iH dynamics
if neuron.name == 'TC':
    i += 1
    ax = axes[i]
    ax.set_ylim([-0.1, 2.1])
    ax.set_ylabel('$i_H\ kin.$', fontsize=fs)
    # ax.plot(t * 1e3, C, linewidth=2, label='$C$')
    ax.plot(t * 1e3, O, linewidth=2, label='$O$')
    ax.plot(t * 1e3, OL, linewidth=2, label='$O_L$')
    ax.plot(t * 1e3, O + 2 * OL, '--', color='black', linewidth=2, label='$O + 2O_L$')
    (ybottom, ytop) = ax.get_ylim()
    ax.add_patch(patches.Rectangle((0.0, ybottom), tstim * 1e3, ytop - ybottom,
                                   color='#8A8A8A', alpha=0.1))
    ax.legend(fontsize=fs, ncol=2, loc=7)
    if i < naxes - 1:
        ax.get_xaxis().set_ticklabels([])
    ax.locator_params(axis='y', nbins=2)
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)

# submembrane [Ca2+] dynamics
if neuron.name in ['TC', 'LeechT']:
    i += 1
    ax = axes[i]
    if neuron.name == 'TC':
        ax.set_ylabel('$[Ca^{2+}_i]\ (uM)$', fontsize=fs)
        ax.plot(t * 1e3, C_Ca * 1e6, linewidth=2, label='$[Ca^{2+}_i]$')
    if neuron.name == 'LeechT':
        ax.set_ylabel('$[Ca^{2+}_i]\ (arb.)$', fontsize=fs)
        ax.plot(t * 1e3, C_Ca, linewidth=2, label='$[Ca^{2+}_i]$')
        ax.plot(t * 1e3, A_Ca, linewidth=2, label='$A_{Ca}$')
        ax.legend(fontsize=fs, loc=7)
    (ybottom, ytop) = ax.get_ylim()
    ax.add_patch(patches.Rectangle((0.0, ybottom), tstim * 1e3, ytop - ybottom,
                                   color='#8A8A8A', alpha=0.1))
    if i < naxes - 1:
        ax.get_xaxis().set_ticklabels([])
    ax.locator_params(axis='y', nbins=2)
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)

# submembrane [Na+] dynamics
if neuron.name == 'LeechT':
    i += 1
    ax = axes[i]
    ax.set_ylabel('$[Na^{+}_i]\ (arb.)$', fontsize=fs)
    ax.plot(t * 1e3, C_Na, linewidth=2, label='$[Na^{+}_i]$')
    ax.plot(t * 1e3, A_Na, linewidth=2, label='$A_{Na}$')
    (ybottom, ytop) = ax.get_ylim()
    ax.add_patch(patches.Rectangle((0.0, ybottom), tstim * 1e3, ytop - ybottom,
                                   color='#8A8A8A', alpha=0.1))
    ax.legend(fontsize=fs, loc=7)
    if i < naxes - 1:
        ax.get_xaxis().set_ticklabels([])
    ax.locator_params(axis='y', nbins=2)
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)

# currents
if show_currents:
    i += 1
    ax = axes[i]
    ax.set_ylabel('$I\ (A/m^2)$', fontsize=fs)
    ax.set_xlabel('$time\ (ms)$', fontsize=fs)
    ax.plot(t * 1e3, iNa * 1e-3, linewidth=2, label='$i_{Na}$')
    ax.plot(t * 1e3, iK * 1e-3, linewidth=2, label='$i_K$')
    if neuron.name in ['RS', 'FS', 'LTS']:
        ax.plot(t * 1e3, iM * 1e-3, linewidth=2, label='$i_M$')
    if neuron.name in ['LTS', 'TC', 'LeechT']:
        ax.plot(t * 1e3, iCa * 1e-3, linewidth=2, label='$i_{T}$')
    if neuron.name == 'RE':
        ax.plot(t * 1e3, iCa * 1e-3, linewidth=2, label='$i_{TS}$')
    if neuron.name == 'TC':
        ax.plot(t * 1e3, Ih * 1e-3, linewidth=2, label='$i_{H}$')
        ax.plot(t * 1e3, IKL * 1e-3, linewidth=2, label='$i_{KL}$')
    if neuron.name == 'LeechT':
        ax.plot(t * 1e3, iKCa * 1e-3, linewidth=2, label='$i_{K,Ca}$')
        ax.plot(t * 1e3, iPumpNa * 1e-3, linewidth=2, label='$i_{Na\ pump}$')
    ax.plot(t * 1e3, iL * 1e-3, linewidth=2, label='$i_L$')
    ax.plot(t * 1e3, iNet * 1e-3, '--', linewidth=2, color='black', label='$i_{Net}$')
    ax.legend(fontsize=fs, ncol=2, loc=7)
    ax.locator_params(axis='y', nbins=2)
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)
    if i < naxes - 1:
        ax.get_xaxis().set_ticklabels([])
    (ybottom, ytop) = ax.get_ylim()
    ax.add_patch(patches.Rectangle((0.0, ybottom), tstim * 1e3, ytop - ybottom,
                                   color='#8A8A8A', alpha=0.1))


axes[-1].set_xlabel('$time\ (ms)$', fontsize=fs)
for item in axes[-1].get_xticklabels():
    item.set_fontsize(fs)

if tstim > 0.0:
    title = '{} neuron ({:.2f} mA/m2, {:.0f} ms)'.format(neuron.name, Astim, tstim * 1e3)
else:
    title = '{} neuron (free, {:.0f} ms)'.format(neuron.name, toffset * 1e3)
fig.suptitle(title, fontsize=fs)

plt.show()
