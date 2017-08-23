#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-11 20:35:38
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-17 13:19:26

""" Plot the voltage-dependent steady-states and time constants of activation and inactivation
    gates of the different ionic currents involved in the neuron's membrane. """

import numpy as np
import matplotlib.pyplot as plt

from PointNICE.channels import *

# Create channels mechanism
neuron = LeechTouch()

# Input membrane potential vector
Vm = np.linspace(-100, 50, 300)

# iNa gating dynamics
if neuron.name in ['RS', 'FS', 'LTS', 'RE', 'TC']:
    am = neuron.alpham(Vm)
    bm = neuron.betam(Vm)
    tm = 1 / (am + bm)
    minf = am * tm
    ah = neuron.alphah(Vm)
    bh = neuron.betah(Vm)
    th = 1 / (ah + bh)
    hinf = ah * th
elif neuron.name == 'LeechT':
    minf = neuron.minf(Vm)
    tm = np.ones(len(Vm)) * neuron.taum
    hinf = neuron.hinf(Vm)
    th = neuron.tauh(Vm)

# iK gating dynamics
if neuron.name in ['RS', 'FS', 'LTS', 'RE', 'TC']:
    an = neuron.alphan(Vm)
    bn = neuron.betan(Vm)
    tn = 1 / (an + bn)
    ninf = an * tn
elif neuron.name == 'LeechT':
    ninf = neuron.ninf(Vm)
    tn = neuron.taun(Vm)

# iM gating dynamics
if neuron.name in ['RS', 'FS', 'LTS']:
    tp = neuron.taup(Vm)
    pinf = neuron.pinf(Vm)

# iT gating dynamics
if neuron.name in ['LTS', 'RE', 'TC']:
    ts = neuron.taus(Vm)
    sinf = neuron.sinf(Vm)
    tu = np.array([neuron.tauu(v) for v in Vm])
    uinf = neuron.uinf(Vm)
elif neuron.name == 'LeechT':
    sinf = neuron.sinf(Vm)
    ts = np.ones(len(Vm)) * neuron.taus

# iH gating dynamics
if neuron.name in ['TC']:
    to = neuron.tauo(Vm)
    oinf = neuron.oinf(Vm)


# -------------- PLOTTING -----------------

fs = 12
fig, axes = plt.subplots(2)

fig.suptitle('Gating dynamics')

ax = axes[0]
ax.get_xaxis().set_ticklabels([])
# ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$X_{\infty}\ (mV)$', fontsize=fs)
ax.plot(Vm, minf, lw=2, c='C1', label='$m_{\infty}$')
ax.plot(Vm, hinf, '--', lw=2, c='C1', label='$h_{\infty}$')
ax.plot(Vm, ninf, lw=2, c='C0', label='$n_{\infty}$')
if neuron.name in ['RS', 'FS', 'LTS']:
    ax.plot(Vm, pinf, lw=2, color='C2', label='$p_{\infty}$')
if neuron.name in ['LTS', 'TC']:
    ax.plot(Vm, sinf, lw=2, color='r', label='$s_{\infty}$')
    ax.plot(Vm, uinf, '--', lw=2, color='r', label='$u_{\infty}$')
if neuron.name in ['RE']:
    ax.plot(Vm, sinf, lw=2, color='C5', label='$s_{\infty}$')
    ax.plot(Vm, uinf, '--', lw=2, color='C5', label='$u_{\infty}$')
if neuron.name in ['TC']:
    ax.plot(Vm, oinf, lw=2, color='#08457E', label='$o_{\infty}$')
if neuron.name in ['LeechT']:
    ax.plot(Vm, sinf, lw=2, color='r', label='$s_{\infty}$')
ax.legend(fontsize=fs, loc=7)

ax = axes[1]
ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$\\tau_X\ (ms)$', fontsize=fs)
ax.plot(Vm, tm * 1e3, lw=2, c='C1', label='$\\tau_m$')
ax.plot(Vm, th * 1e3, '--', lw=2, c='C1', label='$\\tau_h$')
ax.plot(Vm, tn * 1e3, lw=2, c='C0', label='$\\tau_n$')
if neuron.name in ['RS', 'FS', 'LTS']:
    ax.plot(Vm, tp * 1e3, lw=2, color='C2', label='$\\tau_p$')
if neuron.name in ['LTS', 'TC']:
    ax.plot(Vm, ts * 1e3, lw=2, color='r', label='$\\tau_s$')
    ax.plot(Vm, tu * 1e3, '--', lw=2, color='r', label='$\\tau_u$')
if neuron.name in ['RE']:
    ax.plot(Vm, ts * 1e3, lw=2, color='C5', label='$\\tau_s$')
    ax.plot(Vm, tu * 1e3, '--', lw=2, color='C5', label='$\\tau_u$')
if neuron.name in ['TC']:
    ax.plot(Vm, to * 1e3, lw=2, color='#08457E', label='$\\tau_o$')
if neuron.name in ['LeechT']:
    ax.plot(Vm, ts * 1e3, lw=2, color='r', label='$\\tau_s$')
ax.legend(fontsize=fs, loc=7)

plt.show()
