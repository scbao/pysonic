#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-01-11 18:54:00
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-07-18 15:00:34

""" Plot the rate constants of the sodium and potassium channels
as a function of the membrane potential. """

import numpy as np
import matplotlib.pyplot as plt

from PointNICE.channels import CorticalRS

# Create channels mechanism
rs_mech = CorticalRS()

# Define potential and charge input vectors
Cm0 = 1e-2  # F/m2
Vm = np.linspace(-420, 420, 1000)
Qm = Vm * Cm0 * 1e-3  # C/m2

# Plot
fig, axes = plt.subplots(nrows=3, ncols=3)
fs = 16
st = fig.suptitle("Regular Spiking neuron", fontsize=fs)

# 1: Vm
ax = axes[0, 0]
ax.set_xlabel('$Q_m\ (nC/cm2)$', fontsize=fs)
ax.set_ylabel('$V_m\ (mV)$', fontsize=fs)
ax.plot(Qm * 1e5, Vm)
ax.set_xlim([-150, 150])

# 2: alpha_m
ax = axes[0, 1]
ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$\\alpha_m\ (ms^{-1})$', fontsize=fs)
ax.plot(Vm, rs_mech.alpham(Vm) * 1e-3)

# 3: beta_m
ax = axes[0, 2]
ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$\\beta_m\ (ms^{-1})$', fontsize=fs)
ax.plot(Vm, rs_mech.betam(Vm) * 1e-3)

# 4: alpha_h
ax = axes[1, 0]
ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$\\alpha_h\ (ms^{-1})$', fontsize=fs)
ax.plot(Vm, rs_mech.alphah(Vm) * 1e-3)

# 5: beta_h
ax = axes[1, 1]
ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$\\beta_h\ (ms^{-1})$', fontsize=fs)
ax.plot(Vm, rs_mech.betah(Vm) * 1e-3)

# 6: alpha_n
ax = axes[1, 2]
ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$\\alpha_n\ (ms^{-1})$', fontsize=fs)
ax.plot(Vm, rs_mech.alphan(Vm) * 1e-3)

# 7: beta_n
ax = axes[2, 0]
ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$\\beta_n\ (ms^{-1})$', fontsize=fs)
ax.plot(Vm, rs_mech.betan(Vm) * 1e-3)

# 8: pinf_over_taup
ax = axes[2, 1]
ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$p_{\\infty} / \\tau_p\ (ms^{-1})$', fontsize=fs)
ax.plot(Vm, rs_mech.pinf(Vm) / rs_mech.taup(Vm) * 1e-3)

# 9: inv_taup
ax = axes[2, 2]
ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
ax.set_ylabel('$1 / \\tau_p\ (ms^{-1})$', fontsize=fs)
ax.plot(Vm, 1 / rs_mech.taup(Vm) * 1e-3)

plt.show()
