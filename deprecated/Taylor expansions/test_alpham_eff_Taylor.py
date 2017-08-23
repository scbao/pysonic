#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-03-21 11:38:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-03-29 19:40:44


""" Perform Taylor expansions (up to 4th order) of the alpha_m function
    along one acoustic cycle. """

import importlib
import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nblscore
from utils import LoadParams, rescale, bilinearExp
from constants import *
importlib.reload(nblscore)  # reloading nblscore module


# Load NBLS parameters
params = LoadParams("params.yaml")
biomech = params['biomech']
ac_imp = biomech['rhoL'] * biomech['c']  # Rayl

# Set geometry of NBLS structure
a = 32e-9  # in-plane radius (m)
d = 0.0e-6  # embedding tissue thickness (m)
geom = {"a": a, "d": d}

# Create a NBLS instance here (with dummy frequency parameter)
nbls = nblscore.NeuronalBilayerSonophore(geom, params, 0.0, True)

# Set stimulation parameters
Fdrive = 3.5e5  # Hz
Adrive = 1e5  # Pa
phi = np.pi  # acoustic wave phase

# Set charge linear space
nQ = 100
charges = np.linspace(-80.0, 50.0, nQ) * 1e-5  # C/m2
Qmin = np.amin(charges)
Qmax = np.amax(charges)

# Set alpha_m parameters
am_params = (-43.2, -0.32, 0.25)

# Set highest Taylor expansion order
norder = 4

# Set time vector
T = 1 / Fdrive
t = np.linspace(0, T, NPC_FULL)
dt = t[1] - t[0]

# Initialize coefficients vectors
deflections = np.empty((nQ, NPC_FULL))
Vm = np.empty((nQ, NPC_FULL))
alpham = np.empty((nQ, NPC_FULL))


# Run mechanical simulations for each imposed charge density
print('Running {} mechanical simulations with imposed charge densities'.format(nQ))
simcount = 0
for i in range(nQ):
    simcount += 1

    # Log to console
    print('--- sim {}/{}: Q = {:.1f} nC/cm2'.format(simcount, nQ, charges[i] * 1e5))

    # Run simulation and retrieve deflection vector
    (_, y, _) = nbls.runMech(Adrive, Fdrive, phi, charges[i])
    (_, Z, _) = y
    deflections[i, :] = Z[-NPC_FULL:]

    # Compute Vm and alpham vectors
    Vm[i, :] = [charges[i] / nbls.Capct(ZZ) for ZZ in deflections[i, :]]
    alpham[i, :] = bilinearExp(Vm[i, :] * 1e3, am_params, 0)


# time-average Vm and alpham
Vmavg = np.mean(Vm, axis=1)
alphamavg = np.mean(alpham, axis=1)

# (Vm - Vmavg) differences along cycle
Vmavgext = np.tile(Vmavg, (NPC_FULL, 1)).transpose()
Vmdiff = (Vm - Vmavgext) * 1e3

# alpham derivatives
dalpham = np.empty((norder + 1, nQ))
for j in range(norder + 1):
    dalpham[j, :] = bilinearExp(Vmavg * 1e3, am_params, j)

# Taylor expansions along cycle
Talpham = np.empty((norder + 1, nQ, NPC_FULL))
dalphamext = np.tile(dalpham.transpose(), (NPC_FULL, 1, 1)).transpose()
Talpham[0, :, :] = dalphamext[0, :, :]
for j in range(1, norder + 1):
    jterm = dalphamext[j, :, :] * Vmdiff[:, :]**j / factorial(j)
    Talpham[j, :, :] = Talpham[j - 1, :, :] + jterm

# time-averaging of Taylor expansions
Talphamavg = np.mean(Talpham, axis=2)


# ------------------ PLOTS -------------------

mymap = cm.get_cmap('jet')
sm_Q = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Qmin * 1e5, Qmax * 1e5))
sm_Q._A = []
t_factor = 1e6

# 1: time average Vm
_, ax = plt.subplots(figsize=(22, 10))
ax.set_xlabel('$Qm\ [uF/cm^2]$', fontsize=20)
ax.set_ylabel('$\\overline{V_m}\ [mV]$', fontsize=20)
ax.plot(charges * 1e5, Vmavg * 1e3, linewidth=2)

# 2: alpham: standard time-averaged vs.evaluated at time-average Vm
#            vs. Taylor reconstructions around Vm_avg
_, ax = plt.subplots(figsize=(22, 10))
ax.set_xlabel('$Qm\ [uF/cm^2]$', fontsize=20)
ax.set_ylabel('$[ms^{-1}]$', fontsize=20)
ax.plot(charges * 1e5, alphamavg, linewidth=2, label='$\\overline{\\alpha_m(V_m)}$')
for j in range(norder + 1):
    ax.plot(charges * 1e5, Talphamavg[j, :], linewidth=2,
            label='$\\overline{T_' + str(j) + '[\\alpha_m(\\overline{V_m})]}$')
    ax.legend(fontsize=20)

# 3: original alpham vs. highest order Taylor alpham reconstruction
_, ax = plt.subplots(figsize=(22, 10))
ax.set_xlabel('$t \ (us)$', fontsize=20)
ax.set_ylabel('$[ms^{-1}]$', fontsize=20)
ax.plot(t * t_factor, alpham[0, :], linewidth=2,
        c=mymap(rescale(charges[0], Qmin, Qmax)), label='$\\overline{\\alpha_m(V_m)}$')
ax.plot(t * t_factor, Talpham[-1, 0, :], '--', linewidth=2,
        c=mymap(rescale(charges[0], Qmin, Qmax)),
        label='$T_' + str(norder) + '[\\alpha_m(\\overline{V_m})]$')
for i in range(1, nQ):
    ax.plot(t * t_factor, alpham[i, :], linewidth=2,
            c=mymap(rescale(charges[i], Qmin, Qmax)))
    ax.plot(t * t_factor, Talpham[-1, i, :], '--', linewidth=2,
            c=mymap(rescale(charges[i], Qmin, Qmax)))
cbar = plt.colorbar(sm_Q)
cbar.ax.set_ylabel('$Q \ (nC/cm^2)$', fontsize=28)
ax.legend(fontsize=20)
plt.tight_layout()

plt.show()
