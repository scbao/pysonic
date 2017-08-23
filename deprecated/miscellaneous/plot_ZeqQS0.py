#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 14:49:35
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-06-29 18:43:22

""" Plot the balanced quasi-static deflection of the system as a function
    of charge and gas content, in the absence of acoustic stimulus. """

import numpy as np
import matplotlib.pyplot as plt

import PyNICE
from PyNICE.utils import LoadParams


# Initialization: create a NBLS instance
params = LoadParams()
a = 32e-9  # in-plane radius (m)
d = 0.0e-6  # embedding tissue thickness (m)
geom = {"a": a, "d": d}
Fdrive = 0.0  # dummy stimulation frequency
Qm0 = -71.9e-5
bls = PyNICE.BilayerSonophore(geom, params, Fdrive, Qm0)


# Define charge and gas content vectors
nQ = 200
ngas = 10
charges = np.linspace(-0.8, 0.4, nQ) * 1e-5
gas = np.linspace(0.5 * bls.ng0, 2.0 * bls.ng0, ngas)


# Compute balance deflections vs charges and gas content
ZeqQS = np.empty((ngas, nQ))
for i in range(ngas):
    for j in range(nQ):
        ZeqQS[i, j] = bls.balancedefQS(gas[i], charges[j])


# Plotting
fig, ax = plt.subplots()
fig.canvas.set_window_title("balance deflection vs. charge")
ax.set_xlabel('$Q_m\ (nC/cm^2)$', fontsize=18)
ax.set_ylabel('$Z_{eq}\ (nm)$', fontsize=18)
for i in range(ngas):
    ax.plot(charges * 1e5, ZeqQS[i, :] * 1e9,
            label='ng = {:.2f}e-22 mole'.format(gas[i] * 1e22))
ax.legend(fontsize=18)
plt.show()
