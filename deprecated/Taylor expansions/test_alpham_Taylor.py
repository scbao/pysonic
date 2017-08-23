#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-03-21 11:38:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-03-29 19:12:27


""" Taylor expansions of the alpha_m function around different potential values.  """

import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt
from utils import bilinearExp

# Vm vector
nVm = 100
Vm = np.linspace(-80.0, 50.0, nVm)  # mV

# alpha_m vector
am_params = (-43.2, -0.32, 0.25)
alpham = bilinearExp(Vm, am_params, 0)

# alpha_m Taylor expansion
npoints = 10
norder = 4
Vm0 = np.linspace(-80.0, 50.0, npoints)  # mV
Vmdiff = Vm - np.tile(Vm0, (nVm, 1)).transpose()
Talpham = np.empty((npoints, nVm))
for i in range(npoints):
    T = np.zeros(nVm)
    for j in range(norder + 1):
        T[:] += bilinearExp(Vm0[i], am_params, j) * Vmdiff[i, :]**j / factorial(j)
    Talpham[i, :] = T

# Plot standard alpha_m vs. Taylor reconstruction around Vm0
_, ax = plt.subplots(figsize=(22, 10))
ax.set_xlabel('$V_m\ [mV]$', fontsize=20)
ax.set_ylabel('$[ms^{-1}]$', fontsize=20)
ax.plot(Vm, alpham, linewidth=2, label='$\\alpha_m$')
for i in range(npoints):
    ax.plot(Vm, Talpham[i, :], linewidth=2, label='$T_{}\\alpha_m({:.1f})$'.format(norder, Vm0[i]))
ax.legend(fontsize=20)

plt.show()
