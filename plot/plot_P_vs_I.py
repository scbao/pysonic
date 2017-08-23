#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-17 11:47:50
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-07-21 18:40:04

''' plot profile of acoustic Intensity (in W/cm^2) vs Pressure (in kPa) '''

import numpy as np
import matplotlib.pyplot as plt
from PointNICE.utils import Pressure2Intensity

rho = 1075  # kg/m3
c = 1515  # m/s

fig, ax = plt.subplots()
ax.set_xlabel('$Pressure\ (kPa)$')
ax.set_ylabel('$I_{SPPA}\ (W/cm^2)$')
ax.set_xscale('log')
ax.set_yscale('log')

P = np.logspace(np.log10(1e1), np.log10(1e7), num=500)  # Pa
Int = Pressure2Intensity(P, rho, c)  # W/m2
ax.plot(P * 1e-3, Int * 1e-4)


Psnaps = np.logspace(1, 7, 7)  # Pa
for Psnap in Psnaps:
    Isnap = Pressure2Intensity(Psnap, rho, c)  # W/m2
    ax.plot(np.array([Psnap, Psnap]) * 1e-3, np.array([0.0, Isnap]) * 1e-4, '--', color='black')
    ax.plot(np.array([0, Psnap]) * 1e-3, np.array([Isnap, Isnap]) * 1e-4, '--', color='black')

plt.show()
