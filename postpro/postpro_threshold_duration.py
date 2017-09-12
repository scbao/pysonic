#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-30 21:48:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-28 14:18:27

""" Test relationship between stimulus duration and minimum acoustic
amplitude / intensity / energy for AP generation. """

import numpy as np
import matplotlib.pyplot as plt

from PointNICE.utils import ImportExcelCol, Pressure2Intensity

# Import data
xls_file = "C:/Users/admin/Desktop/Model output/NBLS titration duration 0.35MHz/nbls_log_titration_duration_0.35MHz.xlsx"
sheet = 'Data'
f = ImportExcelCol(xls_file, sheet, 'E', 2) * 1e3  # Hz
A = ImportExcelCol(xls_file, sheet, 'F', 2) * 1e3  # Pa
T = ImportExcelCol(xls_file, sheet, 'G', 2) * 1e-3  # s
N = ImportExcelCol(xls_file, sheet, 'Q', 2)

# Convert to appropriate units
durations = T * 1e3  # ms
Trange = np.amax(durations) - np.amin(durations)
amplitudes = A * 1e-3  # kPa
intensities = Pressure2Intensity(A) * 1e-4  # W/cm2
energies = intensities * durations  # mJ/cm2

# Plot threshold amplitude vs. duration
fig1, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$duration \ (ms)$", fontsize=28)
ax.set_ylabel("$Amplitude \ (kPa)$", fontsize=28)
ax.scatter(durations, amplitudes, color='black', s=100)
ax.set_xlim(np.amin(durations) - 0.1 * Trange, np.amax(durations) + 0.1 * Trange)
for item in ax.get_yticklabels():
    item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)

# Plot threshold intensity vs. duration
fig2, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$duration \ (ms)$", fontsize=28)
ax.set_ylabel("$Intensity \ (W/cm^2)$", fontsize=28)
ax.scatter(durations, intensities, color='black', s=100)
ax.set_xlim(np.amin(durations) - 0.1 * Trange, np.amax(durations) + 0.1 * Trange)
ax.set_yticks([np.floor(np.amin(intensities) * 1e2) / 1e2, np.ceil(np.amax(intensities) * 1e2) / 1e2])
for item in ax.get_yticklabels():
    item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)

# Plot threshold energy vs. duration
fig3, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$duration \ (ms)$", fontsize=28)
ax.set_ylabel("$Energy \ (mJ/cm^2)$", fontsize=28)
ax.scatter(durations, energies, color='black', s=100)
ax.set_xlim(np.amin(durations) - 0.1 * Trange, np.amax(durations) + 0.1 * Trange)
for item in ax.get_yticklabels():
    item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)


plt.show()
