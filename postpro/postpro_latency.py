#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-31 10:10:41
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-21 16:07:33

""" Test relationship between stimulus intensity and response latency. """

import numpy as np
import matplotlib.pyplot as plt

from PySONIC.utils import ImportExcelCol, Presssure2Intensity


# Define import settings
xls_file = "C:/Users/admin/Desktop/Model output/NBLS spikes 0.35MHz/nbls_log_spikes_0.35MHz.xlsx"
sheet = 'Data'

# Import data
f = ImportExcelCol(xls_file, sheet, 'E', 2) * 1e3  # Hz
A = ImportExcelCol(xls_file, sheet, 'F', 2) * 1e3  # Pa
T = ImportExcelCol(xls_file, sheet, 'G', 2) * 1e-3  # s
N = ImportExcelCol(xls_file, sheet, 'Q', 2)
L = ImportExcelCol(xls_file, sheet, 'R', 2)  # ms

# Retrieve unique values of latencies (for min. 2 spikes) and corresponding amplitudes
iremove = np.where(N < 2)[0]
A_true = np.delete(A, iremove)
L_true = np.delete(L, iremove).astype(np.float)
latencies, indices = np.unique(L_true, return_index=True)
amplitudes = A_true[indices]

# Convert amplitudes to intensities
intensities = Pressure2Intensity(amplitudes) * 1e-4  # W/cm2

# Plot latency vs. amplitude
fig1, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$Amplitude \ (kPa)$", fontsize=28)
ax.set_ylabel("$Latency \ (ms)$", fontsize=28)
ax.scatter(amplitudes * 1e-3, latencies, color='black', s=100)
for item in ax.get_yticklabels():
        item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)


# Plot latency vs. intensity
fig2, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$Intensity \ (W/cm^2)$", fontsize=28)
ax.set_ylabel("$Latency \ (ms)$", fontsize=28)
ax.scatter(intensities, latencies, color='black', s=100)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticks([25, 35, 55, 65])
for item in ax.get_yticklabels():
        item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)

plt.show()
