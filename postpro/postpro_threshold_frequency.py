#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-30 21:48:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-28 14:18:23

""" Test relationship between stimulus frequency and minimum acoustic intensity
for AP generation. """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from PointNICE.utils import ImportExcelCol, load_BLS_params

# Load NICE parameters
params = load_BLS_params()
biomech = params['biomech']
ac_imp = biomech['rhoL'] * biomech['c']  # Rayl

# Import data
xls_file = "C:/Users/admin/Desktop/Model output/NBLS titration frequency 30ms/nbls_log_titration_frequency_30ms.xlsx"
sheet = 'Data'
f = ImportExcelCol(xls_file, sheet, 'E', 2) * 1e3  # Hz
A = ImportExcelCol(xls_file, sheet, 'F', 2) * 1e3  # Pa
T = ImportExcelCol(xls_file, sheet, 'G', 2) * 1e-3  # s
N = ImportExcelCol(xls_file, sheet, 'Q', 2)

# Convert to appropriate units
frequencies = f * 1e-6  # MHz
amplitudes = A * 1e-3  # kPa
intensities = A**2 / (2 * ac_imp) * 1e-4  # W/cm2

# Plot threshold amplitude vs. duration
fig1, ax = plt.subplots(figsize=(12, 9))
ax.set_xscale('log')
ax.set_xlabel("$Frequency \ (MHz)$", fontsize=28)
ax.set_ylabel("$Amplitude \ (kPa)$", fontsize=28)
ax.scatter(frequencies, amplitudes, color='black', s=100)
ax.set_xlim(1.5e-1, 5e0)
ax.set_xscale('log')
ax.set_xticks([0.2, 1, 4])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_yticks([np.floor(np.amin(amplitudes)), np.ceil(np.amax(amplitudes))])
for item in ax.get_yticklabels():
    item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)


# Plot threshold intensity vs. duration
fig2, ax = plt.subplots(figsize=(12, 9))
ax.set_xscale('log')
ax.set_xlabel("$Frequency \ (MHz)$", fontsize=28)
ax.set_ylabel("$Intensity \ (W/cm^2)$", fontsize=28)
ax.scatter(frequencies, intensities, color='black', s=100)
ax.set_xlim(1.5e-1, 5e0)
ax.set_xscale('log')
ax.set_xticks([0.2, 1, 4])
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_yticks([np.floor(np.amin(intensities) * 1e2) / 1e2, np.ceil(np.amax(intensities) * 1e2) / 1e2])
for item in ax.get_yticklabels():
    item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)

plt.show()
