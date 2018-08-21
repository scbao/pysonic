#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-31 11:27:34
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-21 16:07:34

""" Test relationship between stimulus intensity spike rate. """

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from PySONIC.utils import ImportExcelCol, Pressure2Intensity


def fitfunc(x, a, b):
    """ Fitting function """
    return a * np.power(x, b)


# Import data
xls_file = "C:/Users/admin/Desktop/Model output/NBLS spikes 0.35MHz/nbls_log_spikes_0.35MHz.xlsx"
sheet = 'Data'
f = ImportExcelCol(xls_file, sheet, 'E', 2) * 1e3  # Hz
A = ImportExcelCol(xls_file, sheet, 'F', 2) * 1e3  # Pa
T = ImportExcelCol(xls_file, sheet, 'G', 2) * 1e-3  # s
N = ImportExcelCol(xls_file, sheet, 'Q', 2)
FR = ImportExcelCol(xls_file, sheet, 'S', 2)  # ms

# Retrieve available spike rates values (for min. 3 spikes) and corresponding amplitudes
iremove = np.where(N < 15)[0]
A_true = np.delete(A, iremove)
spikerates = np.delete(FR, iremove).astype(np.float)
amplitudes = np.delete(A, iremove)

# Convert amplitudes to intensities
intensities = Pressure2Intensity(amplitudes) * 1e-4  # W/cm2

# Power law least square fitting
popt, pcov = curve_fit(fitfunc, intensities, spikerates)
print('power product fit: FR = %.2f I^%.2f' % (popt[0], popt[1]))

# Compute predicted data and associated error
spikerates_predicted = fitfunc(intensities, popt[0], popt[1])
residuals = spikerates - spikerates_predicted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((spikerates - np.mean(spikerates))**2)
r_squared = 1 - (ss_res / ss_tot)
print("R-squared =  " + '{:.5f}'.format(r_squared))

# Plot latency vs. amplitude
fig1, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$Amplitude \ (kPa)$", fontsize=28)
ax.set_ylabel("$Spike\ Rate \ (spikes/ms)$", fontsize=28)
ax.scatter(amplitudes * 1e-3, spikerates, color='black')
ax.set_ylim(0, 1.1 * np.amax(spikerates))
for item in ax.get_yticklabels():
    item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)

# Plot latency vs. intensity
fig2, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$Intensity \ (W/cm^2)$", fontsize=28)
ax.set_ylabel("$Spike\ Rate \ (spikes/ms)$", fontsize=28)
ax.scatter(intensities, spikerates, color='black', label='$data$')
ax.plot(intensities, spikerates_predicted, color='blue',
        label='$%.2f\ I^{%.2f}$' % (popt[0], popt[1]))
ax.set_ylim(0, 1.1 * np.amax(spikerates))
ax.legend(fontsize=28)
for item in ax.get_yticklabels():
        item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)

plt.show()
