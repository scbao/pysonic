#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-05 11:04:43
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-07-18 15:00:48

""" Test influence of acoustic amplitude and frequency on cavitation amplitude of embedded BLS. """

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from PointNICE.utils import ImportExcelCol, ConstructMatrix


def powerfit(X_, a_, b_, c_):
    """ Fitting function """
    x, y = X_
    return a_ * np.power(x, b_) * np.power(y, c_)


# Import data
xls_file = "C:/Users/admin/Desktop/Model output/BLS Z 32nm radius/10um embedding/bls_logZ_a32.0nm_d10.0um.xlsx"
sheet = 'Data'
f = ImportExcelCol(xls_file, sheet, 'E', 2) * 1e3
A = ImportExcelCol(xls_file, sheet, 'F', 2) * 1e3
eAmax = ImportExcelCol(xls_file, sheet, 'M', 2)

# Compute best power fit
p0 = 1e-3, 0.8, -0.5
popt, pcov = curve_fit(powerfit, (A, f), eAmax, p0)
(a, b, c) = popt
if a < 1e-4:
    a_str = '{:.2e}'.format(a)
else:
    a_str = '{:.4f}'.format(a)
print("global least-square power fit: eAmax = %s * A^%.2f * f^%.2f" % (a_str, b, c))

# Compute predicted data and associated error
eAmax_predicted = powerfit((A, f), a, b, c)
residuals = eAmax - eAmax_predicted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((eAmax - np.mean(eAmax))**2)
r_squared = 1 - (ss_res / ss_tot)
print("R-squared =  " + '{:.5f}'.format(r_squared))

# Reshape serialized data into 2 dimensions
(freqs, amps, eAmax_2D, nholes) = ConstructMatrix(f, A, eAmax)
nFreqs = freqs.size
nAmps = amps.size
fmax = np.amax(freqs)
fmin = np.amin(freqs)
Amax = np.amax(amps)
Amin = np.amin(amps)
print(str(nholes) + " hole(s) in reconstructed matrix")

# Create colormap
mymap = cm.get_cmap('jet')

# Plot areal strain vs. amplitude (with frequency color code)
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$A \ (kPa)$", fontsize=28)
ax.set_ylabel("$\epsilon_{A, max}$", fontsize=28)
for i in range(nFreqs):
    ax.plot(amps * 1e-3, eAmax_2D[i, :], c=mymap((freqs[i] - fmin) / (fmax - fmin)),
            label='f = ' + str(freqs[i] * 1e-3) + ' kHz')
for item in ax.get_yticklabels():
    item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)
sm_freq = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(fmin * 1e-3, fmax * 1e-3))
sm_freq._A = []
cbar = plt.colorbar(sm_freq)
cbar.ax.set_ylabel('$f \ (kHz)$', fontsize=28)
for item in cbar.ax.get_yticklabels():
    item.set_fontsize(24)

# Plot areal strain vs. frequency (with amplitude color code)
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$f \ (kHz)$", fontsize=28)
ax.set_ylabel("$\epsilon_{A, max}$", fontsize=28)
for j in range(nAmps):
    ax.plot(freqs * 1e-3, eAmax_2D[:, j], c=mymap((amps[j] - Amin) / (Amax - Amin)),
            label='A = ' + str(amps[j] * 1e-3) + ' kPa')
for item in ax.get_yticklabels():
    item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)
sm_amp = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Amin * 1e-3, Amax * 1e-3))
sm_amp._A = []
cbar = plt.colorbar(sm_amp)
cbar.ax.set_ylabel("$A \ (kPa)$", fontsize=28)
for item in cbar.ax.get_yticklabels():
    item.set_fontsize(24)


# 3D surface plot: eAmax = f(f,A)
if nholes == 0:
    X, Y = np.meshgrid(freqs * 1e-6, amps * 1e-6)
    fig = plt.figure(figsize=(12, 9))
    ax = fig.gca(projection=Axes3D.name)
    ax.plot_surface(X, Y, eAmax_2D, rstride=1, cstride=1, cmap=mymap, linewidth=0,
                    antialiased=False)
    ax.set_xlabel("$A \ (MPa)$", fontsize=24, labelpad=20)
    ax.set_ylabel("$f \ (MHz)$", fontsize=24, labelpad=20)
    ax.set_zlabel("$\epsilon_{A, max}$", fontsize=24, labelpad=20)
    ax.view_init(30, 135)
    for item in ax.get_yticklabels():
        item.set_fontsize(24)
    for item in ax.get_xticklabels():
        item.set_fontsize(24)
    for item in ax.get_zticklabels():
        item.set_fontsize(24)


# Plot optimal power fit vs. areal strain (with frequency color code)
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$%s\ A^{%.2f}\ f^{%.2f}$" % (a_str, b, c), fontsize=28)
ax.set_ylabel("$\epsilon_{A, max}$", fontsize=28)
for i in range(nFreqs):
    ax.scatter(a * amps**b * freqs[i]**c, eAmax_2D[i, :], s=40,
               c=mymap((freqs[i] - fmin) / (fmax - fmin)), label='f = %f kHz' % (freqs[i] * 1e-3))
ax.set_xlim([0.0, 1.1 * (a * Amax**b * fmin**c)])
ax.set_ylim([0.0, 1.1 * eAmax_2D[0, -1]])
ax.text(0.4 * eAmax_2D[0, -1], 0.9 * eAmax_2D[0, -1], "$R^2 = " + '{:.5f}'.format(r_squared) + "$",
        fontsize=24, color="black")
ax.set_xticks([0, np.round(np.amax(eAmax_2D) * 1e2) / 1e2])
ax.set_yticks([np.round(np.amax(eAmax_2D) * 1e2) / 1e2])
for item in ax.get_yticklabels():
        item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)
cbar = plt.colorbar(sm_freq)
cbar.ax.set_ylabel('$f \ (kHz)$', fontsize=28)
for item in cbar.ax.get_yticklabels():
        item.set_fontsize(24)

plt.show()
