#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-05 11:04:43
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-07-18 15:00:50

""" Test influence of acoustic pressure amplitude on cavitation amplitude of exposed BLS. """

import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

sys.path.append('C:/Users/admin/Google Drive/PhD/NICE model/PointNICE')
from PointNICE.utils import ImportExcelCol


def powerfit(x, a_, b_):
    """ Fitting function. """
    return a_ * np.power(x, b_)


# Import data
xls_file = "C:/Users/admin/Desktop/Model output/BLS Z 32nm radius/0um embedding/bls_logZ_a32.0nm_d0.0um.xlsx"
sheet = 'Data'
A = ImportExcelCol(xls_file, sheet, 'F', 2) * 1e3
eAmax = ImportExcelCol(xls_file, sheet, 'M', 2)

# Sort data by increasing Pac amplitude
Asort = A.argsort()
A = A[Asort]
eAmax = eAmax[Asort]

# Compute best power fit for eAmax
popt, pcov = curve_fit(powerfit, A, eAmax)
(a, b) = popt
if a < 1e-4:
    a_str = '{:.2e}'.format(a)
else:
    a_str = '{:.4f}'.format(a)
print("global least-square power fit: eAmax = %s * A^%.2f" % (a_str, b))

# Compute predicted data and associated error
eAmax_predicted = powerfit(A, a, b)
residuals = eAmax - eAmax_predicted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((eAmax - np.mean(eAmax))**2)
r_squared_eAmax = 1 - (ss_res / ss_tot)
print("R-squared =  " + '{:.5f}'.format(r_squared_eAmax))

# Plot areal strain vs. acoustic pressure amplitude (data and best fit)
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$A \ (kPa)$", fontsize=28)
ax.set_ylabel("$\epsilon_{A, max}$", fontsize=28)
ax.scatter(A * 1e-3, eAmax, color='blue', linewidth=2, label="data")
ax.plot(A * 1e-3, eAmax_predicted, '--', color='black', linewidth=2,
        label="model: $\epsilon_{A,max} \propto = A^{" + '{:.2f}'.format(b) + "}$")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[0] + 0.1 * (xlim[1] - xlim[0]), ylim[0] + 0.6 * (ylim[1] - ylim[0]),
        "$R^2 = " + '{:.5f}'.format(r_squared_eAmax) + "$", fontsize=28, color="black")
ax.legend(loc=4, fontsize=24)
for item in ax.get_yticklabels():
        item.set_fontsize(24)
for item in ax.get_xticklabels():
    item.set_fontsize(24)

plt.show()
