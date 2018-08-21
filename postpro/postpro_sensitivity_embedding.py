#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-05 11:04:43
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-21 16:07:34

""" Test influence of tissue embedding on BLS cavitation amplitude. """

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from PySONIC.utils import ImportExcelCol


def powerfit(x, a_, b_):
    """fitting function"""
    return a_ * np.power(x, b_)


# Import data
xls_file = "C:/Users/admin/Desktop/Model output/BLS Z tissue/bls_logZ_a32.0nm_embedding.xlsx"
sheet = 'Data'
rd = ImportExcelCol(xls_file, sheet, 'C', 2) * 1e-9
th = ImportExcelCol(xls_file, sheet, 'D', 2) * 1e-6
eAmax = ImportExcelCol(xls_file, sheet, 'M', 2)


# Filter out rows that don't match a specific radius value
a_ref = 32.0e-9  # (m)
imatch = np.where(rd == a_ref)
rd = rd[imatch]
th = th[imatch]
eAmax = eAmax[imatch]
print(str(imatch[0].size) + " values matching required radius")


# Compute best power fit for eAmax
popt, pcov = curve_fit(powerfit, th, eAmax)
(a, b) = popt
if a < 1e-4:
    a_str = '{:.2e}'.format(a)
else:
    a_str = '{:.4f}'.format(a)
print("global least-square power fit: eAmax = " + a_str + " * d^" + '{:.2f}'.format(b))


# Compute predicted data and associated error
eAmax_predicted = powerfit(th, a, b)
residuals = eAmax - eAmax_predicted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((eAmax - np.mean(eAmax))**2)
r_squared_eAmax = 1 - (ss_res / ss_tot)
print("R-squared =  " + '{:.5f}'.format(r_squared_eAmax))


# Plot areal strain vs. thickness (data and best fit)
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$d \ (um)$", fontsize=28)
ax.set_ylabel("$\epsilon_{A, max}$", fontsize=28)
ax.scatter(th * 1e6, eAmax, color='blue', linewidth=2, label="data")
ax.plot(th * 1e6, eAmax_predicted, '--', color='black', linewidth=2,
        label="model: $\epsilon_{A,max} \propto = d^{" + '{:.2f}'.format(b) + "}$")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[0] + 0.4 * (xlim[1] - xlim[0]), ylim[0] + 0.5 * (ylim[1] - ylim[0]),
        "$R^2 = " + '{:.5f}'.format(r_squared_eAmax) + "$", fontsize=28, color="black")
ax.legend(loc=1, fontsize=24)

# Show plots
plt.show()
