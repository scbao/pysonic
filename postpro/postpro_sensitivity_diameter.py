#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-05 11:04:43
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-07-18 15:00:40

""" Test influence of structure diameter on BLS cavitation amplitude. """

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from PointNICE.utils import ImportExcelCol


def f(x, a_, b_):
    """ Fitting function """
    return a_ * np.power(x, b_)


# Import data
xls_file = "C:/Users/admin/Desktop/Model output/BLS Z diameter/bls_logZ_diameter.xlsx"
sheet = 'Data'
rd = ImportExcelCol(xls_file, sheet, 'C', 2) * 1e-9
eAmax = ImportExcelCol(xls_file, sheet, 'M', 2)

# Discard outliers
rd = rd[0:-5]
eAmax = eAmax[0:-5]


# Compute best power fit for eAmax
popt, pcov = curve_fit(f, rd, eAmax)
(a, b) = popt
if a < 1e-4:
    a_str = '{:.2e}'.format(a)
else:
    a_str = '{:.4f}'.format(a)
print("global least-square power fit: eAmax = " + a_str + " * a^" + '{:.2f}'.format(b))

# Compute predicted data and associated error
eAmax_predicted = f(rd, a, b)
residuals = eAmax - eAmax_predicted
ss_res = np.sum(residuals**2)
ss_tot = np.sum((eAmax - np.mean(eAmax))**2)
r_squared_eAmax = 1 - (ss_res / ss_tot)
print("R-squared =  " + '{:.5f}'.format(r_squared_eAmax))
N = residuals.size
std_err = np.sqrt(ss_res / N)
print("standard error: sigma_err = " + str(std_err))

# Plot areal strain vs. in-plane radius (data and best fit)
fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlabel("$a \ (nm)$", fontsize=28)
ax.set_ylabel("$\epsilon_{A, max}$", fontsize=28)
ax.scatter(rd * 1e9, eAmax, color='blue', linewidth=2, label="data")
ax.plot(rd * 1e9, eAmax_predicted, '--', color='black', linewidth=2,
        label="model: $\epsilon_{A,max} \propto = a^{" + '{:.2f}'.format(b) + "}$")
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[0] + 0.1 * (xlim[1] - xlim[0]), ylim[0] + 0.6 * (ylim[1] - ylim[0]),
        "$R^2 = " + '{:.5f}'.format(r_squared_eAmax) + "$", fontsize=28, color="black")
ax.legend(loc=4, fontsize=24)

plt.show()
