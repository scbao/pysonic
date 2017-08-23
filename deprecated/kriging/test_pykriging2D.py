#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-04-24 11:04:39
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-05-26 14:30:02

''' Predict a 1D Vmeff profile using the PyKriging module. '''

import os, ntpath
import pickle
import re
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pyKriging.krige import kriging
from utils import OpenFilesDialog, rescale, rmse


class Variable:
    ''' dummy class to contain information about the variable '''

    name = ''
    unit = ''
    lookup = ''
    factor = 1.
    max_error = 0.

    def __init__(self, var_name, var_unit, var_lookup, var_factor, var_max_error):
        self.name = var_name
        self.unit = var_unit
        self.factor = var_factor
        self.lookup = var_lookup
        self.max_error = var_max_error


# Set data variable and Kriging parameters
varinf = Variable('V_{m, eff}', 'mV', 'V_eff', 1., 1e-1)
# varinf = Variable('\\alpha_{m, eff}', 'ms^{-1}', 'alpha_m_eff', 1e-3, 1e1)
# varinf = Variable('\\beta_{m, eff}', 'ms^{-1}', 'beta_m_eff', 1e-3, 5e0)
# varinf = Variable('\\alpha_{h, eff}', 'ms^{-1}', 'alpha_h_eff', 1e-3, 1e1)
# varinf = Variable('\\beta_{h, eff}', 'ms^{-1}', 'beta_h_eff', 1e-3, 1e1)


# Define true function by interpolation from specific profile
def f(x):
    return griddata(points, values, x, method='linear', rescale=True)


# Select data files (PKL)
lookup_root = '../Output/lookups 0.35MHz charge extended/'
lookup_absroot = os.path.abspath(lookup_root)
lookup_filepaths = OpenFilesDialog(lookup_absroot, 'pkl')
rgxp = re.compile('lookups_a(\d*.\d*)nm_f(\d*.\d*)kHz_A(\d*.\d*)kPa_dQ(\d*.\d*)nC_cm2.pkl')
pltdir = 'C:/Users/admin/Desktop/PyKriging output/'

# Set data variable and Kriging parameters
varinf = Variable('V_{m, eff}', 'mV', 'V_eff', 1., 1.0)
# varinf = Variable('\\alpha_{m, eff}', 'ms^{-1}', 'alpha_m_eff', 1e-3, 1e4)
# varinf = Variable('\\beta_{m, eff}', 'ms^{-1}', 'beta_m_eff', 1e-3, 5e0)
# varinf = Variable('\\alpha_{h, eff}', 'ms^{-1}', 'alpha_h_eff', 1e-3, 1e1)
# varinf = Variable('\\beta_{h, eff}', 'ms^{-1}', 'beta_h_eff', 1e-3, 1e1)

# Check dialog output
if not lookup_filepaths:
    print('error: no lookup table selected')
else:
    print('importing lookup tables')
    nfiles = len(lookup_filepaths)
    amps = np.empty(nfiles)

    for i in range(nfiles):

        # Load lookup table
        lookup_filename = ntpath.basename(lookup_filepaths[i])
        mo = rgxp.fullmatch(lookup_filename)
        if not mo:
            print('Error: lookup file does not match regular expression pattern')
        else:
            # Retrieve stimulus parameters
            Fdrive = float(mo.group(2)) * 1e3
            Adrive = float(mo.group(3)) * 1e3
            dQ = float(mo.group(4)) * 1e-2
            amps[i] = Adrive
            if Adrive == 0:
                baseline_ind = i

            # Retrieve coefficients data
            with open(lookup_filepaths[i], 'rb') as fh:
                lookup = pickle.load(fh)
                if i == 0:
                    Qm = lookup['Q']
                    nQ = np.size(Qm)
                    var = np.empty((nfiles, nQ))
                    var[i, :] = lookup[varinf.lookup]
                else:
                    if np.array_equal(Qm, lookup['Q']):
                        var[i, :] = lookup[varinf.lookup]
                    else:
                        print('Error: charge vector not consistent')

    # Compute data metrics
    namps = amps.size
    Amin = np.amin(amps)
    Amax = np.amax(amps)
    Qmin = np.amin(Qm)
    Qmax = np.amax(Qm)
    varmin = np.amin(var)
    varmax = np.amax(var)
    print('Initial data:', nQ, 'charges,', namps, 'amplitudes')

    # Define points for interpolation function
    Q_mesh, A_mesh = np.meshgrid(Qm, amps)
    points = np.column_stack([A_mesh.flatten(), Q_mesh.flatten()])
    values = var.flatten()

    # Define algorithmic parameters
    n_iter_min = 10
    n_iter_max = 30
    MAE_pred = []
    MAE_true = []
    RMSE_true = []

    # Define estimation matrix
    nAest = 20
    nQest = 100
    print('Initial estimation matrix:', nQest, 'charges,', nAest, 'amplitudes')
    Aest = np.linspace(Amin, Amax, nAest)
    Qest = np.linspace(Qmin, Qmax, nQest)
    Qest_mesh, Aest_mesh = np.meshgrid(Qest, Aest)
    x = np.column_stack([Aest_mesh.flatten(), Qest_mesh.flatten()])
    ytrue = f(x).ravel().reshape((nAest, nQest))

    # Define initial observation matrix
    nAobs = 5
    nQobs = 20
    print('Initial estimation matrix:', nQobs, 'charges,', nAobs, 'amplitudes')
    Aobs = np.linspace(Amin, Amax, nAobs)
    Qobs = np.linspace(Qmin, Qmax, nQobs)
    Qobs_mesh, Aobs_mesh = np.meshgrid(Qobs, Aobs)
    X0 = np.column_stack([Aobs_mesh.flatten(), Qobs_mesh.flatten()])
    Y0 = f(X0).ravel()

    print('creating Kriging model')
    k = kriging(X0, Y0)

    print('initial training')
    k.train()

    print('predicting')
    y0 = np.array([k.predict(xx) for xx in x])
    y0 = y0.reshape((nAest, nQest))

    X = X0
    Y = Y0

    n_iter = 10
    for i in range(n_iter):
        print('Infill iteration {0} of {1}....'.format(i + 1, n_iter))
        newpoints = k.infill(2, method='error')
        for point in newpoints:
            newX = k.inversenormX(point)
            newY = f(newX)[0]
            print('adding point (({:.3f}, {:.3f}), {:.3f})'.format(
                newX[0] * 1e-3, newX[1] * 1e5, newY * varinf.factor))
            X = np.append(X, [newX], axis=0)
            Y = np.append(Y, newY)
            k.addPoint(newX, newY, norm=True)
        k.train()

    y = np.array([k.predict(xx) for xx in x])
    y = y.reshape((nAest, nQest))

    # Plotting
    mymap = cm.get_cmap('viridis')
    sm_amp = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Amin * 1e-3, Amax * 1e-3))
    sm_amp._A = []
    var_levels = np.linspace(varmin, varmax, 20) * varinf.factor
    sm_var = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(varmin * varinf.factor, varmax * varinf.factor))
    sm_var._A = []

    # True function profiles
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
    ax.set_title('True function profiles', fontsize=20)
    for i in range(nAest):
        ax.plot(Qest * 1e5, ytrue[i, :] * varinf.factor, c=mymap(rescale(Aest[i], Amin, Amax)))
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_amp, cax=cbar_ax)
    cbar_ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    fig.savefig(pltdir + 'fig1.png', format='png')

    # True function map
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_title('True function map', fontsize=20)
    ax.contourf(Qest * 1e5, Aest * 1e-3, ytrue * varinf.factor, levels=var_levels,
                cmap='viridis')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_var, cax=cbar_ax)
    cbar_ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
    fig.savefig(pltdir + 'fig2.png', format='png')

    # Initial estimation profiles
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
    ax.set_title('Initial estimation profiles', fontsize=20)
    for i in range(nAest):
        ax.plot(Qest * 1e5, y0[i, :] * varinf.factor, c=mymap(rescale(Aest[i], Amin, Amax)))
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_amp, cax=cbar_ax)
    cbar_ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    fig.savefig(pltdir + 'fig3.png', format='png')

    # Initial estimation map
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_title('Initial estimation map', fontsize=20)
    ax.contourf(Qest * 1e5, Aest * 1e-3, y0 * varinf.factor, levels=var_levels,
                cmap='viridis')
    ax.scatter(X0[:, 1] * 1e5, X0[:, 0] * 1e-3, c='black')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_var, cax=cbar_ax)
    cbar_ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
    fig.savefig(pltdir + 'fig4.png', format='png')

    # Final estimation profiles
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
    ax.set_title('Final estimation profiles', fontsize=20)
    for i in range(nAest):
        ax.plot(Qest * 1e5, y[i, :] * varinf.factor, c=mymap(rescale(Aest[i], Amin, Amax)))
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_amp, cax=cbar_ax)
    cbar_ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    fig.savefig(pltdir + 'fig7.png', format='png')

    # Final estimation map
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_title('Final estimation map', fontsize=20)
    ax.contourf(Qest * 1e5, Aest * 1e-3, y * varinf.factor, levels=var_levels,
                cmap='viridis')
    ax.scatter(X[:, 1] * 1e5, X[:, 0] * 1e-3, c='black')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_var, cax=cbar_ax)
    cbar_ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
    fig.savefig(pltdir + 'fig8.png', format='png')


plt.show()

