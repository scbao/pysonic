#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-04-24 11:04:39
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-06-01 13:38:57

''' Predict a 2D Vmeff profile using Gaussian Process Regression. '''

import os, ntpath
import pickle
import re
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
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


# Define true function by interpolation from specific profiles
def f(x):
    return griddata(points, values, x, method='linear', rescale=True)


# Select data files (PKL)
lookup_root = '../Output/lookups 0.35MHz dense/'
lookup_absroot = os.path.abspath(lookup_root)
lookup_filepaths = OpenFilesDialog(lookup_absroot, 'pkl')
rgxp = re.compile('lookups_a(\d*.\d*)nm_f(\d*.\d*)kHz_A(\d*.\d*)kPa_dQ(\d*.\d*)nC_cm2.pkl')
pltdir = 'C:/Users/admin/Desktop/GPR output/'

# Set data variable and Kriging parameters
varinf = Variable('V_{m, eff}', 'mV', 'V_eff', 1., 1.0)
# varinf = Variable('\\alpha_{m, eff}', 'ms^{-1}', 'alpha_m_eff', 1e-3, 1e4)
# varinf = Variable('\\beta_{m, eff}', 'ms^{-1}', 'beta_m_eff', 1e-3, 5e0)
# varinf = Variable('\\alpha_{h, eff}', 'ms^{-1}', 'alpha_h_eff', 1e-3, 1e1)
# varinf = Variable('\\beta_{h, eff}', 'ms^{-1}', 'beta_h_eff', 1e-3, 1e1)
# varinf = Variable('\\alpha_{n, eff}', 'ms^{-1}', 'alpha_n_eff', 1e-3, 1e1)
# varinf = Variable('\\beta_{n, eff}', 'ms^{-1}', 'beta_n_eff', 1e-3, 1e1)
# varinf = Variable('(p_{\\infty}\ /\ \\tau_p)_{eff}', 'ms^{-1}', 'pinf_over_taup_eff', 1e-3, 1e1)
# varinf = Variable('(1\ /\ \\tau_p)_{eff}', 'ms^{-1}', 'inv_taup_eff', 1e-3, 1e1)
# varinf = Variable('n_{g,on}', 'mole', 'ng_eff_on', 1e22, 1e1)
# varinf = Variable('n_{g,off}', 'mole', 'ng_eff_off', 1e22, 1e1)

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

    np.savetxt('tmp.txt', np.transpose(var))
    quit()

    # Define points for interpolation function
    Q_mesh, A_mesh = np.meshgrid(Qm, amps)
    points = np.column_stack([A_mesh.flatten(), Q_mesh.flatten()])
    values = var.flatten()

    # Define algorithmic parameters
    n_iter_min = 10
    n_iter_max = 100
    MAE_pred = []
    MAE_true = []
    RMSE_true = []

    # Define estimation vector
    nAest = 50
    nQest = 100
    Aest = np.linspace(Amin, Amax, nAest)
    Qest = np.linspace(Qmin, Qmax, nQest)
    Qest_mesh, Aest_mesh = np.meshgrid(Qest, Aest)
    x = np.column_stack([Aest_mesh.flatten(), Qest_mesh.flatten()])
    ytrue = f(x).ravel().reshape((nAest, nQest))

    # Define initial observation vector
    nAobs = 5
    nQobs = 20
    Aobs = np.linspace(Amin, Amax, nAobs)
    Qobs = np.linspace(Qmin, Qmax, nQobs)
    Qobs_mesh, Aobs_mesh = np.meshgrid(Qobs, Aobs)
    X0 = np.column_stack([Aobs_mesh.flatten(), Qobs_mesh.flatten()])
    Y0 = f(X0).ravel()

    # np.savetxt('data_sparse.txt', np.column_stack([X0, Y0]), fmt='% .7e', delimiter='  ', newline='\n  ')
    # quit()


    # Instantiate a Gaussian Process model
    print('Creating Gaussian Process with RBF Kernel')
    kernel = C(100.0, (1.0, 500.0)) * RBF((1e4, 1e-4), ((1e3, 1e5), (1e-5, 1e-3)))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8, normalize_y=True)

    # Fit to initial data using Maximum Likelihood Estimation of the parameters
    print('Initial fitting')
    gp.fit(X0, Y0)
    X = X0
    Y = Y0

    # Make the prediction on the meshed x-axis (ask for MSE as well)
    print('Predicting over linear input range')
    y0, y0_std = gp.predict(x, return_std=True)
    y0 = y0.reshape((nAest, nQest))
    y0_std = y0_std.reshape((nAest, nQest))
    y0_err_true = np.abs(y0 - ytrue)
    MAE_pred.append(np.amax(y0_std))
    MAE_true.append(np.amax(np.abs(y0 - ytrue)))
    RMSE_true.append(rmse(y0, ytrue))
    print('Initialization: Kernel =', gp.kernel_)
    print('predicted MAE = {:.2f} {}, true MAE = {:.2f} {}'.format(MAE_pred[-1] * varinf.factor,
                                                                   varinf.unit,
                                                                   MAE_true[-1] * varinf.factor,
                                                                   varinf.unit))
    # Optimization
    print('Optimizing prediction by adding samples iteratively')
    n_iter = 0
    y_std = y0_std
    while n_iter < n_iter_max and (MAE_pred[-1] > varinf.max_error or n_iter < n_iter_min):
        new_X = x[np.argmax(y_std)]
        X = np.vstack((X, new_X))
        Y = np.append(Y, f(new_X))
        gp.fit(X, Y)
        y, y_std = gp.predict(x, return_std=True)
        y = y.reshape((nAest, nQest))
        y_std = y_std.reshape((nAest, nQest))
        y_err_true = np.abs(y - ytrue)
        MAE_pred.append(np.amax(y_std))
        MAE_true.append(np.amax(np.abs(y - ytrue)))
        RMSE_true.append(rmse(y, ytrue))
        print('step {}:'.format(n_iter + 1), 'Kernel =', gp.kernel_)
        print('predicted MAE = {:.2f} {}, true MAE = {:.2f} {}'.format(MAE_pred[-1] * varinf.factor,
                                                                       varinf.unit,
                                                                       MAE_true[-1] * varinf.factor,
                                                                       varinf.unit))
        n_iter += 1

    # Plotting
    mymap = cm.get_cmap('viridis')
    sm_amp = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Amin * 1e-3, Amax * 1e-3))
    sm_amp._A = []
    var_levels = np.linspace(varmin, varmax, 20) * varinf.factor
    sm_var = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(varmin * varinf.factor, varmax * varinf.factor))
    sm_var._A = []
    varerr0_levels = np.linspace(0., np.amax(y0_err_true), 20) * varinf.factor
    varerr_levels = np.linspace(0., np.amax(y_err_true), 20) * varinf.factor
    sm_varerr0 = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(0., np.amax(y0_err_true) * varinf.factor))
    sm_varerr0._A = []
    sm_varerr = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(0., np.amax(y_err_true) * varinf.factor))
    sm_varerr._A = []

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

    # Initial error profiles
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
    ax.set_title('Initial error profiles', fontsize=20)
    for i in range(nAest):
        ax.plot(Qest * 1e5, (y0[i, :] - ytrue[i, :]) * varinf.factor,
                c=mymap(rescale(Aest[i], Amin, Amax)))
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_amp, cax=cbar_ax)
    cbar_ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    fig.savefig(pltdir + 'fig5.png', format='png')

    # Initial error map
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_title('Initial error map', fontsize=20)
    ax.contourf(Qest * 1e5, Aest * 1e-3, y0_err_true * varinf.factor, levels=varerr0_levels,
                cmap='viridis')
    ax.scatter(X[:, 1] * 1e5, X[:, 0] * 1e-3, c='black')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_varerr0, cax=cbar_ax)
    cbar_ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
    fig.savefig(pltdir + 'fig6.png', format='png')

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

    # Final error profiles
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
    ax.set_title('Final error profiles', fontsize=20)
    for i in range(nAest):
        ax.plot(Qest * 1e5, (y[i, :] - ytrue[i, :]) * varinf.factor,
                c=mymap(rescale(Aest[i], Amin, Amax)))
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_amp, cax=cbar_ax)
    cbar_ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    fig.savefig(pltdir + 'fig9.png', format='png')

    # Final error map
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_title('Final error map', fontsize=20)
    ax.contourf(Qest * 1e5, Aest * 1e-3, y_err_true * varinf.factor, levels=varerr_levels,
                cmap='viridis')
    ax.scatter(X[:, 1] * 1e5, X[:, 0] * 1e-3, c='black')
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_varerr, cax=cbar_ax)
    cbar_ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
    fig.savefig(pltdir + 'fig10.png', format='png')

    # Error evolution
    fig, ax = plt.subplots()
    iters = np.linspace(0, n_iter, n_iter + 1)
    ax.plot(iters, np.array(MAE_true) * varinf.factor, label='true error')
    ax.plot(iters, np.array(MAE_pred) * varinf.factor, label='predicted error')
    ax.plot(iters, np.array(RMSE_true) * varinf.factor, label='true RMSE')
    ax.set_xlabel('# iterations', fontsize=20)
    ax.set_ylabel('Max. absolute error ($' + varinf.unit + ')$', fontsize=20)
    ax.set_title('Error evolution', fontsize=20)
    ax.legend(fontsize=20)
    fig.savefig(pltdir + 'fig11.png', format='png')

    # plt.show()
