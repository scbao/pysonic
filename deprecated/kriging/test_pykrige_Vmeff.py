#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-15 15:59:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-04-26 12:08:41

''' Fit a kriging model to a discrete 2D map of effective potentials
    for various charges and acoustic amplitudes, and use kriging predictor
    to generate a new 2D map of effective potentials within the original input range. '''

import os
import re
import ntpath
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.spatial.distance import pdist, squareform
from utils import OpenFilesDialog, rescale, rmse
from pykrige.ok import OrdinaryKriging
import pykrige.kriging_tools as kt


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


# Select data files (PKL)
lookup_root = '../Output/lookups 0.35MHz charge extended/'
lookup_absroot = os.path.abspath(lookup_root)
lookup_filepaths = OpenFilesDialog(lookup_absroot, 'pkl')
rgxp = re.compile('lookups_a(\d*.\d*)nm_f(\d*.\d*)kHz_A(\d*.\d*)kPa_dQ(\d*.\d*)nC_cm2.pkl')

# Set data variable and Kriging parameters
# varinf = Variable('\\alpha_{m, eff}', 'ms^{-1}', 'alpha_m_eff', 1e-3, 1e-10)
varinf = Variable('V_{m, eff}', 'mV', 'V_eff', 1., 1e-8)
nQ_sparse_target = 30
namps_sparse_target = 10

plot_all = True

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

    # Resample arrays
    print('resampling arrays')
    assert nQ_sparse_target <= nQ and namps_sparse_target <= namps
    Qm_sampling_factor = int(nQ / nQ_sparse_target)
    amps_sampling_factor = int(namps / namps_sparse_target)
    Qm_sparse = Qm[::Qm_sampling_factor]
    amps_sparse = amps[::amps_sampling_factor]
    nQ_sparse = Qm_sparse.size
    namps_sparse = amps_sparse.size
    var_sparse = var[::amps_sampling_factor, ::Qm_sampling_factor]
    Qmin_sparse = np.amin(Qm_sparse)
    Qmax_sparse = np.amax(Qm_sparse)
    Amin_sparse = np.amin(amps_sparse)
    Amax_sparse = np.amax(amps_sparse)
    print('Sparse data:', nQ_sparse, 'charges,', namps_sparse, 'amplitudes')

    # Normalize and serialize
    print('normalizing and serializing sparse data')
    Qm_sparse_norm = rescale(Qm_sparse, Qmin_sparse, Qmax_sparse)
    amps_sparse_norm = rescale(amps_sparse, Amin_sparse, Amax_sparse)
    Qm_sparse_norm_grid, amps_sparse_norm_grid = np.meshgrid(Qm_sparse_norm, amps_sparse_norm)
    Qm_sparse_norm_ser = np.reshape(Qm_sparse_norm_grid, nQ_sparse * namps_sparse)
    amps_sparse_norm_ser = np.reshape(amps_sparse_norm_grid, nQ_sparse * namps_sparse)
    var_sparse_ser = np.reshape(var_sparse, nQ_sparse * namps_sparse)

    # Compute normalized distance matrix and data semivariogram
    # print('computing normalized distance matrix and data semi-variogram')
    # norm_dist = squareform(pdist(np.array([amps_sparse_norm_ser, Qm_sparse_norm_ser]).transpose()))
    # N = norm_dist.shape[0]
    # norm_dist_ub = 1.6
    # assert np.amax(norm_dist) < norm_dist_ub,\
    #     'Error: max normalized distance greater than semi-variogram upper bound'
    # bw = 0.1  # lag bandwidth
    # lags = np.arange(0, 1.6, bw)  # lag array
    # nlags = lags.size
    # sv = np.empty(nlags)
    # for k in range(nlags):
    #     # print('lag = ', lags[k])
    #     Z = list()
    #     for i in range(N):
    #         for j in range(i + 1, N):
    #             if norm_dist[i, j] >= lags[k] - bw and norm_dist[i, j] <= lags[k] + bw:
    #                 Z.append((var_sparse_ser[i] - var_sparse_ser[j])**2.0)
    #     sv[k] = np.sum(Z) / (2.0 * len(Z))


    # Fit kriging model
    print('fitting kriging model to sparse data')
    OK = OrdinaryKriging(amps_sparse_norm_ser, Qm_sparse_norm_ser, var_sparse_ser,
                         variogram_model='linear')

    # Proof-of-concept: dummy prediction at known values of charge and amplitude
    print('re-computing sparse data from kriging predictor')
    var_sparse_krig, _ = OK.execute('grid', rescale(amps_sparse, Amin_sparse, Amax_sparse),
                                    rescale(Qm_sparse, Qmin_sparse, Qmax_sparse))
    var_sparse_krig = var_sparse_krig.transpose()
    var_sparse_max_abs_error = np.amax(np.abs(var_sparse - var_sparse_krig)) * varinf.factor
    assert var_sparse_max_abs_error < varinf.max_error,\
        'High Kriging error in training set ({:.2e} {})'.format(var_sparse_max_abs_error,
                                                                varinf.unit)

    # Predict data at unknown values
    print('re-computing original data from kriging predictor')
    var_krig, var_krig_ss = OK.execute('grid', rescale(amps, Amin, Amax), rescale(Qm, Qmin, Qmax))
    var_krig = var_krig.transpose()
    var_krig_ss = var_krig_ss.transpose()
    var_krig_std = np.sqrt(var_krig_ss)
    var_krig_std_min = np.amin(var_krig_std)
    var_krig_std_max = np.amax(var_krig_std)
    varmin = np.amin([varmin, np.amin(var_krig)])
    varmax = np.amin([varmax, np.amax(var_krig)])
    var_levels = np.linspace(varmin, varmax, 20) * varinf.factor
    var_abs_diff = np.abs(var - var_krig)
    var_abs_diff_max = np.amax(var_abs_diff)
    var_diff_levels = np.linspace(0., np.amax(var_abs_diff), 20) * varinf.factor
    var_std_levels = np.linspace(0., np.amax(var_krig_std_max), 20) * varinf.factor

    # Compare original and predicted profiles
    print('comparing original and predicted profiles')
    var_rmse = rmse(var, var_krig) * varinf.factor
    var_max_abs_error = np.amax(np.abs(var - var_krig)) * varinf.factor
    print('RMSE = {:.2f} {}, MAE = {:.2f} {}'.format(var_rmse, varinf.unit,
                                                     var_max_abs_error, varinf.unit))

    # Plotting
    print('plotting')

    mymap = cm.get_cmap('viridis')
    sm_amp = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Amin * 1e-3, Amax * 1e-3))
    sm_amp._A = []
    sm_var = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(varmin * varinf.factor,
                                                                  varmax * varinf.factor))
    sm_var._A = []
    sm_var_diff = plt.cm.ScalarMappable(cmap=mymap,
                                        norm=plt.Normalize(0., var_abs_diff_max * varinf.factor))
    sm_var_diff._A = []
    sm_var_std = plt.cm.ScalarMappable(cmap=mymap,
                                      norm=plt.Normalize(var_krig_std_min * varinf.factor,
                                                         var_krig_std_max * varinf.factor))
    sm_var_std._A = []

    if plot_all:


        # True function map
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
        ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
        ax.set_title('$' + varinf.name + '(Q_m,\ A_{drive})$ map', fontsize=20)
        ax.contourf(Qm * 1e5, amps * 1e-3, var * varinf.factor, levels=var_levels, cmap='viridis')
        xgrid, ygrid, = np.meshgrid(Qm * 1e5, amps * 1e-3)
        ax.scatter(xgrid, ygrid, c='black', s=5)
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
        fig.add_axes()
        fig.colorbar(sm_var, cax=cbar_ax)
        cbar_ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)

        # True function profiles
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
        ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
        ax.set_title('$' + varinf.name + '(Q_m)$ for different amplitudes', fontsize=20)
        for i in range(namps):
            ax.plot(Qm * 1e5, var[i, :] * varinf.factor, c=mymap(rescale(amps[i], Amin, Amax)))
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
        fig.add_axes()
        fig.colorbar(sm_amp, cax=cbar_ax)
        cbar_ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)

        # Sparse function profiles
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
        # ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
        # ax.set_title('sparse $' + varinf.name + '(Q_m)$ for different amplitudes', fontsize=20)
        # for i in range(namps_sparse):
        #     ax.plot(Qm_sparse * 1e5, var_sparse[i, :] * varinf.factor,
        #             c=mymap(rescale(amps_sparse[i], Amin, Amax)))
        # fig.subplots_adjust(right=0.85)
        # cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
        # fig.add_axes()
        # fig.colorbar(sm_amp, cax=cbar_ax)
        # cbar_ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)

        # 3: sparse var(Qm, Adrive) scattered map
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
        # ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
        # ax.set_title('sparse $' + varinf.name + '(Q_m,\ A_{drive})$ scattered map', fontsize=20)
        # xgrid, ygrid, = np.meshgrid(Qm_sparse * 1e5, amps_sparse * 1e-3)
        # ax.scatter(xgrid, ygrid, c=var_sparse * varinf.factor, cmap='viridis')
        # fig.subplots_adjust(right=0.85)
        # cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
        # fig.add_axes()
        # fig.colorbar(sm_var, cax=cbar_ax)
        # cbar_ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)

        # # 4: data semivariogram
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.set_xlabel('Normalized lag', fontsize=20)
        # ax.set_ylabel('Semivariance', fontsize=20)
        # ax.set_title('Semivariogram', fontsize=20)
        # ax.plot(lags, sv, '.-')

        # Estimate map
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
        ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
        ax.set_title('$' + varinf.name + '(Q_m,\ A_{drive})$ estimate map', fontsize=20)
        ax.contourf(Qm * 1e5, amps * 1e-3, var_krig * varinf.factor, levels=var_levels,
                    cmap='viridis')
        xgrid, ygrid, = np.meshgrid(Qm_sparse * 1e5, amps_sparse * 1e-3)
        ax.scatter(xgrid, ygrid, c='black', s=5)
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
        fig.add_axes()
        fig.colorbar(sm_var, cax=cbar_ax)
        cbar_ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)


        # 5: Prediction: more dense Vm_krig(Qm) plots for each Adrive
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
        ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$', fontsize=20)
        ax.set_title('Kriging: prediction of original $' + varinf.name + '(Q_m)$ profiles',
                     fontsize=20)
        for i in range(namps):
            ax.plot(Qm * 1e5, var_krig[i, :] * varinf.factor, c=mymap(rescale(amps[i], Amin, Amax)))
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
        fig.add_axes()
        fig.colorbar(sm_amp, cax=cbar_ax)
        cbar_ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)

    # 6: Vm(Qm, Adrive) kriging error contour map
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_title('Kriging error: $' + varinf.name + '(Q_m,\ A_{drive})$ contour map', fontsize=20)
    ax.contourf(Qm * 1e5, amps * 1e-3, var_abs_diff * varinf.factor, levels=var_diff_levels,
                cmap='viridis')
    xgrid, ygrid, = np.meshgrid(Qm_sparse * 1e5, amps_sparse * 1e-3)
    ax.scatter(xgrid, ygrid, c='black', s=5)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_var_diff, cax=cbar_ax)
    cbar_ax.set_ylabel('$' + varinf.name + '\ abs.\ error\ (' + varinf.unit + ')$', fontsize=20)

    # 6: Vm(Qm, Adrive) kriging predicted error contour map
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20)
    ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20)
    ax.set_title('Kriging predicted error: $' + varinf.name + '(Q_m,\ A_{drive})$ contour map', fontsize=20)
    ax.contourf(Qm * 1e5, amps * 1e-3, var_krig_std * varinf.factor, cmap='viridis')
    xgrid, ygrid, = np.meshgrid(Qm_sparse * 1e5, amps_sparse * 1e-3)
    ax.scatter(xgrid, ygrid, c='black', s=5)
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
    fig.add_axes()
    fig.colorbar(sm_var_std, cax=cbar_ax)
    cbar_ax.set_ylabel('$' + varinf.name + '\ abs.\ error\ (' + varinf.unit + ')$', fontsize=20)

    plt.show()
