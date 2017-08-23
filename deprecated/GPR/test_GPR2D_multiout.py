#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-04-24 11:04:39
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-06-01 17:05:42

''' Predict nine different 2D coefficients profile using Gaussian Process Regression. '''

import os
import ntpath
import pickle
import re
import logging
import warnings
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from utils import OpenFilesDialog, rmse, lh2DWithCorners


# Define true function by interpolation from specific profile
def f(x):
    out = np.empty((x.shape[0], nvar))
    for k in range(nvar):
        out[:, k] = griddata(points, values[:, k], x, method='linear', rescale=True)
    return out


# Select data files (PKL)
lookup_root = '../Output/lookups 0.35MHz dense/'
lookup_absroot = os.path.abspath(lookup_root)
lookup_filepaths = OpenFilesDialog(lookup_absroot, 'pkl')
rgxp = re.compile('lookups_a(\d*.\d*)nm_f(\d*.\d*)kHz_A(\d*.\d*)kPa_dQ(\d*.\d*)nC_cm2.pkl')
outdir = 'C:/Users/admin/Desktop/GPRmultiout output/'

# Define logging settings and clear log file
logfile = outdir + 'GPR2D_multiout.log'
logging.basicConfig(filename=logfile, level=logging.DEBUG,
                    format='%(asctime)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
with open(logfile, 'w'):
    pass

lookups = ['V_eff', 'alpha_m_eff', 'beta_m_eff', 'alpha_h_eff', 'beta_h_eff', 'alpha_n_eff',
           'beta_n_eff', 'pinf_over_taup_eff', 'inv_taup_eff', 'ng_eff_on', 'ng_eff_off']
nvar = len(lookups)

max_errors = [1e0, 1e3, 1e3, 1e8, 1e2, 1e2, 1e4, 1e8, 1e9]
Ckernels = [C(100.0, (1.0, 500.0)), C(1e3, (1e0, 1e5)), C(1e3, (1e0, 1e5)), C(1e5, (1e0, 1e9)),
            C(1e2, (1e0, 1e4)), C(1e2, (1e0, 1e4)), C(1e4, (1e0, 1e6)), C(1e5, (1e0, 1e9)),
            C(1e5, (1e0, 1e9)), C(1e0, (1e-1, 1e1)), C(1e0, (1e-1, 1e1))]

factors = [1e0] + [1e-3 for i in range(8)] + [1e0 for i in range(2)]

units = ['mV'] + ['ms-1' for i in range(8)] + ['1e-22 mole' for i in range(2)]

plot_names = ['V_{m, eff}', '\\alpha_{m, eff}', '\\beta_{m, eff}', '\\alpha_{h, eff}',
              '\\beta_{h, eff}', '\\alpha_{n, eff}', '\\beta_{n, eff}',
              'p_{\\infty}/\\tau_p', '1/\\tau_p', 'n_{g,on}', 'n_{g,off}']

plot_units = ['mV'] + ['ms^{-1}' for i in range(8)] + ['10^{-22}\ mole' for i in range(2)]

# Check dialog output
if not lookup_filepaths:
    print('error: no lookup table selected')
else:
    print('importing lookup tables')
    logging.info('Files selected  - importing lookup tables')
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
            a = float(mo.group(1)) * 1e-9
            Fdrive = float(mo.group(2)) * 1e3
            Adrive = float(mo.group(3)) * 1e3
            dQ = float(mo.group(4)) * 1e-2
            amps[i] = Adrive
            if Adrive == 0:
                baseline_ind = i

            # Retrieve coefficients data
            with open(lookup_filepaths[i], 'rb') as fh:
                lookup_data = pickle.load(fh)
                if i == 0:
                    Qmfull = lookup_data['Q']
                    Qm = Qmfull[(Qmfull >= -80.0e-5) & (Qmfull <= 50.0e-5)]
                    nQ = np.size(Qm)
                    var = np.empty((nfiles, nQ, nvar))
                    for j in range(nvar):
                        varfull = lookup_data[lookups[j]]
                        var[i, :, j] = varfull[(Qmfull >= -80.0e-5) & (Qmfull <= 50.0e-5)]
                else:
                    Qmfull = lookup_data['Q']
                    if np.array_equal(Qm, Qmfull[(Qmfull >= -80.0e-5) & (Qmfull <= 50.0e-5)]):
                        for j in range(nvar):
                            varfull = lookup_data[lookups[j]]
                            var[i, :, j] = varfull[(Qmfull >= -80.0e-5) & (Qmfull <= 50.0e-5)]
                    else:
                        print('Error: charge vector not consistent')


    # Multiplying the gas molar contents
    var[:, :, -2] = var[:, :, -2] * 1e22
    var[:, :, -1] = var[:, :, -1] * 1e22

    # Compute data metrics
    namps = amps.size
    Amin = np.amin(amps)
    Amax = np.amax(amps)
    Qmin = np.amin(Qm)
    Qmax = np.amax(Qm)
    varmin = np.amin(var, axis=(0, 1))
    varmax = np.amax(var, axis=(0, 1))
    logstr = 'Initial data: {} charges, {} amplitudes'.format(nQ, namps)
    print(logstr)
    logging.info(logstr)

    # Define points for interpolation function
    Q_mesh, A_mesh = np.meshgrid(Qm, amps)
    points = np.column_stack([A_mesh.flatten(), Q_mesh.flatten()])
    values = var.reshape(namps * nQ, nvar)

    # Define algorithmic parameters
    n_iter_max = 100
    MAE_pred = []
    MAE_true = []
    RMSE_true = []

    # Define estimation grid
    nAest = 50
    nQest = 100
    Aest = np.linspace(Amin, Amax, nAest)
    Qest = np.linspace(Qmin, Qmax, nQest)
    Qest_mesh, Aest_mesh = np.meshgrid(Qest, Aest)
    x = np.column_stack([Aest_mesh.flatten(), Qest_mesh.flatten()])
    ytrue = f(x).reshape((nAest, nQest, nvar))
    logstr = 'Estimation grid: {} charges, {} amplitudes'.format(nQest, nAest)
    print(logstr)
    logging.info(logstr)

    # Define initial observation grid
    n0 = 24
    X0 = lh2DWithCorners(n0, (Amin, Amax), (Qmin, Qmax), 'center')
    Y0 = f(X0)
    logstr = 'Initial observation grid: Latin Hypercube ({} samples) with 4 corners'.format(n0 - 4)
    print(logstr)
    logging.info(logstr)

    # Instantiate Gaussian Process models
    logstr = 'Creating {} Gaussian Processes with scaled RBF Kernels'.format(nvar)
    print(logstr)
    logging.info(logstr)
    kernels = [Ck * RBF((1e4, 1e-4), ((1e3, 1e5), (1e-5, 1e-3))) for Ck in Ckernels]
    gprs = [GPR(kernel=k, n_restarts_optimizer=8, normalize_y=True) for k in kernels]


    # Fit to initial data using Maximum Likelihood Estimation of the parameters
    print('Step 0')
    logging.info('-------------------------- Initialization --------------------------')

    logstr = 'Fitting'
    print(logstr)
    logging.info(logstr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(nvar):
            gprs[i].fit(X0, Y0[:, i])


    # Make the prediction on the meshed x-axis (ask for MSE as well)
    logstr = 'Predicting'
    print(logstr)
    logging.info(logstr)
    y0 = np.empty((nAest * nQest, nvar))
    y0_std = np.empty((nAest * nQest, nvar))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(nvar):
            y0[:, i], y0_std[:, i] = gprs[i].predict(x, return_std=True)
    y0 = y0.reshape((nAest, nQest, nvar))
    y0_std = y0_std.reshape((nAest, nQest, nvar))
    MAE_pred.append(np.amax(y0_std, axis=(0, 1)))
    MAE_true.append(np.amax(np.abs(y0 - ytrue), axis=(0, 1)))
    RMSE_true.append(np.array([rmse(y0[:, :, i], ytrue[:, :, i]) for i in range(nvar)]))
    logging.info('Kernels:')
    for i in range(nvar):
        logging.info('   {}: {}'.format(lookups[i], gprs[i].kernel_))
    logging.info('predicted MAEs:')
    for i in range(nvar):
        logging.info('   {}: {:.2f} {}'.format(lookups[i], MAE_pred[-1][i] * factors[i], units[i]))
    logging.info('true MAEs:')
    for i in range(nvar):
        logging.info('   {}: {:.2f} {}'.format(lookups[i], MAE_true[-1][i] * factors[i], units[i]))

    # Copy initial data for iterations
    X = np.moveaxis(np.array([X0 for i in range(nvar)]), 0, -1)
    Y = Y0


    # Optimization
    print('Optimizing prediction by adding samples iteratively')
    n_iter = 0
    y_std = y0_std
    y_flat = np.empty((nAest * nQest, nvar))
    y_std_flat = np.empty((nAest * nQest, nvar))
    while n_iter < n_iter_max:
        print('Step', n_iter + 1)
        logstr = '-------------------------- Step {} --------------------------'.format(n_iter + 1)
        logging.info(logstr)

        print('Determining new samples')
        iMAEs = [np.argmax(y_std[:, :, i]) for i in range(nvar)]
        newX = x[iMAEs, :]
        X = np.concatenate((X, np.expand_dims(np.transpose(newX), axis=0)), axis=0)
        newY = np.expand_dims(np.array([f(newX[i, :])[0, i] for i in range(nvar)]), axis=0)
        Y = np.vstack((Y, newY))

        logstr = 'Fitting'
        print(logstr)
        logging.info(logstr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(nvar):
                gprs[i].fit(X[:, :, i], Y[:, i])

        logstr = 'Predicting'
        print(logstr)
        logging.info(logstr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(nvar):
                y_flat[:, i], y_std_flat[:, i] = gprs[i].predict(x, return_std=True)
        y = y_flat.reshape((nAest, nQest, nvar))
        y_std = y_std_flat.reshape((nAest, nQest, nvar))
        y_err_true = np.abs(y - ytrue)

        MAE_pred.append(np.amax(y_std, axis=(0, 1)))
        MAE_true.append(np.amax(np.abs(y - ytrue), axis=(0, 1)))
        RMSE_true.append(np.array([rmse(y[:, :, i], ytrue[:, :, i]) for i in range(nvar)]))
        logging.info('Kernels:')
        for i in range(nvar):
            logging.info('   {}: {}'.format(lookups[i], gprs[i].kernel_))
        logging.info('predicted MAEs:')
        for i in range(nvar):
            logging.info('   {}: {:.2f} {}'.format(lookups[i], MAE_pred[-1][i] * factors[i], units[i]))
        logging.info('true MAEs:')
        for i in range(nvar):
            logging.info('   {}: {:.2f} {}'.format(lookups[i], MAE_true[-1][i] * factors[i], units[i]))

        n_iter += 1

    # Saving
    gprs_dict = {}
    for i in range(nvar):
        gprs_dict[lookups[i]] = gprs[i]
    predictor_file = 'predictors_a{:.1f}nm_f{:.1f}kHz.pkl'.format(a * 1e9, Fdrive * 1e-3)
    logstr = 'Saving predictors dictionary in output file: {}'.format(predictor_file)
    logging.info(logstr)
    print(logstr)
    with open(outdir + predictor_file, 'wb') as fh:
        pickle.dump(gprs_dict, fh)

    # Plotting
    print('Plotting')
    mymap = cm.get_cmap('viridis')
    sm_amp = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Amin * 1e-3, Amax * 1e-3))
    sm_amp._A = []
    var_levels = np.array([np.linspace(varmin[i], varmax[i], 20) * factors[i] for i in range(nvar)])
    sm_var = [plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(varmin[i] * factors[i],
                                                                   varmax[i] * factors[i]))
              for i in range(nvar)]
    for smv in sm_var:
        smv._A = []
    varerr0_levels = np.array([np.linspace(0., MAE_pred[0][i], 20) * factors[i] for i in range(nvar)])
    varerr_levels = np.array([np.linspace(0., MAE_pred[-1][i], 20) * factors[i] for i in range(nvar)])
    sm_varerr0 = [plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(0.,
                                                                       MAE_pred[0][i] * factors[i]))
                  for i in range(nvar)]
    for smv in sm_varerr0:
        smv._A = []
    sm_varerr = [plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(0.,
                                                                      MAE_pred[-1][i] * factors[i]))
                 for i in range(nvar)]
    for smv in sm_varerr:
        smv._A = []



    for i in range(nvar):

        print('figure {}/{}'.format(i + 1, nvar))

        # RESPONSE SURFACE
        fig = plt.figure(figsize=(24, 12))

        # True function
        ax = fig.add_subplot(2, 3, 1, projection='3d')
        ax.set_title('True function', fontsize=20)
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=18)
        ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=18)
        ax.set_zlabel('$' + plot_names[i] + '\ (' + plot_units[i] + ')$', fontsize=18)
        ax.xaxis._axinfo['label']['space_factor'] = 3.0
        ax.yaxis._axinfo['label']['space_factor'] = 3.0
        ax.zaxis._axinfo['label']['space_factor'] = 3.0
        ax.plot_surface(Qest_mesh * 1e5, Aest_mesh * 1e-3, ytrue[:, :, i] * factors[i], cmap=mymap)

        # Initial prediction
        ax = fig.add_subplot(2, 3, 2, projection='3d')
        ax.set_title('Initial prediction', fontsize=20)
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=18)
        ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=18)
        ax.set_zlabel('$' + plot_names[i] + '\ (' + plot_units[i] + ')$', fontsize=18)
        ax.xaxis._axinfo['label']['space_factor'] = 3.0
        ax.yaxis._axinfo['label']['space_factor'] = 3.0
        ax.zaxis._axinfo['label']['space_factor'] = 3.0
        ax.plot_surface(Qest_mesh * 1e5, Aest_mesh * 1e-3, y0[:, :, i] * factors[i], cmap=mymap)

        # Final prediction
        ax = fig.add_subplot(2, 3, 3, projection='3d')
        ax.set_title('Final prediction', fontsize=20)
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=18)
        ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=18)
        ax.set_zlabel('$' + plot_names[i] + '\ (' + plot_units[i] + ')$', fontsize=18)
        ax.xaxis._axinfo['label']['space_factor'] = 3.0
        ax.yaxis._axinfo['label']['space_factor'] = 3.0
        ax.zaxis._axinfo['label']['space_factor'] = 3.0
        ax.plot_surface(Qest_mesh * 1e5, Aest_mesh * 1e-3, y[:, :, i] * factors[i], cmap=mymap)

        # Sampling map
        ax = fig.add_subplot(2, 3, 4)
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=20, labelpad=10)
        ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=20, labelpad=10)
        ax.set_title('Sampling map', fontsize=20)
        ax.contourf(Qest * 1e5, Aest * 1e-3, ytrue[:, :, i] * factors[i], levels=var_levels[i, :],
                    cmap='viridis')
        ax.scatter(X[:n0, 1, i] * 1e5, X[:n0, 0, i] * 1e-3, c='black', label='init. samples')
        ax.scatter(X[n0:, 1, i] * 1e5, X[n0:, 0, i] * 1e-3, c='red', label='added samples')
        ax.set_ylim(0.0, 1.15 * Amax * 1e-3)
        # ax.legend(fontsize=20, loc=3)
        ax.legend(fontsize=20, loc=9, ncol=2)

        # Initial error
        ax = fig.add_subplot(2, 3, 5, projection='3d')
        ax.set_title('Initial error', fontsize=20)
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=18)
        ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=18)
        ax.set_zlabel('$' + plot_names[i] + '\ (' + plot_units[i] + ')$', fontsize=18)
        ax.xaxis._axinfo['label']['space_factor'] = 3.0
        ax.yaxis._axinfo['label']['space_factor'] = 3.0
        ax.zaxis._axinfo['label']['space_factor'] = 3.0
        ax.plot_surface(Qest_mesh * 1e5, Aest_mesh * 1e-3,
                        (y0[:, :, i] - ytrue[:, :, i]) * factors[i], cmap=mymap)

        # Final error
        ax = fig.add_subplot(2, 3, 6, projection='3d')
        ax.set_title('Final error', fontsize=20)
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=18)
        ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=18)
        ax.set_zlabel('$' + plot_names[i] + '\ (' + plot_units[i] + ')$', fontsize=18)
        ax.xaxis._axinfo['label']['space_factor'] = 3.0
        ax.yaxis._axinfo['label']['space_factor'] = 3.0
        ax.zaxis._axinfo['label']['space_factor'] = 3.0
        ax.plot_surface(Qest_mesh * 1e5, Aest_mesh * 1e-3,
                        (y[:, :, i] - ytrue[:, :, i]) * factors[i], cmap=mymap)


        plt.tight_layout()
        fig.savefig(outdir + lookups[i] + '_surf.png', format='png')
        plt.close(fig)
