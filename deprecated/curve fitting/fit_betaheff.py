#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-07 15:15:11
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-02-14 15:48:36

''' Detailed fitting strategy of the beta_h_eff profiles '''

import os
import ntpath
import re
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit
from utils import OpenFilesDialog, rescale, rmse, find_nearest


def gaussian(x, mu, sigma, a):
    return a * np.exp(-((x - mu) / (2 * sigma))**2)


def gauss3(x, a1, mu1, sig1, a2, mu2, sig2, a3, mu3, sig3):
    return gaussian(x, mu1, sig1, a1) + gaussian(x, mu2, sig2, a2) + gaussian(x, mu3, sig3, a3)


def sigmoid(x, x0, a, b):
    return 1 - 1 / (1 + np.abs((x - x0) / a)**b)


# Select data files (PKL)
lookup_root = '../Output/lookups extended 0.35MHz/'
lookup_absroot = os.path.abspath(lookup_root)
lookup_filepaths = OpenFilesDialog(lookup_absroot, 'pkl')
rgxp = re.compile('lookups_a(\d*.\d*)nm_f(\d*.\d*)kHz_A(\d*.\d*)kPa_dQ(\d*.\d*)nC_cm2.pkl')
plot_bool = 1

nQ = 300
baseline_ind = -1

# Check dialog output
if not lookup_filepaths:
    print('error: no lookup table selected')
else:
    print('importing betah_eff profiles from lookup tables')
    nfiles = len(lookup_filepaths)

    # Initialize coefficients matrices
    amps = np.empty(nfiles)
    betah_eff = np.empty((nfiles, nQ))

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
                Qm = lookup['Q']
                betah_eff[i, :] = lookup['beta_h_eff']

    if baseline_ind == -1:
        print('Error: no baseline profile selected')
    else:

        Amin = np.amin(amps)
        Amax = np.amax(amps)
        Qmin = np.amin(Qm)
        Qmax = np.amax(Qm)
        namps = nfiles

        i_trueQ_lb, trueQ_lb = find_nearest(Qm, -0.8)
        i_trueQ_ub, trueQ_ub = find_nearest(Qm, 0.4)

        # Baseline subtraction
        print('subtracting baseline (Adrive = 0) from profiles')
        betah_eff_sub = (betah_eff - betah_eff[baseline_ind, :])

        # Peaks detection on subtracted profiles
        print('finding peaks on subtracted profiles')
        betah_eff_sub_peaks = np.amax(np.abs(betah_eff_sub), axis=1)

        # Normalization
        print('normalizing subtracted profiles')
        betah_eff_sub_norm = betah_eff_sub[1:, :]\
            / betah_eff_sub_peaks[1:].reshape(namps - 1, 1)

        # Normalized profiles fitting
        print('fitting "mexican hat" to normalized betaheff-sub')
        betah_eff_sub_norm_fit = np.empty((namps - 1, nQ))
        params = np.empty((namps - 1, 9))
        for i in range(namps - 1):
            popt, _ = curve_fit(gauss3, Qm, betah_eff_sub_norm[i],
                                bounds=([0.0, -0.5, 0.0, -1.2, -0.2, 0., 0.0, 0.0, 0.0],
                                        [0.3, -0.2, np.inf, -0.8, 0.0, np.inf, 0.1, 0.1, np.inf]),
                                max_nfev=100000)
            betah_eff_sub_norm_fit[i, :] = gauss3(Qm, *popt)
            params[i, :] = np.asarray(popt)

        # Predict betah_eff profiles
        print('predicting betah_eff by reconstructing from fits')
        betah_eff_sub_predict = np.vstack((np.zeros(nQ), betah_eff_sub_norm_fit))\
            * betah_eff_sub_peaks.reshape(namps, 1)
        betah_eff_predict = betah_eff_sub_predict + betah_eff[baseline_ind, :]

        # Analyze prediction accuracy, in wide and realistic charge ranges
        betah_eff_trueQ = betah_eff[:, i_trueQ_lb:i_trueQ_ub]
        betah_eff_predict_trueQ = betah_eff_predict[:, i_trueQ_lb:i_trueQ_ub]
        betah_eff_diff = betah_eff_predict - betah_eff
        betah_eff_diff_trueQ = betah_eff_diff[:, i_trueQ_lb:i_trueQ_ub]
        betah_eff_maxdiff = np.amax(np.abs(betah_eff_diff), axis=1)
        betah_eff_maxdiff_trueQ = np.amax(np.abs(betah_eff_diff_trueQ), axis=1)
        betah_eff_rmse = np.empty(namps)
        betah_eff_rmse_trueQ = np.empty(namps)
        for i in range(namps):
            betah_eff_rmse[i] = rmse(betah_eff[i, :], betah_eff_predict[i, :])
            betah_eff_rmse_trueQ[i] = rmse(betah_eff_trueQ[i, :], betah_eff_predict_trueQ[i, :])


        if plot_bool == 1:

            # Plotting
            print('plotting')

            mymap = cm.get_cmap('jet')
            sm_amp = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Amin * 1e-3, Amax * 1e-3))
            sm_amp._A = []

            # 1: betah_eff
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\beta_{h,\ eff}\ (ms^{-1})$', fontsize=28)
            ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            for i in range(namps):
                ax.plot(Qm * 1e2, betah_eff[i, :] * 1e-3, c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()

            # 2: betah_eff_sub
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\beta_{h,\ eff-sub}\ (ms^{-1})$', fontsize=28)
            ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            for i in range(namps):
                ax.plot(Qm * 1e2, betah_eff_sub[i, :] * 1e-3,
                        c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()

            # 3: betah_eff_sub_peaks
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$A_{drive}\ (kPa)$', fontsize=28)
            ax.set_ylabel('$\\beta_{h,\ eff-sub-peaks}\ (ms^{-1})$', fontsize=28)
            ax.scatter(amps * 1e-3, betah_eff_sub_peaks * 1e-3, s=30, c='C0', label='data')
            # ax.plot(amps * 1e-3, betah_eff_sub_peaks_fit * 1e-3, c='C1', label='fit')
            ax.legend(fontsize=28)
            plt.tight_layout()

            # 5: betah_eff_sub_norm
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\beta_{h,\ eff-sub-norm}\ (-)$', fontsize=28)
            ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            ax.grid()
            for i in range(namps - 1):
                ax.plot(Qm * 1e2, betah_eff_sub_norm[i, :],
                        c=mymap(rescale(amps[i], Amin, Amax)))
            for i in range(namps - 1):
                ax.plot(Qm * 1e2, betah_eff_sub_norm_fit[i, :], '--',
                        c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()


            # 6: parameters
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$A_{drive}\ (kPa)$', fontsize=28)
            ax.set_ylabel('$\\beta_{h,\ eff-sub-norm}\ fit\ params$', fontsize=28)
            ax.plot(amps[1:] * 1e-3, params[:, 0], label='a1')
            ax.plot(amps[1:] * 1e-3, params[:, 1], label='mu1')
            ax.plot(amps[1:] * 1e-3, params[:, 2], label='sigma1')
            ax.plot(amps[1:] * 1e-3, params[:, 3], label='a2')
            ax.plot(amps[1:] * 1e-3, params[:, 4], label='mu2')
            ax.plot(amps[1:] * 1e-3, params[:, 5], label='sigma2')
            ax.plot(amps[1:] * 1e-3, params[:, 6], label='a3')
            ax.plot(amps[1:] * 1e-3, params[:, 7], label='mu3')
            ax.plot(amps[1:] * 1e-3, params[:, 8], label='sigma3')
            ax.grid()
            ax.legend(fontsize=28)


            # 7: betah_eff_predict
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\beta_{h,\ eff}\ prediction\ (ms^{-1})$', fontsize=28)
            ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            for i in range(namps):
                ax.plot(Qm * 1e2, betah_eff_predict[i, :] * 1e-3,
                        c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()


            # 8: betah_eff_predict - betah_eff
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\beta_{h,\ eff}\ difference\ (ms^{-1})$', fontsize=28)
            ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            for i in range(namps):
                ax.plot(Qm * 1e2, betah_eff_diff[i, :] * 1e-3,
                        c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()

            # 9: RMSE & max absolute error
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$A_{drive} \ (kPa)$', fontsize=28)
            ax.set_ylabel('$Error\ (ms^{-1})$', fontsize=28)
            ax.plot(amps * 1e-3, betah_eff_rmse * 1e-3, c='C0',
                    label='$RMSE\ -\ entire\ Q_m\ range$')
            ax.plot(amps * 1e-3, betah_eff_rmse_trueQ * 1e-3, c='C1',
                    label='$RMSE\ -\ realistic\ Q_m\ range$')
            ax.plot(amps * 1e-3, betah_eff_maxdiff * 1e-3, '--', c='C0',
                    label='$MAE\ -\ entire\ Q_m\ range$')
            ax.plot(amps * 1e-3, betah_eff_maxdiff_trueQ * 1e-3, '--', c='C1',
                    label='$MAE\ -\ realistic\ Q_m\ range$')
            ax.legend(fontsize=28)
            plt.tight_layout()

            plt.show()

