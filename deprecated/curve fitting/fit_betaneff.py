#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-07 18:55:49
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-02-14 15:48:50

''' Detailed fitting strategy of the beta_n_eff profiles '''

import os
import ntpath
import re
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit
import scipy.special as sp
from utils import OpenFilesDialog, rescale, rmse, find_nearest


def skewed_gaussian(x, mu=0, sigma=1, alpha=0, a=1, c=0):
    normpdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.power((x - mu), 2) / (2 * np.power(sigma, 2))))
    normcdf = (0.5 * (1 + sp.erf((alpha * ((x - mu) / sigma)) / (np.sqrt(2)))))
    return 2 * a * normpdf * normcdf + c


def gaussian(x, mu, sigma, a):
    return a * np.exp(-((x - mu) / (2 * sigma))**2)


def Exponential(x, x0, b, c):
    return b * np.exp(c * (x - x0))


def Exp0(x, b, c):
    return Exponential(x, 0.0, b, c)


def hybridExpGauss(x, mu, sigma, a, b, c):
    return gaussian(x, mu, sigma, a) + Exponential(x, 0.0, b, -c)


def dualGauss(x, mu1, mu2, sigma1, sigma2, a1, a2):
    return gaussian(x, mu1, sigma1, a1) + gaussian(x, mu2, sigma2, a2)




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
    print('importing betan_eff profiles from lookup tables')
    nfiles = len(lookup_filepaths)

    # Initialize coefficients matrices
    amps = np.empty(nfiles)
    betan_eff = np.empty((nfiles, nQ))

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
                betan_eff[i, :] = lookup['beta_n_eff']

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
        betan_eff_sub = (betan_eff - betan_eff[baseline_ind, :])

        # Peaks fitting on even profiles
        print('fitting exponential law to profiles peaks')
        betan_eff_sub_peaks = np.amax(betan_eff_sub, axis=1)
        popt, _ = curve_fit(Exp0, amps, betan_eff_sub_peaks, p0=(1.8e14, 3e-5))
        betan_eff_sub_peaks_fit = Exp0(amps, *popt)

        # Normalization
        print('normalizing subtracted profiles')
        betan_eff_sub_norm = betan_eff_sub[1:, :]\
            / betan_eff_sub_peaks[1:].reshape(namps - 1, 1)

        # Normalized profiles fitting
        print('fitting hybrid gaussian-exp law to normalized betaneff-sub')
        betan_eff_sub_norm_fit = np.empty((namps - 1, nQ))
        params = np.empty((namps - 1, 6))
        for i in range(namps - 1):
            print(i)
            popt, _ = curve_fit(dualGauss, Qm, betan_eff_sub_norm[i],
                                bounds=([-np.infty, -np.infty, 0., 0., 0., 0.],
                                        [0., 0., np.infty, np.infty, np.infty, np.infty]),
                                max_nfev=100000)
            betan_eff_sub_norm_fit[i, :] = dualGauss(Qm, *popt)
            params[i, :] = np.asarray(popt)



        # Predict betan_eff profiles
        print('predicting betan_eff by reconstructing from fits')
        betan_eff_sub_predict = np.vstack((np.zeros(nQ), betan_eff_sub_norm_fit))\
            * betan_eff_sub_peaks_fit.reshape(namps, 1)
        betan_eff_predict = betan_eff_sub_predict + betan_eff[baseline_ind, :]

        # Analyze prediction accuracy, in wide and realistic charge ranges
        betan_eff_trueQ = betan_eff[:, i_trueQ_lb:i_trueQ_ub]
        betan_eff_predict_trueQ = betan_eff_predict[:, i_trueQ_lb:i_trueQ_ub]
        betan_eff_diff = betan_eff_predict - betan_eff
        betan_eff_diff_trueQ = betan_eff_diff[:, i_trueQ_lb:i_trueQ_ub]
        betan_eff_maxdiff = np.amax(np.abs(betan_eff_diff), axis=1)
        betan_eff_maxdiff_trueQ = np.amax(np.abs(betan_eff_diff_trueQ), axis=1)
        betan_eff_rmse = np.empty(namps)
        betan_eff_rmse_trueQ = np.empty(namps)
        for i in range(namps):
            betan_eff_rmse[i] = rmse(betan_eff[i, :], betan_eff_predict[i, :])
            betan_eff_rmse_trueQ[i] = rmse(betan_eff_trueQ[i, :], betan_eff_predict_trueQ[i, :])


        if plot_bool == 1:

            # Plotting
            print('plotting')

            mymap = cm.get_cmap('jet')
            sm_amp = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Amin * 1e-3, Amax * 1e-3))
            sm_amp._A = []

            # 1: betan_eff
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\beta_{n,\ eff}\ (ms^{-1})$', fontsize=28)
            ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            for i in range(namps):
                ax.plot(Qm * 1e2, betan_eff[i, :] * 1e-3, c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()

            # 2: betan_eff_sub
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\beta_{n,\ eff-sub}\ (ms^{-1})$', fontsize=28)
            ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            for i in range(namps):
                ax.plot(Qm * 1e2, betan_eff_sub[i, :] * 1e-3,
                        c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()

            # 3: betan_eff_sub_peaks
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$A_{drive}\ (kPa)$', fontsize=28)
            ax.set_ylabel('$\\beta_{n,\ eff-sub-peaks}\ (ms^{-1})$', fontsize=28)
            ax.scatter(amps * 1e-3, betan_eff_sub_peaks * 1e-3, s=30, c='C0', label='data')
            ax.plot(amps * 1e-3, betan_eff_sub_peaks_fit * 1e-3, c='C1', label='fit')
            ax.legend(fontsize=28)
            plt.tight_layout()

            # 5: betan_eff_sub_norm
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\beta_{n,\ eff-sub-norm}\ (-)$', fontsize=28)
            ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            ax.grid()
            for i in range(namps - 1):
                ax.plot(Qm * 1e2, betan_eff_sub_norm[i, :],
                        c=mymap(rescale(amps[i], Amin, Amax)))
            for i in range(namps - 1):
                ax.plot(Qm * 1e2, betan_eff_sub_norm_fit[i, :], '--',
                        c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()


            # 6: parameters
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$A_{drive}\ (kPa)$', fontsize=28)
            ax.set_ylabel('$\\beta_{n,\ eff-sub-norm}\ fit\ params$', fontsize=28)
            ax.plot(amps[1:] * 1e-3, params[:, 0], label='mu1')
            ax.plot(amps[1:] * 1e-3, params[:, 1], label='mu2')
            ax.plot(amps[1:] * 1e-3, params[:, 2], label='sigma1')
            ax.plot(amps[1:] * 1e-3, params[:, 3], label='sigma2')
            ax.plot(amps[1:] * 1e-3, params[:, 4], label='a1')
            ax.plot(amps[1:] * 1e-3, params[:, 5], label='a2')
            ax.grid()
            ax.legend(fontsize=28)


            # 7: betan_eff_predict
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\beta_{n,\ eff}\ prediction\ (ms^{-1})$', fontsize=28)
            ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            for i in range(namps):
                ax.plot(Qm * 1e2, betan_eff_predict[i, :] * 1e-3,
                        c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()


            # 8: betan_eff_predict - betan_eff
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\beta_{n,\ eff}\ difference\ (ms^{-1})$', fontsize=28)
            ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            for i in range(namps):
                ax.plot(Qm * 1e2, betan_eff_diff[i, :] * 1e-3,
                        c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()

            # 9: RMSE & max absolute error
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$A_{drive} \ (kPa)$', fontsize=28)
            ax.set_ylabel('$Error\ (ms^{-1})$', fontsize=28)
            ax.plot(amps * 1e-3, betan_eff_rmse * 1e-3, c='C0',
                    label='$RMSE\ -\ entire\ Q_m\ range$')
            ax.plot(amps * 1e-3, betan_eff_rmse_trueQ * 1e-3, c='C1',
                    label='$RMSE\ -\ realistic\ Q_m\ range$')
            ax.plot(amps * 1e-3, betan_eff_maxdiff * 1e-3, '--', c='C0',
                    label='$MAE\ -\ entire\ Q_m\ range$')
            ax.plot(amps * 1e-3, betan_eff_maxdiff_trueQ * 1e-3, '--', c='C1',
                    label='$MAE\ -\ realistic\ Q_m\ range$')
            ax.legend(fontsize=28)
            plt.tight_layout()

            plt.show()

