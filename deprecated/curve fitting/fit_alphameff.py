#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-01-17 11:41:53
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-02-15 15:44:31

''' Detailed fitting strategy of the alpha_m_eff profiles '''

import os
import ntpath
import re
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.optimize import curve_fit
from utils import OpenFilesDialog, rescale, rmse, find_nearest


def supraGauss(x, x0, a, b):
    return 2 / (np.exp(a * np.abs(x - x0)**b) + np.exp(-a * np.abs(x - x0)**b))


def absPow(x, x0, a, b, c):
    return a * np.abs(x - x0)**b + c


def sigmoid(x, x0, a, b):
    return 1 - 1 / (1 + np.abs((x - x0) / a)**b)


def hybridPowGauss(x, a, b, c, d):
    return supraGauss(x, 0.0, a, b) * absPow(x, 0.0, c, d, 1.0)


def hybridPowSigmoid(x, a, b, c, d):
    return sigmoid(x, 0.0, a, b) * absPow(x, 0.0, c, d, 0.0)


def piecewiseSigPowGauss(x, x0, a, b, c, d, e, f):
    y = np.empty(x.size)
    y[x < 0.] = sigmoid(x[x < 0.], x0, a, b)
    y[x >= 0.] = hybridPowGauss(x[x >= 0.], c, d, e, f)
    return y



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
    print('importing alpham_eff profiles from lookup tables')
    nfiles = len(lookup_filepaths)

    # Initialize coefficients matrices
    amps = np.empty(nfiles)
    alpham_eff = np.empty((nfiles, nQ))
    Vm_eff = np.empty((nfiles, nQ))

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
                alpham_eff[i, :] = lookup['alpha_m_eff']
                Vm_eff[i, :] = lookup['V_eff']

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
        alpham_eff_sub = (alpham_eff - alpham_eff[baseline_ind, :])
        Vm_eff_sub = (Vm_eff - Vm_eff[baseline_ind, :])

        # Suppressing Qm component
        print('dividing by Qm')
        alpham_eff_sub_even = alpham_eff_sub / Qm

        # Peaks fitting on even profiles
        print('fitting power law to peaks of even profiles')
        alpham_eff_sub_even_peaks = np.amax(alpham_eff_sub_even, axis=1)
        # pguess_peaks = (1e4, 1.6, 3.5, 0.4)
        # popt, _ = curve_fit(hybridPowSigmoid, amps, alpham_eff_sub_even_peaks, p0=pguess_peaks)
        # alpham_eff_sub_even_peaks_fit = hybridPowSigmoid(amps, *popt)

        # Normalization
        print('normalizing even profiles')
        alpham_eff_sub_even_norm = alpham_eff_sub_even[1:, :]\
            / alpham_eff_sub_even_peaks[1:].reshape(namps - 1, 1)

        # Normalized profiles fitting
        # print('fitting hybrid gaussian-power law to normalized alphameff-sub-even')
        # alpham_eff_sub_even_norm_fit = np.empty((namps - 1, nQ))
        # params = np.empty((namps - 1, 7))
        # for i in range(namps - 1):
        #     popt, _ = curve_fit(piecewiseSigPowGauss, Qm, alpham_eff_sub_even_norm[i, :],
        #                         bounds=([-np.infty, -1., -np.infty, 0., 0., -np.infty, 0.],
        #                                 [np.infty, 0., 0., np.infty, np.infty, 0., np.infty]))
        #     alpham_eff_sub_even_norm_fit[i, :] = piecewiseSigPowGauss(Qm, *popt)
        #     params[i, :] = np.asarray(popt)

        # Predict alpham_eff profiles
        # print('predicting alpham_eff by reconstructing from fits')
        # alpham_eff_sub_even_predict = np.vstack((np.zeros(nQ), alpham_eff_sub_even_norm_fit))\
        #     * alpham_eff_sub_even_peaks_fit.reshape(namps, 1)
        # alpham_eff_sub_predict = alpham_eff_sub_even_predict * Qm
        # alpham_eff_predict = alpham_eff_sub_predict + alpham_eff[baseline_ind, :]

        # # Analyze prediction accuracy, in wide and realistic charge ranges
        # alpham_eff_trueQ = alpham_eff[:, i_trueQ_lb:i_trueQ_ub]
        # alpham_eff_predict_trueQ = alpham_eff_predict[:, i_trueQ_lb:i_trueQ_ub]
        # alpham_eff_diff = alpham_eff_predict - alpham_eff
        # alpham_eff_diff_trueQ = alpham_eff_diff[:, i_trueQ_lb:i_trueQ_ub]
        # alpham_eff_maxdiff = np.amax(np.abs(alpham_eff_diff), axis=1)
        # alpham_eff_maxdiff_trueQ = np.amax(np.abs(alpham_eff_diff_trueQ), axis=1)
        # alpham_eff_rmse = np.empty(namps)
        # alpham_eff_rmse_trueQ = np.empty(namps)
        # for i in range(namps):
        #     alpham_eff_rmse[i] = rmse(alpham_eff[i, :], alpham_eff_predict[i, :])
        #     alpham_eff_rmse_trueQ[i] = rmse(alpham_eff_trueQ[i, :], alpham_eff_predict_trueQ[i, :])


        if plot_bool == 1:

            # Plotting
            print('plotting')

            mymap = cm.get_cmap('jet')
            sm_amp = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Amin * 1e-3, Amax * 1e-3))
            sm_amp._A = []

            # 1: alpham_eff
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\alpha_{m,\ eff}\ (ms^{-1})$', fontsize=28)
            ax.set_xlim(Qmin * 1e5, Qmax * 1e5)
            for i in range(namps):
                ax.plot(Qm * 1e5, alpham_eff[i, :] * 1e-3, c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()

            # 2: alpham_eff_sub
            fig, ax = plt.subplots(figsize=(21, 7))
            ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            ax.set_ylabel('$\\alpha_{m,\ eff-sub}\ (ms^{-1})$', fontsize=28)
            ax.set_xlim(Qmin * 1e5, Qmax * 1e5)
            for i in range(namps):
                ax.plot(Qm * 1e5, alpham_eff_sub[i, :] * 1e-3,
                        c=mymap(rescale(amps[i], Amin, Amax)))
            cbar = plt.colorbar(sm_amp)
            cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            plt.tight_layout()

            # # 3: alpham_eff_sub_even
            # fig, ax = plt.subplots(figsize=(21, 7))
            # ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            # ax.set_ylabel('$\\alpha_{m,\ eff-sub-even}\ (ms^{-1}\ cm^2/nC)$', fontsize=28)
            # ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            # for i in range(namps):
            #     ax.plot(Qm * 1e2, alpham_eff_sub_even[i, :] * 1e-3,
            #             c=mymap(rescale(amps[i], Amin, Amax)))
            # cbar = plt.colorbar(sm_amp)
            # cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            # plt.tight_layout()

            # # 4: alpham_eff_sub_even_peaks
            # fig, ax = plt.subplots(figsize=(21, 7))
            # ax.set_xlabel('$A_{drive}\ (kPa)$', fontsize=28)
            # ax.set_ylabel('$\\alpha_{m,\ eff-sub-even-peaks}\ (ms^{-1}\ cm^2/nC)$', fontsize=28)
            # ax.scatter(amps * 1e-3, alpham_eff_sub_even_peaks * 1e-3, s=30, c='C0', label='data')
            # ax.plot(amps * 1e-3, alpham_eff_sub_even_peaks_fit * 1e-3, c='C1', label='fit')
            # ax.legend(fontsize=28)
            # plt.tight_layout()

            # # 5: alpham_eff_sub_even_norm
            # fig, ax = plt.subplots(figsize=(21, 7))
            # ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            # ax.set_ylabel('$\\alpha_{m,\ eff-sub-even-norm}\ (-)$', fontsize=28)
            # ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            # ax.grid()
            # for i in range(namps - 1):
            #     ax.plot(Qm * 1e2, alpham_eff_sub_even_norm[i, :],
            #             c=mymap(rescale(amps[i], Amin, Amax)))
            # for i in range(namps - 1):
            #     ax.plot(Qm * 1e2, alpham_eff_sub_even_norm_fit[i, :], '--',
            #             c=mymap(rescale(amps[i], Amin, Amax)))
            # cbar = plt.colorbar(sm_amp)
            # cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            # plt.tight_layout()


            # # 6: piecewise function parameters
            # fig, ax = plt.subplots(figsize=(21, 7))
            # ax.set_xlabel('$A_{drive}\ (kPa)$', fontsize=28)
            # ax.set_ylabel('$\\alpha_{m,\ eff-sub-even-norm}\ fit\ params$', fontsize=28)
            # ax.plot(amps[1:] * 1e-3, params[:, 0], label='x0')
            # ax.plot(amps[1:] * 1e-3, params[:, 1], label='a')
            # ax.plot(amps[1:] * 1e-3, params[:, 2], label='b')
            # ax.plot(amps[1:] * 1e-3, params[:, 3], label='c')
            # ax.plot(amps[1:] * 1e-3, params[:, 4], label='d')
            # ax.plot(amps[1:] * 1e-3, params[:, 5], label='e')
            # ax.plot(amps[1:] * 1e-3, params[:, 6], label='f')
            # ax.grid()
            # ax.legend(fontsize=28)


            # # 7: alpham_eff_predict
            # fig, ax = plt.subplots(figsize=(21, 7))
            # ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            # ax.set_ylabel('$\\alpha_{m,\ eff}\ prediction\ (ms^{-1})$', fontsize=28)
            # ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            # for i in range(namps):
            #     ax.plot(Qm * 1e2, alpham_eff_predict[i, :] * 1e-3, linewidth=2,
            #             c=mymap(rescale(amps[i], Amin, Amax)))
            # cbar = plt.colorbar(sm_amp)
            # cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            # plt.tight_layout()


            # # 8: alpham_eff_predict - alpham_eff
            # # fig, ax = plt.subplots(figsize=(21, 7))
            # # ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
            # # ax.set_ylabel('$\\alpha_{m,\ eff}\ difference\ (ms^{-1})$', fontsize=28)
            # # ax.set_xlim(Qmin * 1e2, Qmax * 1e2)
            # # for i in range(namps):
            # #     ax.plot(Qm * 1e2, alpham_eff_diff[i, :] * 1e-3, linewidth=2,
            # #             c=mymap(rescale(amps[i], Amin, Amax)))
            # # cbar = plt.colorbar(sm_amp)
            # # cbar.ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=28)
            # # plt.tight_layout()

            # # 9: RMSE & max absolute error
            # fig, ax = plt.subplots(figsize=(21, 7))
            # ax.set_xlabel('$A_{drive} \ (kPa)$', fontsize=28)
            # ax.set_ylabel('$RMSE\ (ms^{-1})$', fontsize=28)
            # ax.plot(amps * 1e-3, alpham_eff_rmse * 1e-3, linewidth=2, c='C0',
            #         label='$RMSE\ -\ entire\ Q_m\ range$')
            # ax.plot(amps * 1e-3, alpham_eff_rmse_trueQ * 1e-3, linewidth=2, c='C1',
            #         label='$RMSE\ -\ realistic\ Q_m\ range$')
            # ax.plot(amps * 1e-3, alpham_eff_maxdiff * 1e-3, '--', linewidth=2, c='C0',
            #         label='$MAE\ -\ entire\ Q_m\ range$')
            # ax.plot(amps * 1e-3, alpham_eff_maxdiff_trueQ * 1e-3, '--', linewidth=2, c='C1',
            #         label='$MAE\ -\ realistic\ Q_m\ range$')
            # ax.legend(fontsize=28)
            # plt.tight_layout()

            plt.show()

