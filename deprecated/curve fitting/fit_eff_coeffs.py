#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-01-15 18:08:06
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-01-20 10:18:50

''' Plot the profiles of the 9 charge-dependent "effective" HH coefficients,
along with a fitted mathematical expression '''


import os
import ntpath
import pickle
import matplotlib.pyplot as plt
import numpy as np
# import math as math
import scipy.special as sp
from scipy.optimize import curve_fit
from utils import OpenFilesDialog, rsquared





def fit_amn(x, a, b, c, d):
    # return a * c * (x - c - b) * np.exp((x - b) / c) - x + d
    return a * c**2 * sp.spence(1 - (-np.exp(-b / c) * (np.exp(x / c) - np.exp(b / c)))) + d


# --------------------------------------------------------------------

def gaus(x, a, x0, sigma):
    return a * np.exp(- (x - x0)**2 / (2 * sigma**2))


def compexp(x, a, b, c, d, e, f):
    return (a * x + b) / (c * np.exp(d * x + e) + f)


def expgrowth(x, x0, a):
    return np.exp(a * (x - x0))


def expdecay(x, x0, a):
    return np.exp(-a * (x - x0))


def sigmoid(x, x0, a, b):
    return a / (1 + np.exp(- b * (x - x0)))


def dualexp(x, x1, x2, a, b):
    return np.exp(a * (x - x1)) + np.exp(- b * (x - x2))


def dualregime(x, x0, a, b):
    return a * (x - x0) / (np.exp(- b * (x - x0)) - 1)


def skewed_gaussian(x, mu=0, sigma=1, alpha=0, a=1, c=0):
    normpdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.power((x - mu), 2) / (2 * np.power(sigma, 2))))
    normcdf = (0.5 * (1 + sp.erf((alpha * ((x - mu) / sigma)) / (np.sqrt(2)))))
    return 2 * a * normpdf * normcdf + c


# Select data files (PKL)
lookup_root = '../Output/lookups 0.35MHz linear amplitude/'
lookup_absroot = os.path.abspath(lookup_root)
lookup_filepath = OpenFilesDialog(lookup_absroot, 'pkl')

# Check dialog output
if not lookup_filepath:
    print('error: no lookup table selected')
elif len(lookup_filepath) > 1:
    print('error multiple lookup tables selected')
else:

    # Load lookup table
    lookup_filename = ntpath.basename(lookup_filepath[0])
    print('loading lookup table')
    with open(lookup_filepath[0], 'rb') as fh:
        lookup = pickle.load(fh)

        print('finding best fits with analytical expressions')

        # Vm_eff
        print('Vm_eff')
        z = np.polyfit(lookup['Q'], lookup['V_eff'], 3)
        p = np.poly1d(z)
        Veff_fit = p(lookup['Q'])
        r2 = rsquared(lookup['V_eff'], Veff_fit)
        fig, ax = plt.subplots(figsize=(21, 7))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
        ax.set_ylabel('$V_{m,\ eff}\ (mV)$', fontsize=28)
        ax.plot(lookup['Q'] * 1e2, lookup['V_eff'], linewidth=2, label='data')
        ax.plot(lookup['Q'] * 1e2, Veff_fit, linewidth=2, label='fit')
        ax.text(0.45, 0.9, '$R^2 = {:.5f}$'.format(r2), transform=ax.transAxes, fontsize=28)
        ax.legend()

        # alpha_m_eff
        print('alpha_m_eff')
        # z = np.polyfit(lookup['Q'], lookup['alpha_m_eff'], 5)
        # p = np.poly1d(z)
        # alpha_m_eff_fit = p(lookup['Q'])
        popt, _ = curve_fit(fit_amn, lookup['Q'], lookup['alpha_m_eff'], maxfev=100000)
        alpha_m_eff_fit = fit_amn(lookup['Q'], *popt)
        r2 = rsquared(lookup['alpha_m_eff'], alpha_m_eff_fit)
        fig, ax = plt.subplots(figsize=(21, 7))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
        ax.set_ylabel('$\\alpha_{m,\ eff}\ (ms^{-1})$', fontsize=28)
        ax.plot(lookup['Q'] * 1e2, lookup['alpha_m_eff'] * 1e-3, linewidth=2, label='data')
        ax.plot(lookup['Q'] * 1e2, alpha_m_eff_fit * 1e-3, linewidth=2, label='fit')
        ax.text(0.45, 0.9, '$R^2 = {:.5f}$'.format(r2), transform=ax.transAxes, fontsize=28)
        ax.legend()

        # beta_m_eff
        print('beta_m_eff')
        pguess = (-0.7, 0.2, 3, 5000)
        beta_m_eff_guess = skewed_gaussian(lookup['Q'], *pguess)
        popt, _ = curve_fit(skewed_gaussian, lookup['Q'], lookup['beta_m_eff'], p0=pguess)
        beta_m_eff_fit = skewed_gaussian(lookup['Q'], *popt)
        r2 = rsquared(lookup['beta_m_eff'], beta_m_eff_fit)
        fig, ax = plt.subplots(figsize=(21, 7))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
        ax.set_ylabel('$\\beta_{m,\ eff}\ (ms^{-1})$', fontsize=28)
        ax.plot(lookup['Q'] * 1e2, lookup['beta_m_eff'] * 1e-3, linewidth=2, label='data')
        ax.plot(lookup['Q'] * 1e2, beta_m_eff_fit * 1e-3, linewidth=2, label='fit')
        ax.plot(lookup['Q'] * 1e2, beta_m_eff_guess * 1e-3, linewidth=2, label='guess')
        ax.text(0.45, 0.9, '$R^2 = {:.5f}$'.format(r2), transform=ax.transAxes, fontsize=28)
        ax.legend()

        # alpha_h_eff
        print('alpha_h_eff')
        pguess = (-0.7, 0.2, 3, 20000)
        alpha_h_eff_guess = skewed_gaussian(lookup['Q'], *pguess)
        popt, _ = curve_fit(skewed_gaussian, lookup['Q'], lookup['alpha_h_eff'], p0=pguess)
        alpha_h_eff_fit = skewed_gaussian(lookup['Q'], *popt)
        r2 = rsquared(lookup['alpha_h_eff'], alpha_h_eff_fit)
        fig, ax = plt.subplots(figsize=(21, 7))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
        ax.set_ylabel('$\\alpha_{h,\ eff}\ (ms^{-1})$', fontsize=28)
        ax.plot(lookup['Q'] * 1e2, lookup['alpha_h_eff'] * 1e-3, linewidth=2, label='data')
        ax.plot(lookup['Q'] * 1e2, alpha_h_eff_fit * 1e-3, linewidth=2, label='fit')
        ax.plot(lookup['Q'] * 1e2, alpha_h_eff_guess * 1e-3, label='guess')
        ax.text(0.45, 0.9, '$R^2 = {:.5f}$'.format(r2), transform=ax.transAxes, fontsize=28)
        ax.legend()

        # beta_h_eff
        print('beta_h_eff')
        popt, _ = curve_fit(sigmoid, lookup['Q'], lookup['beta_h_eff'], p0=(-0.1, 4000, 20))
        beta_h_eff_fit = sigmoid(lookup['Q'], *popt)
        r2 = rsquared(lookup['beta_h_eff'], beta_h_eff_fit)
        fig, ax = plt.subplots(figsize=(21, 7))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
        ax.set_ylabel('$\\beta_{h,\ eff}\ (ms^{-1})$', fontsize=28)
        ax.plot(lookup['Q'] * 1e2, lookup['beta_h_eff'] * 1e-3, linewidth=2, label='data')
        ax.plot(lookup['Q'] * 1e2, beta_h_eff_fit * 1e-3, linewidth=2, label='fit')
        ax.text(0.45, 0.9, '$R^2 = {:.5f}$'.format(r2), transform=ax.transAxes, fontsize=28)
        ax.legend()

        # alpha_n_eff
        print('alpha_n_eff')
        popt, _ = curve_fit(gaus, lookup['Q'], lookup['alpha_n_eff'])
        alpha_n_eff_fit = gaus(lookup['Q'], *popt)
        r2 = rsquared(lookup['alpha_n_eff'], alpha_n_eff_fit)
        fig, ax = plt.subplots(figsize=(21, 7))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
        ax.set_ylabel('$\\alpha_{n,\ eff}\ (ms^{-1})$', fontsize=28)
        ax.plot(lookup['Q'] * 1e2, lookup['alpha_n_eff'] * 1e-3, linewidth=2, label='data')
        ax.plot(lookup['Q'] * 1e2, alpha_n_eff_fit * 1e-3, linewidth=2, label='fit')
        ax.text(0.45, 0.9, '$R^2 = {:.5f}$'.format(r2), transform=ax.transAxes, fontsize=28)
        ax.legend()

        # beta_n_eff
        print('beta_n_eff')
        popt, _ = curve_fit(expdecay, lookup['Q'], lookup['beta_n_eff'])
        beta_n_eff_fit = expdecay(lookup['Q'], *popt)
        r2 = rsquared(lookup['beta_n_eff'], beta_n_eff_fit)
        fig, ax = plt.subplots(figsize=(21, 7))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
        ax.set_ylabel('$\\beta_{n,\ eff}\ (ms^{-1})$', fontsize=28)
        ax.plot(lookup['Q'] * 1e2, lookup['beta_n_eff'] * 1e-3, linewidth=2, label='data')
        ax.plot(lookup['Q'] * 1e2, beta_n_eff_fit * 1e-3, linewidth=2, label='fit')
        ax.text(0.45, 0.9, '$R^2 = {:.5f}$'.format(r2), transform=ax.transAxes, fontsize=28)
        ax.legend()

        # pinf_over_taup_eff
        print('pinf_over_taup_eff')
        popt, _ = curve_fit(expgrowth, lookup['Q'], lookup['pinf_over_taup_eff'])
        pinf_over_taup_eff_fit = expgrowth(lookup['Q'], *popt)
        r2 = rsquared(lookup['pinf_over_taup_eff'], pinf_over_taup_eff_fit)
        fig, ax = plt.subplots(figsize=(21, 7))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
        ax.set_ylabel('$p_{\\infty} / \\tau_p\ (ms^{-1})$', fontsize=28)
        ax.plot(lookup['Q'] * 1e2, lookup['pinf_over_taup_eff'] * 1e-3, linewidth=2, label='data')
        ax.plot(lookup['Q'] * 1e2, pinf_over_taup_eff_fit * 1e-3, linewidth=2, label='fit')
        ax.text(0.45, 0.9, '$R^2 = {:.5f}$'.format(r2), transform=ax.transAxes, fontsize=28)
        ax.legend()

        # inv_taup_eff
        print('inv_taup_eff')
        popt, _ = curve_fit(dualexp, lookup['Q'], lookup['inv_taup_eff'], p0=(-0.2, -0.04, 15, 15))
        inv_taup_eff_fit = dualexp(lookup['Q'], *popt)
        r2 = rsquared(lookup['inv_taup_eff'], inv_taup_eff_fit)
        fig, ax = plt.subplots(figsize=(21, 7))
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=28)
        ax.set_ylabel('$1 / \\tau_p\ (ms^{-1})$', fontsize=28)
        ax.plot(lookup['Q'] * 1e2, lookup['inv_taup_eff'] * 1e-3, linewidth=2, label='data')
        ax.plot(lookup['Q'] * 1e2, inv_taup_eff_fit * 1e-3, linewidth=2, label='fit')
        ax.text(0.45, 0.9, '$R^2 = {:.5f}$'.format(r2), transform=ax.transAxes, fontsize=28)
        ax.legend()

        plt.show()
