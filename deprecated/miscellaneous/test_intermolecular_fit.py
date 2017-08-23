#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-07 16:04:34
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-07-08 17:04:26

""" Test the Lennard-Jones fitting of the average intermolecular pressure """

import time
import timeit
import numpy as np
from scipy.optimize import brentq, curve_fit
import matplotlib.pyplot as plt
import logging

import PyNICE
from PyNICE.utils import LoadParams, LJfit, rsquared, rmse

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PyNICE')
logger.setLevel(logging.DEBUG)


def Pmpred_loop(x, a, b, c, d, e):
    nx = len(x)
    y = np.empty(nx)
    for i in range(nx):
        y[i] = LJfit(x[i], a, b, c, d, e)
    return y


def Pminterp_loop(xtest, xtrain, ytrain):
    nx = len(xtest)
    ytest = np.empty(nx)
    for i in range(nx):
        ytest[i] = np.interp(xtest[i], xtrain, ytrain)
    return ytest


# Initialization: create a NBLS instance
params = LoadParams()

a = 32e-9
d = 0.0e-6  # embedding tissue thickness (m)
Fdrive = 0.0  # dummy stimulation frequency
Cm0 = 1e-2  # F/m2
Qm0 = -80e-5  # C/m2
bls = PyNICE.BilayerSonophore({"a": a, "d": d}, params, Fdrive, Cm0, Qm0)

quit()

# Determine deflection range
Pmmax = 100e6  # Pa
ZMinlb = -0.49 * bls.Delta
ZMinub = 0.0
f = lambda Z, Pmmax: bls.PMavg(Z, bls.curvrad(Z), bls.surface(Z)) - Pmmax
Zmin = brentq(f, ZMinlb, ZMinub, args=(Pmmax), xtol=1e-16)
print('Zmin fit = %f nm (%.2f Delta)' % (Zmin * 1e9, Zmin / bls.Delta))

# Create vectors for geometric variables
print('Generating training samples')
Zmax = 2 * bls.a
Z_train = np.hstack((np.arange(Zmin, bls.a / 3, 1e-11), np.arange(bls.a / 3, Zmax, 5e-10)))
Pmavg_train = np.array([bls.PMavg(ZZ, bls.curvrad(ZZ), bls.surface(ZZ)) for ZZ in Z_train])
print('')

# Compute optimal nonlinear fit of custom LJ function
print('Fitting LJ function parameters to training data')
x0_guess = 2e-9
C_guess = 1e4
nrep_guess = 5.0
nattr_guess = 3.0
pguess = (x0_guess, C_guess, nrep_guess, nattr_guess)
popt, _ = curve_fit(lambda x, x0, C, nrep, nattr:
                    LJfit(x, bls.Delta, x0, C, nrep, nattr), Z_train, Pmavg_train,
                    p0=pguess, maxfev=10000)
(x0_opt, C_opt, nrep_opt, nattr_opt) = popt
print('')

# Compute intermolecular pressure vector
print('generating testing samples')
Z_test = np.linspace(Zmin, Zmax, 10000)
t0 = time.time()
Pmavg_test = np.array([bls.PMavg(ZZ, bls.curvrad(ZZ), bls.surface(ZZ)) for ZZ in Z_test])
tdirect = time.time() - t0
print('direct time: {} ms'.format(tdirect * 1e3))
print('')

nrep = 100

print('evaluating model 1 (LJ fit) on testing set')
t0 = time.time()
Pmavg_fit = Pmpred_loop(Z_test, bls.Delta, x0_opt, C_opt, nrep_opt, nattr_opt)
# Pmavg_fit = LJfit(Z_test, bls.Delta, x0_opt, C_opt, nrep_opt, nattr_opt)
tpred = time.time() - t0
print('pred time: {} ms'.format(tpred * 1e3))
tpred = timeit.timeit(lambda: Pmpred_loop(Z_test, bls.Delta, x0_opt, C_opt, nrep_opt, nattr_opt),
                      number=nrep)
print('pred time: {} ms'.format(tpred * 1e3))
r2 = rsquared(Pmavg_test, Pmavg_fit)
residuals = Pmavg_test - Pmavg_fit
ss_res = np.sum(residuals**2)
N = residuals.size
std_err = np.sqrt(ss_res / N)
print("R-squared opt =  " + '{:.10}'.format(r2))
print("standard error: sigma_err = " + str(std_err))
print('')

print('evaluating model 2 (interpolation) on testing set')
t0 = time.time()
Pmavg_interp = Pminterp_loop(Z_test, Z_train, Pmavg_train)
# Pmavg_interp = np.interp(Z_test, Z_train, Pmavg_train)
tinterp = time.time() - t0
print('interp time: {} ms'.format(tinterp * 1e3))
tinterp = timeit.timeit(lambda: Pminterp_loop(Z_test, Z_train, Pmavg_train), number=nrep)
print('interp time: {} ms'.format(tinterp * 1e3))
r2 = rsquared(Pmavg_test, Pmavg_interp)
residuals = Pmavg_test - Pmavg_interp
ss_res = np.sum(residuals**2)
N = residuals.size
std_err = np.sqrt(ss_res / N)
print("R-squared opt =  " + '{:.10}'.format(r2))
print("standard error: sigma_err = " + str(std_err))
print('')


# Plot Pm data and predictions
print('plotting')
fig, ax = plt.subplots()
fig.canvas.set_window_title('a = ' + '{:.2f}'.format(a * 1e9) + ' nm')
ax.plot(Z_test * 1e9, Pmavg_test * 1e-6, label="direct")
ax.plot(Z_test * 1e9, Pmavg_fit * 1e-6, '-', linewidth=2, label='fit')
ax.plot(Z_test * 1e9, Pmavg_interp * 1e-6, '-', linewidth=2, label='interpolated')
ax.plot(Z_test * 1e9, np.abs(Pmavg_test - Pmavg_fit) * 1e-6, '-', linewidth=2, label='fit error')
ax.plot(Z_test * 1e9, np.abs(Pmavg_test - Pmavg_interp) * 1e-6, '-', linewidth=2, label='interp error')
ax.set_xlabel('deflection (nm)')
ax.set_ylabel('pressure (MPa)')
ax.grid(True)
ax.legend()

plt.show()
