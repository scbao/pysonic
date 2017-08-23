#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-04-24 11:04:39
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-05-26 18:34:14

''' Predict a 1D Vmeff profile using Gaussian Process Regression. '''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


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
varinf = Variable('V_{m, eff}', 'mV', 'V_eff', 1., 1e-2)
# varinf = Variable('\\alpha_{m, eff}', 'ms^{-1}', 'alpha_m_eff', 1e-3, 1e1)
# varinf = Variable('\\beta_{m, eff}', 'ms^{-1}', 'beta_m_eff', 1e-3, 5e0)
# varinf = Variable('\\alpha_{h, eff}', 'ms^{-1}', 'alpha_h_eff', 1e-3, 1e1)
# varinf = Variable('\\beta_{h, eff}', 'ms^{-1}', 'beta_h_eff', 1e-3, 1e1)


# Define true function by interpolation from specific profile
def f(x):
    return np.interp(x, Qm, xvect)


# Load coefficient profile
dirpath = 'C:/Users/admin/Google Drive/PhD/NBLS model/Output/lookups 0.35MHz charge extended/'
filepath = dirpath + 'lookups_a32.0nm_f350.0kHz_A100.0kPa_dQ1.0nC_cm2.pkl'
filepath0 = dirpath + 'lookups_a32.0nm_f350.0kHz_A0.0kPa_dQ1.0nC_cm2.pkl'
with open(filepath, 'rb') as fh:
    lookup = pickle.load(fh)
    Qm = lookup['Q']
    xvect = lookup[varinf.lookup]
with open(filepath0, 'rb') as fh:
    lookup = pickle.load(fh)
    xvect0 = lookup[varinf.lookup]

# xvect = xvect - xvect0

# Define algorithmic parameters
n_iter_min = 10
n_iter_max = 20
max_pred_errors = []
max_errors = []
delta_factor = 10

# Define prediction vector
x = np.atleast_2d(np.linspace(-150., 150., 1000) * 1e-5).T
y = f(x).ravel()

# Define initial samples and compute function at these points
X0 = np.atleast_2d(np.linspace(-150., 150., 10) * 1e-5).T
Y0 = f(X0).ravel()

# Instantiate a Gaussian Process model
print('Creating Gaussian Process with RBF Kernel')
kernel = C(100.0, (1.0, 500.0)) * RBF(1e-4, (1e-5, 1e-3))  # + C(100.0, (1.0, 500.0)) * RBF(1e-5, (1e-5, 1e-3))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=8, normalize_y=True)

# Fit to data using Maximum Likelihood Estimation of the parameters
print('Initial fitting')
gp.fit(X0, Y0)

# Make the prediction on the meshed x-axis (ask for MSE as well)
print('Predicting over linear input range')
ypred0, ypred0_std = gp.predict(x, return_std=True)
max_err = np.amax(np.abs(y - ypred0))
max_errors.append(max_err)
max_pred_error = np.amax(ypred0_std)
max_pred_errors.append(max_pred_error)
print('Initialization: Kernel =', gp.kernel_,
      ', Max err = {:.2f} {}, Max pred. err = {:.2f} {}'.format(
          max_err * varinf.factor, varinf.unit, max_pred_error * varinf.factor, varinf.unit))


# Initial observation and prediction
yminus0 = ypred0 - delta_factor * ypred0_std
yplus0 = ypred0 + delta_factor * ypred0_std
fig, ax = plt.subplots()
ax.plot(x * 1e5, y * varinf.factor, 'r:', label=u'True Function')
ax.plot(X0 * 1e5, Y0 * varinf.factor, 'r.', markersize=10, label=u'Initial Observations')
ax.plot(x * 1e5, ypred0 * varinf.factor, 'b-', label=u'Initial Prediction')
ax.fill(np.concatenate([x, x[::-1]]) * 1e5,
        np.concatenate([yminus0, yplus0[::-1]]) * varinf.factor,
        alpha=.5, fc='b', ec='None', label='$\\pm\ {:.0f} \\sigma$'.format(delta_factor))
ax.set_xlabel('$Q_m\ (nC/cm^2)$')
ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$')
ax.legend()
ax.set_title('Initial observation and prediction')

print('Optimizing prediction by adding samples iteratively')

X = X0
Y = Y0
ypred = ypred0
ypred_std = ypred0_std

n_iter = 0
while (max_pred_error > varinf.max_error and n_iter < n_iter_max) or n_iter < n_iter_min:
    newX = x[np.argmax(ypred_std)]
    newY = f(newX)
    X = np.atleast_2d(np.insert(X.ravel(), -1, newX)).T
    Y = np.insert(Y, -1, newY)
    gp.fit(X, Y)
    ypred, ypred_std = gp.predict(x, return_std=True)
    max_err = np.amax(np.abs(y - ypred))
    max_errors.append(max_err)
    max_pred_error = np.amax(ypred_std)
    max_pred_errors.append(max_pred_error)
    print('Step {}:'.format(n_iter + 1), ' Kernel =', gp.kernel_,
          ', Max err = {:.2f} {}, Max pred. err = {:.2f} {}'.format(
              max_err * varinf.factor, varinf.unit, max_pred_error * varinf.factor, varinf.unit))
    if (n_iter + 1) % 5 == 0:
        yminus = ypred - delta_factor * ypred_std
        yplus = ypred + delta_factor * ypred_std
        fig, ax = plt.subplots()
        ax.plot(x * 1e5, y * varinf.factor, 'r:', label=u'True Function')
        ax.plot(X * 1e5, Y * varinf.factor, 'r.', markersize=10, label=u'Final Observations')
        ax.plot(x * 1e5, ypred * varinf.factor, 'b-', label=u'Final Prediction')
        ax.fill(np.concatenate([x, x[::-1]]) * 1e5,
                np.concatenate([yminus, yplus[::-1]]) * varinf.factor,
                alpha=.5, fc='b', ec='None', label='$\\pm\ {:.0f} \\sigma$'.format(delta_factor))
        ax.set_xlabel('$Q_m\ (nC/cm^2)$')
        ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$')
        ax.legend()
        ax.set_title('After {} steps'.format(n_iter + 1))
    n_iter += 1


# Final observation and prediction
yminus = ypred - delta_factor * ypred_std
yplus = ypred + delta_factor * ypred_std
fig, ax = plt.subplots()
ax.plot(x * 1e5, y * varinf.factor, 'r:', label=u'True Function')
ax.plot(X * 1e5, Y * varinf.factor, 'r.', markersize=10, label=u'Final Observations')
ax.plot(x * 1e5, ypred * varinf.factor, 'b-', label=u'Final Prediction')
ax.fill(np.concatenate([x, x[::-1]]) * 1e5,
        np.concatenate([yminus, yplus[::-1]]) * varinf.factor,
        alpha=.5, fc='b', ec='None', label='$\\pm\ {:.0f} \\sigma$'.format(delta_factor))
ax.set_xlabel('$Q_m\ (nC/cm^2)$')
ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$')
ax.legend()
ax.set_title('Final observation and prediction')

# Evolution of max. absolute error
fig, ax = plt.subplots()
ax.plot(np.linspace(0, n_iter, n_iter + 1), max_errors)
ax.set_xlabel('# iterations')
ax.set_ylabel('Max. error ($' + varinf.unit + ')$')

# Evolution of max. predicted error
fig, ax = plt.subplots()
ax.plot(np.linspace(0, n_iter, n_iter + 1), max_pred_errors)
ax.set_xlabel('# iterations')
ax.set_ylabel('Max. predicted error ($' + varinf.unit + ')$')

plt.show()
