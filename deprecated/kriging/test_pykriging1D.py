#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-04-24 11:04:39
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-05-26 13:44:14

''' Predict a 1D Vmeff profile using the PyKriging module. '''

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pyKriging.krige import kriging


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


print('defining estimation vector')
x = np.atleast_2d(np.linspace(-150., 150., 1000) * 1e-5).T
y = f(x).ravel()

print('defining prediction vector')
X0 = np.atleast_2d(np.linspace(-150., 150., 10) * 1e-5).T
Y0 = f(X0).ravel()

print('creating kriging model')
k = kriging(X0, Y0)

print('training kriging model')
k.train()

print('predicting')
y_pred0 = np.array([k.predict(xx) for xx in x])

X = X0
Y = Y0

numiter = 10
for i in range(numiter):
    print('Infill iteration {0} of {1}....'.format(i + 1, numiter))
    newpoints = k.infill(1, method='error')
    for point in newpoints:
        newX = k.inversenormX(point)
        newY = f(newX)[0]
        print('adding point ({:.3f}, {:.3f})'.format(newX[0] * 1e5, newY * varinf.factor))
        X = np.append(X, [newX], axis=0)
        Y = np.append(Y, newY)
        k.addPoint(newX, newY, norm=True)
    k.train()

y_pred = np.array([k.predict(xx) for xx in x])

fig, ax = plt.subplots()
ax.plot(x * 1e5, y * varinf.factor, 'r:', label=u'true function')
ax.plot(X0 * 1e5, Y0 * varinf.factor, 'r.', markersize=10, label=u'Initial observations')
ax.plot(x * 1e5, y_pred0 * varinf.factor, 'b-', label=u'Initial prediction')
ax.set_xlabel('$Q_m\ (nC/cm^2)$')
ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$')
ax.legend()


fig, ax = plt.subplots()
ax.plot(x * 1e5, y * varinf.factor, 'r:', label=u'true function')
ax.plot(X * 1e5, Y * varinf.factor, 'r.', markersize=10, label=u'Final observations')
ax.plot(x * 1e5, y_pred * varinf.factor, 'b-', label=u'Final prediction')
ax.set_xlabel('$Q_m\ (nC/cm^2)$')
ax.set_ylabel('$' + varinf.name + '\ (' + varinf.unit + ')$')
ax.legend()


plt.show()

