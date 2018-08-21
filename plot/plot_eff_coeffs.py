#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-15 15:59:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-21 16:07:31

import numpy as np
import matplotlib.pyplot as plt

from PySONIC.plt import plotEffVars
from PySONIC.neurons import *

''' Plot the profiles of effective variables as a function of charge density
    with amplitude color code.
'''

# Set parameters
neuron = CorticalRS()
Fdrive = 500e3
amps = np.logspace(np.log10(1), np.log10(600), 10) * 1e3
charges = np.linspace(neuron.Vm0, 50, 100) * 1e-5

# Define variables to plot
gates = ['m', 'h', 'n', 'p']
keys = ['V', 'ng']
for x in gates:
    keys += ['alpha{}'.format(x), 'beta{}'.format(x)]

# Plot effective variables
fig = plotEffVars(neuron, Fdrive, amps=amps, charges=charges, keys=keys, ncolmax=2, fs=8)

plt.show()
