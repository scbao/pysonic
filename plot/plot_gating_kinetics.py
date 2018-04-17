#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-11 20:35:38
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-04-17 15:42:07

""" Plot the voltage-dependent steady-states and time constants of activation and inactivation
    gates of the different ionic currents involved in the neuron's membrane. """

from PointNICE.plt import plotGatingKinetics
from PointNICE.neurons import *

# Instantiate neuron(s)
neurons = [CorticalIB()]

# Plot gating kinetics for each neuron(s)
for neuron in neurons:
    plotGatingKinetics(neuron)
