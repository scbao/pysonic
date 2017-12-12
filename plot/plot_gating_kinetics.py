#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-11 20:35:38
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-12-01 15:45:07

""" Plot the voltage-dependent steady-states and time constants of activation and inactivation
    gates of the different ionic currents involved in the neuron's membrane. """

from PointNICE.plt import plotGatingKinetics
from PointNICE.channels import *

# Instantiate neuron(s)
neurons = [LeechPressure()]

# Plot gating kinetics for each neuron(s)
for neuron in neurons:
    plotGatingKinetics(neuron)
