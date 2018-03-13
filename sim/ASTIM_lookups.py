#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-02 17:50:10
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-13 17:13:16

""" Create lookup tables for different acoustic frequencies. """

import logging
import numpy as np

import PointNICE
from PointNICE.utils import logger
from PointNICE.neurons import *

# Set logging level
logger.setLevel(logging.DEBUG)

# BLS diameter (m)
a = 32e-9

# Channel mechanisms
neurons = [CorticalRS()]

# Stimulation parameters
freqs = np.array([20., 100., 500., 1000., 2000., 3000., 4000.]) * 1e3  # Hz
amps = np.logspace(np.log10(0.1), np.log10(600), num=50) * 1e3  # Pa
amps = np.insert(amps, 0, 0.0)  # adding amplitude 0

logger.info('Starting batch lookup creation')

for neuron in neurons:
    # Create a SolverUS instance (with dummy frequency parameter)
    solver = PointNICE.SolverUS(a, neuron, 0.0)

    # Create lookup file
    try:
        solver.createLookup(neuron, freqs, amps)
        logger.info('%s Lookup table successfully created', neuron.name)
    except AssertionError as err:
        logger.error(err)
