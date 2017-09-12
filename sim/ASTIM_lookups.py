#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-02 17:50:10
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-10 18:42:14

""" Create lookup tables for different acoustic frequencies. """

import logging
import numpy as np

import PointNICE
from PointNICE.utils import logger
from PointNICE.channels import *

# Set logging level
logger.setLevel(logging.DEBUG)

# BLS diameter (m)
a = 32e-9  

# Channel mechanisms
neurons = [CorticalRS()]

# Stimulation parameters
freqs = np.arange(100, 1001, 100) * 1e3  # Hz
amps = np.logspace(np.log10(0.1), np.log10(600), num=50) * 1e3  # Pa
amps = np.insert(amps, 0, 0.0)  # adding amplitude 0

logger.info('Starting batch lookup creation')

for ch_mech in neurons:
    # Create a SolverUS instance (with dummy frequency parameter)
    solver = PointNICE.SolverUS(a, ch_mech, 0.0)
    charges = np.arange(np.round(ch_mech.Vm0 - 10.0), 50.0 + 1.0, 1.0) * 1e-5  # C/m2

    # Create lookup file
    solver.createLookup(ch_mech, freqs, amps, charges)

logger.info('Lookup tables successfully created')
