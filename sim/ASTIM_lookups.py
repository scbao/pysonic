#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-02 17:50:10
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-30 15:26:04

""" Create lookup tables for different acoustic frequencies. """

import logging
import numpy as np

import PointNICE
from PointNICE.utils import load_BLS_params
from PointNICE.channels import *

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# BLS parameters
params = load_BLS_params()

# Geometry of NBLS structure
a = 32e-9  # in-plane radius (m)
d = 0.0e-6  # embedding tissue thickness (m)
geom = {"a": a, "d": d}

# Channel mechanisms
neurons = [CorticalRS()]

# Stimulation parameters
freqs = np.arange(100, 1001, 100) * 1e3  # Hz
amps = np.logspace(np.log10(0.1), np.log10(600), num=50) * 1e3  # Pa
amps = np.insert(amps, 0, 0.0)  # adding amplitude 0

logger.info('Starting batch lookup creation')

for ch_mech in neurons:
    # Create a SolverUS instance (with dummy frequency parameter)
    solver = PointNICE.SolverUS(geom, params, ch_mech, 0.0)
    charges = np.arange(np.round(ch_mech.Vm0 - 10.0), 50.0 + 1.0, 1.0) * 1e-5  # C/m2

    # Create lookup file
    solver.createLookup(ch_mech, freqs, amps, charges)

logger.info('Lookup tables successfully created')
