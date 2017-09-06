#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-21 10:46:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-06 13:40:19

""" Run batch simulations of the NICE mechanical model with imposed charge densities """

import os
import logging
import numpy as np

from PointNICE.utils import load_BLS_params
from PointNICE.solvers import setBatchDir, checkBatchLog, runMechBatch
from PointNICE.plt import plotBatch

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# BLS parameters
bls_params = load_BLS_params()

# Geometry of BLS structure
a = 32e-9  # in-plane radius (m)
d = 0.0e-6  # embedding tissue thickness (m)
geom = {"a": a, "d": d}

# Electrical properties of the membrane
Cm0 = 1e-2  # membrane resting capacitance (F/m2)
Qm0 = -80e-5  # membrane resting charge density (C/m2)

# Stimulation parameters
stim_params = {
    'freqs': [3.5e5],  # Hz
    'amps': [100e3],  # Pa
    'charges': [50e-5]  # C/m2
}

# Select output directory
try:
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'MECH')
except AssertionError as err:
    logger.error(err)
    quit()

# Run MECH batch
pkl_filepaths = runMechBatch(batch_dir, log_filepath, bls_params, geom, Cm0, Qm0, stim_params)
pkl_dir, _ = os.path.split(pkl_filepaths[0])

# Plot resulting profiles
plotBatch(pkl_dir, pkl_filepaths)
