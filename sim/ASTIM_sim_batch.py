#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-22 18:39:57

""" Run batch acoustic simulations of the NICE model. """

import logging
import numpy as np
from PointNICE.solvers import runSimBatch
from PointNICE.channels import *
from PointNICE.utils import LoadParams, CheckBatchLog

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# BLS parameters
bls_params = LoadParams()

# Geometry of NBLS structure
a = 32e-9  # in-plane radius (m)
d = 0.0e-6  # embedding tissue thickness (m)
geom = {"a": a, "d": d}

# Channels mechanisms
neurons = [CorticalRS()]

# Stimulation parameters
stim_params = {
    'freqs': [3.5e5],  # Hz
    'amps': [100e3],  # Pa
    'durations': [50e-3],  # s
    'PRFs': [1e2],  # Hz
    'DFs': [1.0]
}
stim_params['offsets'] = [30e-3] * len(stim_params['durations'])  # s

# stim_params = {
#     'freqs': np.array([200, 400, 600, 800, 1000]) * 1e3,  # Hz
#     'amps': np.array([10, 20, 40, 80, 150, 300, 600]) * 1e3,  # Pa
#     'durs': np.array([20, 40, 60, 80, 100, 150, 200, 250, 300]) * 1e-3,  # s
#     'PRFs': np.array([0.1, 0.2, 0.5, 1, 2, 5, 10]) * 1e3,  # Hz
#     'DFs': np.array([1, 2, 5, 10, 25, 50, 75, 100]) / 100
# }
# stim_params['offsets'] = 350e-3 - stim_params['durations']  # s

# Simulation type
sim_type = 'effective'

# Select output directory
try:
    (batch_dir, log_filepath) = CheckBatchLog('elec')
except AssertionError as err:
    logger.error(err)
    quit()

# Run simulation batch
runSimBatch(batch_dir, log_filepath, neurons, bls_params, geom, stim_params, sim_type)
