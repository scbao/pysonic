#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-28 14:18:21

""" Run batch acoustic simulations of specific "point-neuron" models. """

import os
import logging
import numpy as np
from PointNICE.solvers import checkBatchLog, runAStimBatch
from PointNICE.channels import *
from PointNICE.utils import load_BLS_params
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

# Channels mechanisms
neurons = [CorticalLTS()]

# Stimulation parameters
stim_params = {
    'freqs': [3.5e5],  # Hz
    'amps': [150e3],  # Pa
    'durations': [50e-3],  # s
    'PRFs': [1e2],  # Hz
    'DFs': [0.5]
}
stim_params['offsets'] = [30e-3] * len(stim_params['durations'])  # s

# Select output directory
try:
    (batch_dir, log_filepath) = checkBatchLog('A-STIM')
except AssertionError as err:
    logger.error(err)
    quit()

# Run A-STIM batch
pkl_filepaths = runAStimBatch(batch_dir, log_filepath, neurons, bls_params, geom, stim_params)
pkl_dir, _ = os.path.split(pkl_filepaths[0])

# Plot resulting profiles
plotBatch(pkl_dir, pkl_filepaths)
