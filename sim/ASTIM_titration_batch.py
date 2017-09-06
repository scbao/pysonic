#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-06 13:34:19

""" Run batch acoustic titrations of specific "point-neuron" models. """

import os
import logging
import numpy as np
from PointNICE.solvers import setBatchDir, checkBatchLog, titrateAStimBatch
from PointNICE.channels import *
from PointNICE.utils import load_BLS_params
from PointNICE.plt import plotBatch

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# BLS parameters
bls_params = load_BLS_params()

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
    # 'DFs': [1.0]
}

# Select output directory
try:
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'A-STIM')
except AssertionError as err:
    logger.error(err)
    quit()

# Run titration batch
pkl_filepaths = titrateAStimBatch(batch_dir, log_filepath, neurons, bls_params, geom, stim_params)
pkl_dir, _ = os.path.split(pkl_filepaths[0])

# Plot resulting profiles
plotBatch(pkl_dir, pkl_filepaths, {'Q_m': ['Qm']})
