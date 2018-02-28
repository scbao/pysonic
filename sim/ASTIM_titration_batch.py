#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-01-26 10:46:35

""" Run batch acoustic titrations of specific "point-neuron" models. """

import sys
import os
import logging
import numpy as np
from PointNICE.utils import logger
from PointNICE.solvers import setBatchDir, checkBatchLog, titrateAStimBatch
from PointNICE.plt import plotBatch

# Set logging level
logger.setLevel(logging.DEBUG)

# Channels mechanisms
neurons = ['RS']

# Stimulation parameters
stim_params = {
    'freqs': [5e5],  # Hz
    # 'amps': [100e3],  # Pa
    'durations': [100e-3],  # s
    'PRFs': [1e2],  # Hz
    'DFs': [1.0, 0.05]
}

try:
    # Select output directory
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'A-STIM')

    # Run titration batch
    pkl_filepaths = titrateAStimBatch(batch_dir, log_filepath, neurons, stim_params)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    plotBatch(pkl_dir, pkl_filepaths, {'Q_m': ['Qm']})

except AssertionError as err:
    logger.error(err)
    sys.exit(1)
