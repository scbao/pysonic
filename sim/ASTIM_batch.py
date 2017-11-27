#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-11-24 15:08:45

""" Run batch acoustic simulations of specific "point-neuron" models. """

import sys
import os
import logging
import numpy as np

from PointNICE.utils import logger
from PointNICE.solvers import setBatchDir, checkBatchLog, runAStimBatch
from PointNICE.plt import plotBatch

# Set logging level
logger.setLevel(logging.DEBUG)

# Neurons
neurons = ['LeechT']

# Stimulation parameters
stim_params = {
    'freqs': [350e3],  # Hz
    'amps': [100e3],  # Pa
    'durations': [150e-3],  # s
    'PRFs': [100.0],  # Hz
    'DFs': [1.0]
}
stim_params['offsets'] = [100e-3] * len(stim_params['durations'])  # s

try:
    # Select output directory
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'A-STIM')

    # Run A-STIM batch
    pkl_filepaths = runAStimBatch(batch_dir, log_filepath, neurons, stim_params,
                                  int_method='classic')
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    yvars = {'Q_m': ['Qm']}
    plotBatch(pkl_dir, pkl_filepaths, yvars)

except AssertionError as err:
    logger.error(err)
    sys.exit(1)
