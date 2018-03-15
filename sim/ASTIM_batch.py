#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-15 16:55:31

""" Run batch acoustic simulations of specific "point-neuron" models. """

import sys
import os
import logging
import numpy as np

from PointNICE.utils import logger, InputError
from PointNICE.solvers import setBatchDir, checkBatchLog, runAStimBatch
from PointNICE.plt import plotBatch

# Set logging level
logger.setLevel(logging.INFO)

# Neurons
neurons = ['RS']

# Stimulation parameters
stim_params = {
    'freqs': [1000e3],  # Hz
    'amps': np.array([10, 20, 40, 80, 150, 300, 600]) * 1e3,  # Pa
    'durations': np.array([20, 40, 60, 80, 100, 150, 200, 250, 300]) * 1e-3,  # s
    'PRFs': np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]) * 1e3,  # Hz
    'DCs': np.array([1, 2, 5, 10, 25, 50, 75, 100]) * 1e-2
}
stim_params['offsets'] = 350e-3 - stim_params['durations']  # s


try:
    # Select output directory
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'A-STIM')

    # Run A-STIM batch
    pkl_filepaths = runAStimBatch(batch_dir, log_filepath, neurons, stim_params,
                                  int_method='effective')
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    # yvars = {'Q_m': ['Qm']}
    # plotBatch(pkl_dir, pkl_filepaths, yvars)

except InputError as err:
    logger.error(err)
    sys.exit(1)
