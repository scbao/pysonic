# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-12 20:17:41

""" Run batch electrical simulations of specific "point-neuron" models. """

import sys
import os
import logging
import numpy as np
from PointNICE.utils import logger
from PointNICE.solvers import setBatchDir, checkBatchLog, runEStimBatch
from PointNICE.plt import plotBatch

# Set logging level
logger.setLevel(logging.INFO)

# Neurons
neurons = ['LeechP']

# Stimulation parameters
stim_params = {
    'amps': [2.0],  # mA/m2
    'durations': np.array([20, 40, 60, 80, 100, 150, 200, 250, 300]) * 1e-3,  # s
    'PRFs': np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]) * 1e3,  # Hz
    'DCs': np.array([1, 2, 5, 10, 25, 50, 75, 100]) * 1e-2
}
stim_params['offsets'] = 350e-3 - stim_params['durations']  # s


try:
    # Select output directory
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'E-STIM')

    # Run E-STIM batch
    pkl_filepaths = runEStimBatch(batch_dir, log_filepath, neurons, stim_params)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    # plotBatch(pkl_dir, pkl_filepaths)

except AssertionError as err:
    logger.error(err)
    sys.exit(1)
