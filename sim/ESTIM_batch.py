# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-28 16:23:35

""" Run batch electrical simulations of specific "point-neuron" models. """

import sys
import os
import logging
from PointNICE.utils import logger
from PointNICE.solvers import setBatchDir, checkBatchLog, runEStimBatch
from PointNICE.plt import plotBatch

# Set logging level
logger.setLevel(logging.INFO)

# Neurons
neurons = ['LeechT']

# Stimulation parameters
stim_params = {
    'amps': [-40.],  # mA/m2
    'durations': [0.5],  # s
    'PRFs': [1e2],  # Hz
    'DFs': [1.0]
}
stim_params['offsets'] = [1.0] * len(stim_params['durations'])  # s

try:
    # Select output directory
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'E-STIM')

    # Run E-STIM batch
    pkl_filepaths = runEStimBatch(batch_dir, log_filepath, neurons, stim_params)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    plotBatch(pkl_dir, pkl_filepaths)

except AssertionError as err:
    logger.error(err)
    sys.exit(1)
