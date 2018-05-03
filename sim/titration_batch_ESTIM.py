# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-25 14:50:39
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-05-03 12:22:29

""" Run batch electrical titrations of specific "point-neuron" models. """

import sys
import os
import logging
import numpy as np

from PointNICE.utils import logger, InputError
from PointNICE.solvers import setBatchDir, checkBatchLog, titrateEStimBatch
from PointNICE.plt import plotBatch

# Set logging level
logger.setLevel(logging.DEBUG)

# Neurons
neurons = ['IB']

# Stimulation parameters
stim_params = {
    # 'amps': [20.0],  # mA/m2
    'durations': [0.5],  # s
    'PRFs': [1e2],  # Hz
    'DCs': [1.0]
}

try:
    # Select output directory
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'E-STIM')

    # Run titration batch
    pkl_filepaths = titrateEStimBatch(batch_dir, log_filepath, neurons, stim_params)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    plotBatch(pkl_dir, pkl_filepaths, {'V_m': ['Vm']})

except InputError as err:
    logger.error(err)
    sys.exit(1)
