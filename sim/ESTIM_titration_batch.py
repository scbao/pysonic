# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-25 14:50:39
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-06 13:37:34

""" Run batch electrical titrations of specific "point-neuron" models. """

import os
import logging
import numpy as np
from PointNICE.solvers import setBatchDir, checkBatchLog, titrateEStimBatch
from PointNICE.channels import *
from PointNICE.plt import plotBatch

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# Channels mechanisms
neurons = [CorticalRS()]

# Stimulation parameters
stim_params = {
    'amps': [20.0],  # mA/m2
    'durations': [0.5],  # s
    'PRFs': [1e2],  # Hz
    # 'DFs': [1.0]
}

# Select output directory
try:
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'E-STIM')
except AssertionError as err:
    logger.error(err)
    quit()

# Run titration batch
pkl_filepaths = titrateEStimBatch(batch_dir, log_filepath, neurons, stim_params)
pkl_dir, _ = os.path.split(pkl_filepaths[0])

# Plot resulting profiles
plotBatch(pkl_dir, pkl_filepaths, {'V_m': ['Vm']})
