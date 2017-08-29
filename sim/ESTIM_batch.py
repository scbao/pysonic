# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-29 15:09:38

""" Run batch electrical simulations of specific "point-neuron" models. """

import os
import logging
from PointNICE.solvers import checkBatchLog, runEStimBatch
from PointNICE.channels import *
from PointNICE.plt import plotBatch

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# Channels mechanisms
# neurons = [LeechTouch()]
neurons = [ThalamicRE()]

# Stimulation parameters
stim_params = {
    'amps': [3.1],  # mA/m2
    'durations': [0.5],  # s
    'PRFs': [1e2],  # Hz
    'DFs': [1.]
}
stim_params['offsets'] = [1.0] * len(stim_params['durations'])  # s

# Select output directory
try:
    (batch_dir, log_filepath) = checkBatchLog('E-STIM')
except AssertionError as err:
    logger.error(err)
    quit()

# Run E-STIM batch
pkl_filepaths = runEStimBatch(batch_dir, log_filepath, neurons, stim_params)
pkl_dir, _ = os.path.split(pkl_filepaths[0])

# Plot resulting profiles
plotBatch(pkl_dir, pkl_filepaths)
