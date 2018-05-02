#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-05-02 21:18:18

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
    'freqs': [500e3],  # Hz
    'amps': np.logspace(np.log10(10), np.log10(600), num=30) * 1e3,  # Pa
    'durations': [1],  # s
    'PRFs': [100.0],  # Hz
    'DCs': (np.arange(100) + 1) / 1e2,
    'offsets': [0]
}
# stim_params['offsets'] = 350e-3 - stim_params['durations']  # s


try:
    # Select output directory
    # batch_dir = setBatchDir()
    batch_dir = '../../data/activation maps/RS 500kHz PRF100Hz 1s'
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
