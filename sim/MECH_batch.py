#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-21 10:46:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-07 11:03:12

""" Run batch simulations of the NICE mechanical model with imposed charge densities """

import sys
import os
import logging
import numpy as np

from PointNICE.utils import logger
from PointNICE.solvers import setBatchDir, checkBatchLog, runMechBatch
from PointNICE.plt import plotBatch

# Set logging level
logger.setLevel(logging.DEBUG)

# Electrical properties of the membrane
Cm0 = 1e-2  # membrane resting capacitance (F/m2)
Qm0 = -80e-5  # membrane resting charge density (C/m2)
a = 500e-9  # in-plane diameter (m)

# Stimulation parameters
stim_params = {
    'freqs': [3.5e5],  # Hz
    'amps': [500e3],  # Pa
    'charges': [-80e-5]  # C/m2
}

# Select output directory
try:
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'MECH')

    # Run MECH batch
    pkl_filepaths = runMechBatch(batch_dir, log_filepath, Cm0, Qm0, stim_params, a)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    plotBatch(pkl_dir, pkl_filepaths)

except AssertionError as err:
    logger.error(err)
    sys.exit(1)
