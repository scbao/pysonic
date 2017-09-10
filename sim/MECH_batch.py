#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-21 10:46:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-10 17:51:56

""" Run batch simulations of the NICE mechanical model with imposed charge densities """

import sys
import os
import logging
import numpy as np

from PointNICE.solvers import setBatchDir, checkBatchLog, runMechBatch
from PointNICE.plt import plotBatch

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# Electrical properties of the membrane
Cm0 = 1e-2  # membrane resting capacitance (F/m2)
Qm0 = -80e-5  # membrane resting charge density (C/m2)

# Stimulation parameters
stim_params = {
    'freqs': [3.5e5],  # Hz
    'amps': [100e3],  # Pa
    'charges': [50e-5]  # C/m2
}

# Select output directory
try:
    batch_dir = setBatchDir()
    log_filepath, _ = checkBatchLog(batch_dir, 'MECH')

    # Run MECH batch
    pkl_filepaths = runMechBatch(batch_dir, log_filepath, Cm0, Qm0, stim_params)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    plotBatch(pkl_dir, pkl_filepaths)

except AssertionError as err:
    logger.error(err)
    sys.exit(1)
