#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-21 10:46:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-06-13 19:21:36

""" Run batch simulations of the NICE mechanical model with imposed charge densities """

import sys
import os
import logging
import numpy as np

from PointNICE.utils import logger, InputError
from PointNICE.solvers import setBatchDir, checkBatchLog, runMechBatch
from PointNICE.neurons import *
from PointNICE.plt import plotBatch

# Set logging level
logger.setLevel(logging.DEBUG)

a = 32e-9  # in-plane diameter (m)

# Electrical properties of the membrane
neuron = CorticalRS()
Cm0 = neuron.Cm0
Qm0 = neuron.Vm0 * 1e-5
# Cm0 = 1e-2  # membrane resting capacitance (F/m2)
# Qm0 = -80e-5  # membrane resting charge density (C/m2)

# Stimulation parameters
stim_params = {
    'freqs': [500e3],  # Hz
    'amps': [50e3],  # Pa
    'charges': np.linspace(-72, 0, 5) * 1e-5  # C/m2
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

except InputError as err:
    logger.error(err)
    sys.exit(1)