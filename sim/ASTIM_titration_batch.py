#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-22 17:15:29

""" Run batch parameter titrations of the NICE model. """

import logging
import numpy as np
from PointNICE.solvers import runTitrationBatch
from PointNICE.channels import *
from PointNICE.utils import LoadParams, CheckBatchLog

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# BLS parameters
bls_params = LoadParams()

# Geometry of NBLS structure
a = 32e-9  # in-plane radius (m)
d = 0.0e-6  # embedding tissue thickness (m)
geom = {"a": a, "d": d}

# Channels mechanisms
neurons = [CorticalRS()]

# Stimulation parameters
stim_params = {
    'freqs': [3.5e5],  # Hz
    # 'amps': [100e3],  # Pa
    'durations': [50e-3],  # s
    'PRFs': [1e2],  # Hz
    'DFs': [1.0]
}

# Select output directory
try:
    (batch_dir, log_filepath) = CheckBatchLog('elec')
except AssertionError as err:
    logger.error(err)
    quit()

# Run titration batch
runTitrationBatch(batch_dir, log_filepath, neurons, bls_params, geom, stim_params)
