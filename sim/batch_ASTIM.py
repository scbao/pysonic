#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-25 02:10:49

""" Run batch acoustic simulations of specific "point-neuron" models. """

import sys
import os
import logging
import numpy as np
from argparse import ArgumentParser

from PySONIC.utils import logger, InputError
from PySONIC.solvers import setBatchDir, checkBatchLog, runAStimBatch
from PySONIC.plt import plotBatch

# Set logging level
logger.setLevel(logging.INFO)

# Neurons
neurons = ['RS']

# Stimulation parameters
stim_params = {
    'freqs': [500e3],  # Hz
    'amps': [100e3],  # Pa
    'durations': [100e-3],  # s
    'PRFs': [100.0],  # Hz
    'DCs': [0.1, 0.5, 1.],
    'offsets': [0]
}


if __name__ == '__main__':

    # Define argument parser
    ap = ArgumentParser()
    ap.add_argument('-m', '--multiprocessing', default=False, action='store_true',
                    help='Use multiprocessing')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-p', '--plot', default=False, action='store_true',
                    help='Plot results')

    # Parse arguments
    args = ap.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    try:
        # Select output directory
        batch_dir = setBatchDir()
        log_filepath, _ = checkBatchLog(batch_dir, 'A-STIM')

        # Run A-STIM batch
        pkl_filepaths = runAStimBatch(batch_dir, log_filepath, neurons, stim_params,
                                      int_method='sonic', multiprocess=args.multiprocessing)
        pkl_dir, _ = os.path.split(pkl_filepaths[0])

        # Plot resulting profiles
        if args.plot:
            yvars = {'Q_m': ['Qm']}
            plotBatch(pkl_dir, pkl_filepaths, yvars)

    except InputError as err:
        logger.error(err)
        sys.exit(1)
