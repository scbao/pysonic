# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-07-23 17:30:52

""" Run batch electrical simulations of specific "point-neuron" models. """

import sys
import os
import logging
import numpy as np
from argparse import ArgumentParser

from PointNICE.utils import logger, InputError
from PointNICE.solvers import setBatchDir, checkBatchLog, runEStimBatch
from PointNICE.plt import plotBatch


# Neurons
neurons = ['RS']

# Stimulation parameters
stim_params = {
    'amps': [10.0],  # mA/m2
    'durations': [300e-3],  # np.array([20, 40, 60, 80, 100, 150, 200, 250, 300]) * 1e-3,  # s
    'PRFs': [1e2],  # np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]) * 1e3,  # Hz
    'DCs': [0.7, 0.9, 1.0],  # np.array([1, 2, 5, 10, 25, 50, 75, 100]) * 1e-2
    'offsets': [100e-3]
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
        log_filepath, _ = checkBatchLog(batch_dir, 'E-STIM')

        # Run E-STIM batch
        pkl_filepaths = runEStimBatch(batch_dir, log_filepath, neurons, stim_params,
                                      multiprocess=args.multiprocessing)
        pkl_dir, _ = os.path.split(pkl_filepaths[0])

        # Plot resulting profiles
        if args.plot:
            yvars = {'V_m': ['Vm']}
            plotBatch(pkl_dir, pkl_filepaths, yvars)

    except InputError as err:
        logger.error(err)
        sys.exit(1)
