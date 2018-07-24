# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-25 14:50:39
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-07-24 11:54:34

""" Run batch electrical titrations of specific "point-neuron" models. """

import sys
import os
import logging
import numpy as np
from argparse import ArgumentParser

from PointNICE.utils import logger, InputError
from PointNICE.solvers import setBatchDir, checkBatchLog, titrateEStimBatch
from PointNICE.plt import plotBatch


# Neurons
neurons = ['RS']

# Stimulation parameters
stim_params = {
    # 'amps': [20.0],  # mA/m2
    'durations': [0.5],  # s
    'PRFs': [1e1, 1e2],  # Hz
    'DCs': [1.0]
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

        # Run titration batch
        pkl_filepaths = titrateEStimBatch(batch_dir, log_filepath, neurons, stim_params,
                                          multiprocess=args.multiprocessing)
        pkl_dir, _ = os.path.split(pkl_filepaths[0])

        # Plot resulting profiles
        if args.plot:
            yvars = {'V_m': ['Vm']}
            plotBatch(pkl_dir, pkl_filepaths, yvars)

    except InputError as err:
        logger.error(err)
        sys.exit(1)
