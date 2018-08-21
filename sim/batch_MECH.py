#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-21 10:46:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-21 16:07:36

""" Run batch simulations of the NICE mechanical model with imposed charge densities """

import sys
import os
import logging
import numpy as np
from argparse import ArgumentParser

from PySONIC.utils import logger, InputError
from PySONIC.solvers import setBatchDir, checkBatchLog, runMechBatch
from PySONIC.neurons import *
from PySONIC.plt import plotBatch


a = 32e-9  # in-plane diameter (m)

# Electrical properties of the membrane
neuron = CorticalRS()
Cm0 = neuron.Cm0
Qm0 = neuron.Vm0 * 1e-5

# Stimulation parameters
stim_params = {
    'freqs': [500e3],  # Hz
    'amps': [50e3],  # Pa
    'charges': np.linspace(-72, 0, 3) * 1e-5  # C/m2
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
        log_filepath, _ = checkBatchLog(batch_dir, 'MECH')

        # Run MECH batch
        pkl_filepaths = runMechBatch(batch_dir, log_filepath, Cm0, Qm0, stim_params, a,
                                     multiprocess=args.multiprocessing)
        pkl_dir, _ = os.path.split(pkl_filepaths[0])

        # Plot resulting profiles
        if args.plot:
            plotBatch(pkl_dir, pkl_filepaths)

    except InputError as err:
        logger.error(err)
        sys.exit(1)
