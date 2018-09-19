#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-19 16:02:38

""" Script to run ESTIM simulations from command line. """

import sys
import logging
from argparse import ArgumentParser

from PySONIC.utils import logger, getNeuronsDict, InputError, selectDirDialog
from PySONIC.solvers import checkBatchLog, SolverElec, EStimWorker
from PySONIC.plt import plotBatch


# Default parameters
default = {
    'neuron': 'RS',
    'A': 10.0,  # mA/m2
    't': 150.0,  # ms
    'off': 20.0,  # ms
    'PRF': 100.0,  # Hz
    'DC': 100.0  # %
}


def main():

    # Define argument parser
    ap = ArgumentParser()

    # ASTIM parameters
    ap.add_argument('-n', '--neuron', type=str, default=default['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-A', '--amplitude', type=float, default=default['A'],
                    help='Stimulus amplitude (mA/m2)')
    ap.add_argument('-t', '--duration', type=float, default=default['t'],
                    help='Stimulus duration (ms)')
    ap.add_argument('--offset', type=float, default=default['off'],
                    help='Offset duration (ms)')
    ap.add_argument('--PRF', type=float, default=default['PRF'],
                    help='PRF (Hz)')
    ap.add_argument('--DC', type=float, default=default['DC'],
                    help='Duty cycle (%%)')
    ap.add_argument('-o', '--outputdir', type=str, default=None,
                    help='Output directory')

    # Boolean arguments
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-p', '--plot', default=False, action='store_true',
                    help='Plot results')

    # Parse arguments
    args = ap.parse_args()
    neuron_str = args.neuron
    Astim = args.amplitude  # mA/m2
    tstim = args.duration * 1e-3  # s
    toffset = args.offset * 1e-3  # s
    PRF = args.PRF  # Hz
    DC = args.DC * 1e-2
    output_dir = selectDirDialog() if args.outputdir is None else args.outputdir

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    try:
        if neuron_str not in getNeuronsDict():
            raise InputError('Unknown neuron type: "{}"'.format(neuron_str))
        log_filepath, _ = checkBatchLog(output_dir, 'E-STIM')
        neuron = getNeuronsDict()[neuron_str]()
        worker = EStimWorker(0, output_dir, log_filepath, SolverElec(), neuron, Astim, tstim, toffset,
                             PRF, DC, 1)
        logger.info('%s', worker)
        outfile = worker.__call__()
        logger.info('Finished')
        if args.plot:
            plotBatch(output_dir, [outfile])

    except InputError as err:
        logger.error(err)
        sys.exit(1)


if __name__ == '__main__':
    main()
