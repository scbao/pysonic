#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-07 18:51:34

""" Script to run ESTIM simulations from command line. """

import sys
import os
import logging
from argparse import ArgumentParser

from PointNICE.utils import logger, getNeuronsDict
from PointNICE.solvers import checkBatchLog, SolverElec, runEStim
from PointNICE.plt import plotBatch


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
    ap.add_argument('-o', '--outputdir', type=str, default=os.getcwd(),
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
    output_dir = args.outputdir

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    PW_str = ', PRF = {:.2f} Hz, DC = {:.1f}'.format(PRF, DC * 1e2)

    try:
        assert neuron_str in getNeuronsDict(), '{} neuron type not implemented'.format(neuron_str)
        log_filepath, _ = checkBatchLog(output_dir, 'E-STIM')
        neuron = getNeuronsDict()[neuron_str]()
        solver = SolverElec()
        logger.info('Running E-STIM simulation on %s neuron: A = %.2f mA/m2, t = %.2f ms%s',
                    neuron_str, Astim, tstim * 1e3, PW_str if DC < 1.0 else "")
        outfile = runEStim(output_dir, log_filepath, solver, neuron, Astim, tstim, toffset, PRF, DC)
        logger.info('Finished')
        if args.plot:
            plotBatch(output_dir, [outfile])

    except AssertionError as err:
        logger.error(err)
        sys.exit(1)


if __name__ == '__main__':
    main()
