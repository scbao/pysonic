# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-03-15 18:33:59
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-19 16:02:47


""" Script to run MECH simulations from command line. """

import sys
import logging
from argparse import ArgumentParser

from PySONIC.utils import logger, InputError, selectDirDialog
from PySONIC.bls import BilayerSonophore
from PySONIC.solvers import checkBatchLog, MechWorker
from PySONIC.plt import plotBatch


# Default parameters
default = {
    'a': 32.0,  # nm
    'd': 0.0,  # um
    'f': 500.0,  # kHz
    'A': 100.0,  # kPa
    'Cm0': 1.0,  # uF/cm2
    'Qm0': 0.0,  # nC/cm2
    'Qm': 0.0,  # nC/cm2
}


def main():

    # Define argument parser
    ap = ArgumentParser()

    # ASTIM parameters
    ap.add_argument('-a', '--diameter', type=float, default=default['a'],
                    help='Sonophore diameter (nm)')
    ap.add_argument('-d', '--embedding', type=float, default=default['d'],
                    help='Embedding depth (um)')
    ap.add_argument('-f', '--frequency', type=float, default=default['f'],
                    help='Acoustic drive frequency (kHz)')
    ap.add_argument('-A', '--amplitude', type=float, default=default['A'],
                    help='Acoustic pressure amplitude (kPa)')
    ap.add_argument('-Cm0', '--restcapct', type=float, default=default['Cm0'],
                    help='Membrane resting capacitance (uF/cm2)')
    ap.add_argument('-Qm0', '--restcharge', type=float, default=default['Qm0'],
                    help='Membrane resting charge density (nC/cm2)')
    ap.add_argument('-Qm', '--charge', type=float, default=default['Qm'],
                    help='Applied charge density (nC/cm2)')
    ap.add_argument('-o', '--outputdir', type=str, default=None,
                    help='Output directory')

    # Boolean parameters
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-p', '--plot', default=False, action='store_true',
                    help='Plot results')

    # Parse arguments
    args = ap.parse_args()
    a = args.diameter * 1e-9  # m
    d = args.embedding * 1e-6  # m
    Fdrive = args.frequency * 1e3  # Hz
    Adrive = args.amplitude * 1e3  # Pa
    Cm0 = args.restcapct * 1e-2  # F/m2
    Qm0 = args.restcharge * 1e-5  # C/m2
    Qm = args.charge * 1e-5  # C/m2
    output_dir = selectDirDialog() if args.outputdir is None else args.outputdir

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    try:
        log_filepath, _ = checkBatchLog(output_dir, 'MECH')
        worker = MechWorker(0, output_dir, log_filepath, BilayerSonophore(a, Fdrive, Cm0, Qm0, d),
                            Fdrive, Adrive, Qm, 1)
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
