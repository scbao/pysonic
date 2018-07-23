# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-03-15 18:33:59
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-05-03 12:13:26


""" Script to run MECH simulations from command line. """

import sys
import os
import logging
from argparse import ArgumentParser

from PointNICE.utils import logger, InputError, si_format
from PointNICE.bls import BilayerSonophore
from PointNICE.solvers import checkBatchLog, runMech
from PointNICE.plt import plotBatch


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
    ap.add_argument('-o', '--outputdir', type=str, default=os.getcwd(),
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
    output_dir = args.outputdir

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    try:
        log_filepath, _ = checkBatchLog(output_dir, 'MECH')
        logger.info('Running MECH simulation: a = %sm, f = %sHz, A = %sPa, Cm0 = %sF/cm2, '
                    'Qm0 = %sC/cm2, Qm0 = %sC/cm2', si_format(a, 1), si_format(Fdrive, 1),
                    si_format(Adrive, 2), si_format(Cm0 * 1e-4), si_format(Qm0 * 1e-4),
                    si_format(Qm * 1e-4))
        bls = BilayerSonophore(a, Fdrive, Cm0, Qm0, d)
        outfile = runMech(output_dir, log_filepath, bls, Fdrive, Adrive, Qm)
        logger.info('Finished')
        if args.plot:
            plotBatch(output_dir, [outfile])

    except InputError as err:
        logger.error(err)
        sys.exit(1)


if __name__ == '__main__':
    main()