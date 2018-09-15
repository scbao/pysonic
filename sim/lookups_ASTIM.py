#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-02 17:50:10
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-15 17:07:56

""" Create lookup table for specific neuron. """

import os
import sys
import pickle
import logging
import numpy as np
from argparse import ArgumentParser

from PySONIC.solvers import computeAStimLookups, computeTestLookups
from PySONIC.utils import logger, InputError, getNeuronsDict, getLookupDir


# Default parameters
default = {
    'neuron': 'RS',
    'diams': np.array([15.0, 32.0, 70.0, 150.0]),  # nm
    'freqs': np.array([20., 100., 500., 1e3, 2e3, 3e3, 4e3]),  # kHz
    'amps': np.insert(np.logspace(np.log10(0.1), np.log10(600), num=50), 0, 0.0),  # kPa
}


def main():

    # Define argument parser
    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=default['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-a', '--diameters', nargs='+', type=float, default=None,
                    help='Sonophore diameters (nm)')
    ap.add_argument('-f', '--frequencies', nargs='+', type=float, default=None,
                    help='Acoustic drive frequencies (kHz)')
    ap.add_argument('-A', '--amplitudes', nargs='+', type=float, default=None,
                    help='Acoustic pressure amplitudes (kPa)')

    # Boolean parameters
    ap.add_argument('-m', '--multiprocessing', default=False, action='store_true',
                    help='Use multiprocessing')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-t', '--test', default=False, action='store_true',
                    help='Run test batch')

    # Parse arguments
    args = ap.parse_args()
    neuron_str = args.neuron
    diams = (default['diams'] if args.diameters is None else np.array(args.diameters)) * 1e-9  # m
    freqs = (default['freqs'] if args.frequencies is None else np.array(args.frequencies)) * 1e3  # Hz
    amps = (default['amps'] if args.amplitudes is None else np.array(args.amplitudes)) * 1e3  # Pa

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Check neuron name validity
    if neuron_str not in getNeuronsDict():
        raise InputError('Unknown neuron type: "{}"'.format(neuron_str))
    neuron = getNeuronsDict()[neuron_str]()

    # Check if lookup file already exists
    lookup_file = '{}_lookups.pkl'.format(neuron.name)
    lookup_filepath = '{0}/{1}'.format(getLookupDir(), lookup_file)
    if os.path.isfile(lookup_filepath):
        logger.warning('"%s" file already exists and will be overwritten. ' +
                       'Continue? (y/n)', lookup_file)
        user_str = input()
        if user_str not in ['y', 'Y']:
            logger.info('%s Lookup creation canceled', neuron.name)
            sys.exit(0)

    try:
        if args.test:
            # Compute test lookups
            lookup_dict = computeTestLookups(neuron, diams, freqs, amps,
                                             multiprocess=args.multiprocessing)
        else:
            # Compute real lookups
            lookup_dict = computeAStimLookups(neuron, diams, freqs, amps,
                                              multiprocess=args.multiprocessing)
        # Save dictionary in lookup file
        logger.info('Saving %s neuron lookup table in file: "%s"', neuron.name, lookup_file)
        with open(lookup_filepath, 'wb') as fh:
            pickle.dump(lookup_dict, fh)

    except InputError as err:
        logger.error(err)
        sys.exit(1)


if __name__ == '__main__':
    main()
