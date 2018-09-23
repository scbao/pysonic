#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-02 17:50:10
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-23 11:24:24

""" Create lookup table for specific neuron. """

import os
import pickle
import logging
import numpy as np
from argparse import ArgumentParser

from PySONIC.utils import logger, getLookupDir
from PySONIC.batches import createQueue, runBatch
from PySONIC.neurons import getNeuronsDict
from PySONIC.core import SonicNeuron


# Default parameters
default = {
    'neuron': 'RS',
    'diams': np.array([16.0, 32.0, 64.0]),  # nm
    'freqs': np.array([20., 100., 500., 1e3, 2e3, 3e3, 4e3]),  # kHz
    'amps': np.insert(np.logspace(np.log10(0.1), np.log10(600), num=50), 0, 0.0),  # kPa
}


def computeAStimLookups(neuron, aref, fref, Aref, phi=np.pi, mpi=False, loglevel=logging.INFO):
    ''' Run simulations of the mechanical system for a multiple combinations of
        imposed US frequencies, acoustic amplitudes and charge densities, compute
        effective coefficients and store them in a dictionary of 3D arrays.

        :param neuron: neuron object
        :param aref: array of sonophore diameters (m)
        :param fref: array of acoustic drive frequencies (Hz)
        :param Aref: array of acoustic drive amplitudes (Pa)
        :param phi: acoustic drive phase (rad)
        :param mpi: boolean statting wether or not to use multiprocessing
        :return: lookups dictionary
    '''

    # Check validity of input parameters
    for key, values in {'diameters': aref, 'frequencies': fref, 'amplitudes': Aref}.items():
        if not (isinstance(values, list) or isinstance(values, np.ndarray)):
            raise TypeError('Invalid {} (must be provided as list or numpy array)'.format(key))
        if not all(isinstance(x, float) for x in values):
            raise TypeError('Invalid {} (must all be float typed)'.format(key))
        if len(values) == 0:
            raise ValueError('Empty {} array'.format(key))
        if key in ('diameters', 'frequencies') and min(values) <= 0:
            raise ValueError('Invalid {} (must all be strictly positive)'.format(key))
        if key is 'amplitudes' and min(values) < 0:
            raise ValueError('Invalid {} (must all be positive or null)'.format(key))

    # create simulation queue
    Qref = np.arange(neuron.Qbounds[0], neuron.Qbounds[1] + 1e-5, 1e-5)  # C/m2
    na, nf, nA, nQ = len(aref), len(fref), len(Aref), len(Qref)
    queue = createQueue((fref, Aref, Qref))

    # run simulations and populate outputs (list of lists)
    logger.info('Starting simulation batch for %s neuron', neuron.name)
    outputs = []
    for a in aref:
        sonic_neuron = SonicNeuron(a, neuron)
        outputs += runBatch(sonic_neuron, 'computeEffVars', queue, mpi=mpi, loglevel=loglevel)
    outputs = np.array(outputs).T

    # populate lookups dictionary with input vectors
    lookups = dict(
        a=aref,  # nm
        f=fref,  # Hz
        A=Aref,  # Pa
        Q=Qref  # C/m2
    )

    # reshape outputs into 4D arrays and add them to lookups dictionary
    logger.info('Reshaping output into lookup tables')
    for cname, output in zip(neuron.coeff_names, outputs):
        lookups[cname] = output.reshape(na, nf, nA, nQ)

    return lookups


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

    # Parse arguments
    args = ap.parse_args()
    neuron_str = args.neuron
    diams = (default['diams'] if args.diameters is None else np.array(args.diameters)) * 1e-9  # m
    freqs = (default['freqs'] if args.frequencies is None else np.array(args.frequencies)) * 1e3  # Hz
    amps = (default['amps'] if args.amplitudes is None else np.array(args.amplitudes)) * 1e3  # Pa

    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(loglevel)

    # Check neuron name validity
    if neuron_str not in getNeuronsDict():
        logger.error('Unknown neuron type: "%s"', neuron_str)
        return
    neuron = getNeuronsDict()[neuron_str]()

    # Check if lookup file already exists
    lookup_file = 'SONIC_{}.pkl'.format(neuron.name)
    lookup_filepath = '{0}/{1}'.format(getLookupDir(), lookup_file)
    if os.path.isfile(lookup_filepath):
        logger.warning('"%s" file already exists and will be overwritten. ' +
                       'Continue? (y/n)', lookup_file)
        user_str = input()
        if user_str not in ['y', 'Y']:
            logger.error('%s Lookup creation canceled', neuron.name)
            return

    lookup_dict = computeAStimLookups(neuron, diams, freqs, amps,
                                      mpi=args.multiprocessing, loglevel=loglevel)
    # Save dictionary in lookup file
    logger.info('Saving %s neuron lookup table in file: "%s"', neuron.name, lookup_file)
    with open(lookup_filepath, 'wb') as fh:
        pickle.dump(lookup_dict, fh)


if __name__ == '__main__':
    main()
