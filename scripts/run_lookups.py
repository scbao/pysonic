#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-02 17:50:10
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-07 15:41:10

''' Create lookup table for specific neuron. '''

import os
import itertools
import pickle
import logging
import numpy as np
from argparse import ArgumentParser

from PySONIC.utils import logger, getNeuronLookupsFile, isIterable
from PySONIC.neurons import getPointNeuron
from PySONIC.core import NeuronalBilayerSonophore, createQueue, Batch


# Default parameters
defaults = dict(
    neuron='RS',
    radius=np.array([16.0, 32.0, 64.0]),  # nm
    freq=np.array([20., 100., 500., 1e3, 2e3, 3e3, 4e3]),  # kHz
    amp=np.insert(np.logspace(np.log10(0.1), np.log10(600), num=50), 0, 0.0),  # kPa
)


def computeAStimLookups(neuron, aref, fref, Aref, Qref, fsref=None,
                        mpi=False, loglevel=logging.INFO):
    ''' Run simulations of the mechanical system for a multiple combinations of
        imposed sonophore radius, US frequencies, acoustic amplitudes charge densities and
        (spatially-averaged) sonophore membrane coverage fractions, compute effective
        coefficients and store them in a dictionary of n-dimensional arrays.

        :param neuron: neuron object
        :param aref: array of sonophore radii (m)
        :param fref: array of acoustic drive frequencies (Hz)
        :param Aref: array of acoustic drive amplitudes (Pa)
        :param Qref: array of membrane charge densities (C/m2)
        :param fsref: acoustic drive phase (rad)
        :param mpi: boolean statting wether or not to use multiprocessing
        :param loglevel: logging level
        :return: lookups dictionary
    '''

    descs = {
        'a': 'sonophore radii',
        'f': 'US frequencies',
        'A': 'US amplitudes',
        'fs': 'sonophore membrane coverage fractions'
    }

    # Populate inputs dictionary
    inputs = {
        'a': aref,  # nm
        'f': fref,  # Hz
        'A': Aref,  # Pa
        'Q': Qref  # C/m2
    }

    # Add fs to inputs if provided, otherwise add default value (1)
    err_fs = 'cannot span {} for more than 1 {}'
    if fsref is not None:
        for x in ['a', 'f']:
            assert inputs[x].size == 1, err_fs.format(descs['fs'], descs[x])
        inputs['fs'] = fsref
    else:
        inputs['fs'] = np.array([1.])

    # Check validity of input parameters
    for key, values in inputs.items():
        if not isIterable(values):
            raise TypeError(
                'Invalid {} (must be provided as list or numpy array)'.format(descs[key]))
        if not all(isinstance(x, float) for x in values):
            raise TypeError('Invalid {} (must all be float typed)'.format(descs[key]))
        if len(values) == 0:
            raise ValueError('Empty {} array'.format(key))
        if key in ('a', 'f') and min(values) <= 0:
            raise ValueError('Invalid {} (must all be strictly positive)'.format(descs[key]))
        if key in ('A', 'fs') and min(values) < 0:
            raise ValueError('Invalid {} (must all be positive or null)'.format(descs[key]))

    # Get dimensions of inputs that have more than one value
    dims = np.array([x.size for x in inputs.values()])
    dims = dims[dims > 1]
    ncombs = dims.prod()

    # Create simulation queue per radius
    queue = createQueue(fref, Aref, Qref)
    for i in range(len(queue)):
        queue[i].append(inputs['fs'])

    # Run simulations and populate outputs (list of lists)
    logger.info('Starting simulation batch for %s neuron', neuron.name)
    outputs = []
    for a in aref:
        nbls = NeuronalBilayerSonophore(a, neuron)
        batch = Batch(nbls.computeEffVars, queue)
        outputs += batch(mpi=mpi, loglevel=loglevel)

    # Split comp times and effvars from outputs
    tcomps, effvars = [list(x) for x in zip(*outputs)]
    effvars = list(itertools.chain.from_iterable(effvars))

    # Reshape effvars into nD arrays and add them to lookups dictionary
    logger.info('Reshaping output into lookup tables')
    varkeys = list(effvars[0].keys())
    nout = len(effvars)
    assert nout == ncombs, 'number of outputs does not match number of combinations'
    lookups = {}
    for key in varkeys:
        effvar = [effvars[i][key] for i in range(nout)]
        lookups[key] = np.array(effvar).reshape(dims)

    # Reshape comp times into nD array (minus fs dimension)
    if fsref is not None:
        dims = dims[:-1]
    tcomps = np.array(tcomps).reshape(dims)

    # Store inputs, lookup data and comp times in dictionary
    df = {
        'input': inputs,
        'lookup': lookups,
        'tcomp': tcomps
    }

    return df


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('--mpi', default=False, action='store_true', help='Use multiprocessing')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-t', '--test', default=False, action='store_true', help='Test configuration')

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-a', '--radius', nargs='+', type=float, help='Sonophore radius (nm)')
    ap.add_argument('-f', '--freq', nargs='+', type=float, help='US frequency (kHz)')
    ap.add_argument('-A', '--amp', nargs='+', type=float, help='Acoustic pressure amplitude (kPa)')
    ap.add_argument('-Q', '--charge', nargs='+', type=float,
                    help='Membrane charge density (nC/cm2)')
    ap.add_argument('--spanFs', default=False, action='store_true',
                    help='Span sonophore coverage fraction')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    loglevel = logging.DEBUG if args['verbose'] is True else logging.INFO
    logger.setLevel(loglevel)
    mpi = args['mpi']
    neuron_str = args['neuron']
    radii = np.array(args.get('radius', defaults['radius'])) * 1e-9  # m
    freqs = np.array(args.get('freq', defaults['freq'])) * 1e3  # Hz
    amps = np.array(args.get('amp', defaults['amp'])) * 1e3  # Pa

    # Check neuron name validity
    try:
        neuron = getPointNeuron(neuron_str)
    except ValueError as err:
        logger.error(err)
        return

    # Determine charge vector
    if 'charge' in args:
        charges = np.array(args['charge']) * 1e-5  # C/m2
    else:
        charges = np.arange(neuron.Qbounds()[0], neuron.Qbounds()[1] + 1e-5, 1e-5)  # C/m2

    # Determine fs vector
    fs = None
    if args['spanFs']:
        fs = np.linspace(0, 100, 101) * 1e-2  # (-)

    # Determine output filename
    lookup_path = {
        True: getNeuronLookupsFile(neuron.name),
        False: getNeuronLookupsFile(neuron.name, a=radii[0], Fdrive=freqs[0], fs=True)
    }[fs is None]

    # Combine inputs into single list
    inputs = [radii, freqs, amps, charges, fs]

    # Adapt inputs and output filename if test case
    if args['test']:
        for i, x in enumerate(inputs):
            if x is not None and x.size > 1:
                inputs[i] = np.array([x.min(), x.max()])
        lookup_path = '{}_test{}'.format(*os.path.splitext(lookup_path))

    # Check if lookup file already exists
    if os.path.isfile(lookup_path):
        logger.warning('"%s" file already exists and will be overwritten. ' +
                       'Continue? (y/n)', lookup_path)
        user_str = input()
        if user_str not in ['y', 'Y']:
            logger.error('%s Lookup creation canceled', neuron.name)
            return

    # Compute lookups
    df = computeAStimLookups(neuron, *inputs, mpi=mpi, loglevel=loglevel)

    # Save dictionary in lookup file
    logger.info('Saving %s neuron lookup table in file: "%s"', neuron.name, lookup_path)
    with open(lookup_path, 'wb') as fh:
        pickle.dump(df, fh)


if __name__ == '__main__':
    main()
