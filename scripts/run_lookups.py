# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-02 17:50:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-25 17:26:56

''' Create lookup table for specific neuron. '''

import os
import itertools
import pickle
import logging
import numpy as np

from PySONIC.utils import logger, isIterable
from PySONIC.neurons import NEURONS_LOOKUP_DIR, getNeuronLookupsFileName
from PySONIC.core import NeuronalBilayerSonophore, createQueue, Batch
from PySONIC.parsers import MechSimParser


def computeAStimLookups(pneuron, aref, fref, Aref, Qref, fsref=None,
                        mpi=False, loglevel=logging.INFO):
    ''' Run simulations of the mechanical system for a multiple combinations of
        imposed sonophore radius, US frequencies, acoustic amplitudes charge densities and
        (spatially-averaged) sonophore membrane coverage fractions, compute effective
        coefficients and store them in a dictionary of n-dimensional arrays.

        :param pneuron: point-neuron model
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

    # Check inputs compatibility
    err_fs = 'cannot span {} for more than 1 {}'
    if fsref.size > 1 or fsref[0] != 1.:
        for x in ['a', 'f']:
            assert inputs[x].size == 1, err_fs.format(descs['fs'], descs[x])
    inputs['fs'] = fsref

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

    # Get inputs dimensions
    dims = np.array([x.size for x in inputs.values()])
    ncombs = dims.prod()

    # Create simulation queue per radius
    queue = createQueue(fref, Aref, Qref)
    for i in range(len(queue)):
        queue[i].append(inputs['fs'])

    # Run simulations and populate outputs (list of lists)
    logger.info('Starting simulation batch for %s neuron', pneuron.name)
    outputs = []
    for a in aref:
        nbls = NeuronalBilayerSonophore(a, pneuron)
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
    tcomps = np.array(tcomps).reshape(dims[:-1])

    # Store inputs, lookup data and comp times in dictionary
    return {
        'input': inputs,
        'lookup': lookups,
        'tcomp': tcomps
    }


def main():

    parser = MechSimParser(outputdir=NEURONS_LOOKUP_DIR)
    parser.addNeuron()
    parser.addTest()
    parser.defaults['neuron'] = 'RS'
    parser.defaults['radius'] = np.array([16.0, 32.0, 64.0])  # nm
    parser.defaults['freq'] = np.array([20., 100., 500., 1e3, 2e3, 3e3, 4e3])  # kHz
    parser.defaults['amp'] = np.insert(
        np.logspace(np.log10(0.1), np.log10(600), num=50), 0, 0.0)  # kPa
    parser.defaults['charge'] = np.nan
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    for pneuron in args['neuron']:

        # Determine charge vector
        charges = args['charge']
        if charges.size == 1 and np.isnan(charges[0]):
            charges = np.arange(
                pneuron.Qbounds()[0], pneuron.Qbounds()[1] + 1e-5, 1e-5)  # C/m2

        # Determine output filename
        if args['fs'].size == 1 and args['fs'][0] == 1.:
            lookup_fname = getNeuronLookupsFileName(pneuron.name)
        else:
            lookup_fname = getNeuronLookupsFileName(
                pneuron.name, a=args['radius'][0], Fdrive=args['freq'][0], fs=True)

        # Combine inputs into single list
        inputs = [args[x] for x in ['radius', 'freq', 'amp']] + [charges, args['fs']]

        # Adapt inputs and output filename if test case
        if args['test']:
            for i, x in enumerate(inputs):
                if x is not None and x.size > 1:
                    inputs[i] = np.array([x.min(), x.max()])
            lookup_fname = '{}_test{}'.format(*os.path.splitext(lookup_fname))

        lookup_fpath = os.path.join(args['outputdir'], lookup_fname)

        # Check if lookup file already exists
        if os.path.isfile(lookup_fpath):
            logger.warning('"%s" file already exists and will be overwritten. ' +
                           'Continue? (y/n)', lookup_fpath)
            user_str = input()
            if user_str not in ['y', 'Y']:
                logger.error('%s Lookup creation canceled', pneuron.name)
                return

        # Compute lookups
        df = computeAStimLookups(pneuron, *inputs, mpi=args['mpi'], loglevel=args['loglevel'])

        # Save dictionary in lookup file
        logger.info('Saving %s neuron lookup table in file: "%s"', pneuron.name, lookup_fpath)
        with open(lookup_fpath, 'wb') as fh:
            pickle.dump(df, fh)


if __name__ == '__main__':
    main()
