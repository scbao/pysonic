# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-02 17:50:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-02 12:47:44

''' Create lookup table for specific neuron. '''

import os
import itertools
import logging
import numpy as np

from PySONIC.utils import logger, isIterable, alert
from PySONIC.core import NeuronalBilayerSonophore, Batch, Lookup
from PySONIC.parsers import MechSimParser
from PySONIC.constants import DQ_LOOKUP


@alert
def computeAStimLookup(pneuron, aref, fref, Aref, Qref, fsref=None,
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

    # Populate reference vectors dictionary
    refs = {
        'a': aref,  # nm
        'f': fref,  # Hz
        'A': Aref,  # Pa
        'Q': Qref  # C/m2
    }

    # If multiple sonophore coverage values, ensure that only 1 value of
    # sonophore radius and US frequency are provided
    err_fs = 'cannot span {} for more than 1 {}'
    if fsref.size > 1 or fsref[0] != 1.:
        for x in ['a', 'f']:
            assert refs[x].size == 1, err_fs.format(descs['fs'], descs[x])

    # Add sonophore coverage vector to references
    refs['fs'] = fsref

    # Check validity of all reference vectors
    for key, values in refs.items():
        if not isIterable(values):
            raise TypeError(f'Invalid {descs[key]} (must be provided as list or numpy array)')
        if not all(isinstance(x, float) for x in values):
            raise TypeError(f'Invalid {descs[key]} (must all be float typed)')
        if len(values) == 0:
            raise ValueError(f'Empty {key} array')
        if key in ('a', 'f') and min(values) <= 0:
            raise ValueError(f'Invalid {descs[key]} (must all be strictly positive)')
        if key in ('A', 'fs') and min(values) < 0:
            raise ValueError(f'Invalid {descs[key]} (must all be positive or null)')

    # Get references dimensions
    dims = np.array([x.size for x in refs.values()])

    # Create simulation queue per sonophore radius
    queue = Batch.createQueue(fref, Aref, Qref)
    for i in range(len(queue)):
        queue[i].append(refs['fs'])

    # Run simulations and populate outputs
    logger.info('Starting simulation batch for %s neuron', pneuron.name)
    outputs = []
    for a in aref:
        nbls = NeuronalBilayerSonophore(a, pneuron)
        batch = Batch(nbls.computeEffVars, queue)
        outputs += batch(mpi=mpi, loglevel=loglevel)

    # Split comp times and effvars from outputs
    effvars, tcomps = [list(x) for x in zip(*outputs)]
    effvars = list(itertools.chain.from_iterable(effvars))

    # Make sure outputs size matches inputs dimensions product
    nout = len(effvars)
    assert nout == dims.prod(), 'Number of outputs does not match number of combinations'

    # Reshape effvars into nD arrays and add them to lookups dictionary
    logger.info('Reshaping output into lookup tables')
    varkeys = list(effvars[0].keys())
    tables = {}
    for key in varkeys:
        effvar = [effvars[i][key] for i in range(nout)]
        tables[key] = np.array(effvar).reshape(dims)

    # Reshape computation times, tile over extra fs dimension, and add it as a lookup table
    tcomps = np.array(tcomps).reshape(dims[:-1])
    tcomps = np.moveaxis(np.array([tcomps for i in range(dims[-1])]), 0, -1)
    tables['tcomp'] = tcomps

    # Construct and return lookup object
    return Lookup(refs, tables)


def main():

    parser = MechSimParser(outputdir='.')
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
            Qmin, Qmax = pneuron.Qbounds
            charges = np.arange(Qmin, Qmax + DQ_LOOKUP, DQ_LOOKUP)  # C/m2

        # Determine output filename
        f = NeuronalBilayerSonophore(32e-9, pneuron).getLookupFilePath
        if args['fs'].size == 1 and args['fs'][0] == 1.:
            lookup_fpath = f()
        else:
            lookup_fpath = f(a=args['radius'][0], Fdrive=args['freq'][0], fs=True)

        # Combine inputs into single list
        inputs = [args[x] for x in ['radius', 'freq', 'amp']] + [charges, args['fs']]

        # Adapt inputs and output filename if test case
        if args['test']:
            for i, x in enumerate(inputs):
                if x is not None and x.size > 1:
                    inputs[i] = np.array([x.min(), x.max()])
            fcode, fext = os.path.splitext(lookup_fpath)
            lookup_fpath = f'{fcode}_test{fext}'

        # Check if lookup file already exists
        if os.path.isfile(lookup_fpath):
            logger.warning('"%s" file already exists and will be overwritten. ' +
                           'Continue? (y/n)', lookup_fpath)
            user_str = input()
            if user_str not in ['y', 'Y']:
                logger.error('%s Lookup creation canceled', pneuron.name)
                return

        # Compute lookup
        lkp = computeAStimLookup(pneuron, *inputs, mpi=args['mpi'], loglevel=args['loglevel'])
        logger.info(f'Generated lookup: {lkp}')

        # Save lookup in PKL file
        logger.info('Saving %s neuron lookup in file: "%s"', pneuron.name, lookup_fpath)
        lkp.toPickle(lookup_fpath)


if __name__ == '__main__':
    main()
