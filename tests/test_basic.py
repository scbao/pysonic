#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-14 18:37:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-12 12:53:09

''' Test the basic functionalities of the package. '''

import os
import sys
import logging
import time
import cProfile
import pstats
from argparse import ArgumentParser

from PySONIC.core import BilayerSonophore, NeuronalBilayerSonophore
from PySONIC.utils import logger
from PySONIC.neurons import getPointNeuron


def execute(func_str, globals, locals, is_profiled):
    ''' Execute function with or without profiling. '''
    if is_profiled:
        pfile = 'tmp.stats'
        cProfile.runctx(func_str, globals, locals, pfile)
        stats = pstats.Stats(pfile)
        os.remove(pfile)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats()
    else:
        eval(func_str, globals, locals)


def test_MECH(is_profiled=False):
    ''' Mechanical simulation. '''
    logger.info('Test: running MECH simulation')

    # Create BLS instance
    a = 32e-9  # m
    Qm0 = -80e-5  # membrane resting charge density (C/m2)
    Cm0 = 1e-2  # membrane resting capacitance (F/m2)
    bls = BilayerSonophore(a, Cm0, Qm0)

    # Stimulation parameters
    Fdrive = 350e3  # Hz
    Adrive = 100e3  # Pa
    Qm = 50e-5  # C/m2

    # Run simulation
    execute('bls.simulate(Fdrive, Adrive, Qm)', globals(), locals(), is_profiled)


def test_ESTIM(is_profiled=False):
    ''' Electrical simulation '''

    logger.info('Test: running ESTIM simulation')

    # Initialize neuron
    pneuron = getPointNeuron('RS')

    # Stimulation parameters
    Astim = 10.0  # mA/m2
    tstim = 100e-3  # s
    toffset = 50e-3  # s

    # Run simulation
    execute('pneuron.simulate(Astim, tstim, toffset)', globals(), locals(), is_profiled)


def test_ASTIM_sonic(is_profiled=False):
    ''' Effective acoustic simulation '''

    logger.info('Test: ASTIM sonic simulation')

    # Default parameters
    a = 32e-9  # m
    pneuron = getPointNeuron('RS')
    nbls = NeuronalBilayerSonophore(a, pneuron)

    Fdrive = 500e3  # Hz
    Adrive = 100e3  # Pa
    tstim = 50e-3  # s
    toffset = 10e-3  # s

    # test error 1: sonophore radius outside of lookup range
    try:
        nbls = NeuronalBilayerSonophore(100e-9, pneuron)
        nbls.simulate(Fdrive, Adrive, tstim, toffset, method='sonic')
    except ValueError as err:
        logger.debug('Out of range radius: OK')

    # test error 2: frequency outside of lookups range
    try:
        nbls = NeuronalBilayerSonophore(a, pneuron)
        nbls.simulate(10e3, Adrive, tstim, toffset, method='sonic')
    except ValueError as err:
        logger.debug('Out of range frequency: OK')

    # test error 3: amplitude outside of lookups range
    try:
        nbls = NeuronalBilayerSonophore(a, pneuron)
        nbls.simulate(Fdrive, 1e6, tstim, toffset, method='sonic')
    except ValueError as err:
        logger.debug('Out of range amplitude: OK')

    # Run simulation
    execute("nbls.simulate(Fdrive, Adrive, tstim, toffset, method='sonic')",
            globals(), locals(), is_profiled)


def test_ASTIM_full(is_profiled=False):
    ''' Classic acoustic simulation '''

    logger.info('Test: running ASTIM classic simulation')

    # Initialize sonic neuron
    a = 32e-9  # m
    pneuron = getPointNeuron('RS')
    nbls = NeuronalBilayerSonophore(a, pneuron)

    # Stimulation parameters
    Fdrive = 500e3  # Hz
    Adrive = 100e3  # Pa
    tstim = 1e-6  # s
    toffset = 1e-6  # s

    # Run simulation
    execute("nbls.simulate(Fdrive, Adrive, tstim, toffset, method='full')",
            globals(), locals(), is_profiled)


def test_ASTIM_hybrid(is_profiled=False):
    ''' Hybrid acoustic simulation '''

    logger.info('Test: running ASTIM hybrid simulation')

    # Initialize sonic neuron
    a = 32e-9  # m
    pneuron = getPointNeuron('RS')
    nbls = NeuronalBilayerSonophore(a, pneuron)

    # Stimulation parameters
    Fdrive = 350e3  # Hz
    Adrive = 100e3  # Pa
    tstim = 1e-3  # s
    toffset = 1e-3  # s

    # Run simulation
    execute("nbls.simulate(Fdrive, Adrive, tstim, toffset, method='hybrid')",
            globals(), locals(), is_profiled)


def test_all():
    t0 = time.time()
    test_MECH()
    test_ESTIM()
    test_ASTIM_sonic()
    test_ASTIM_full()
    test_ASTIM_hybrid()
    tcomp = time.time() - t0
    logger.info('All tests completed in %.0f s', tcomp)



def main():


    # Define valid test sets
    valid_testsets = [
        'MECH',
        'ESTIM',
        'ASTIM_sonic',
        'ASTIM_full',
        'ASTIM_hybrid',
        'all'
    ]

    # Define argument parser
    ap = ArgumentParser()

    ap.add_argument('-t', '--testset', type=str, default='all', choices=valid_testsets,
                    help='Specific test set')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-p', '--profile', default=False, action='store_true',
                    help='Profile test set')

    # Parse arguments
    args = ap.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    if args.profile and args.testset == 'all':
        logger.error('profiling can only be run on individual tests')
        sys.exit(2)

    # Run test
    if args.testset == 'all':
        test_all()
    else:
        possibles = globals().copy()
        possibles.update(locals())
        method = possibles.get('test_{}'.format(args.testset))
        method(args.profile)
    sys.exit(0)


if __name__ == '__main__':
    main()
