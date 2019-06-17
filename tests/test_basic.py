# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-06-14 18:37:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-17 21:37:43

''' Test the basic functionalities of the package. '''

import os
import time
import cProfile
import pstats

from PySONIC.core import BilayerSonophore, NeuronalBilayerSonophore
from PySONIC.utils import logger
from PySONIC.neurons import getPointNeuron
from PySONIC.parsers import TestParser


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
    except ValueError:
        logger.debug('Out of range radius: OK')

    # test error 2: frequency outside of lookups range
    try:
        nbls = NeuronalBilayerSonophore(a, pneuron)
        nbls.simulate(10e3, Adrive, tstim, toffset, method='sonic')
    except ValueError:
        logger.debug('Out of range frequency: OK')

    # test error 3: amplitude outside of lookups range
    try:
        nbls = NeuronalBilayerSonophore(a, pneuron)
        nbls.simulate(Fdrive, 1e6, tstim, toffset, method='sonic')
    except ValueError:
        logger.debug('Out of range amplitude: OK')

    # Run simulation
    execute("nbls.simulate(Fdrive, Adrive, tstim, toffset, method='sonic')",
            globals(), locals(), is_profiled)


def test_ASTIM_dense(is_profiled=False):
    ''' Classic acoustic simulation '''

    logger.info('Test: running ASTIM detailed simulation')

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


def main():

    test_funcs = {
        'MECH': test_MECH,
        'ESTIM': test_ESTIM,
        'ASTIM_sonic': test_ASTIM_sonic,
        'ASTIM_dense': test_ASTIM_dense,
        'ASTIM_hybrid': test_ASTIM_hybrid
    }

    parser = TestParser(list(test_funcs.keys()))
    args = parser.parse()
    logger.setLevel(args['loglevel'])

    # Parse arguments
    if args['profile'] and args['subset'] == 'all':
        logger.error('profiling can only be run on individual tests')
        return

    # Run test
    if args['subset'] == ['all']:
        t0 = time.time()
        for k, test_func in test_funcs.items():
            test_func(args['profile'])
        tcomp = time.time() - t0
        logger.info('All tests completed in %.0f s', tcomp)
    else:
        for s in args['subset']:
            test_funcs[s](args['profile'])


if __name__ == '__main__':
    main()
