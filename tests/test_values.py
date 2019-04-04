#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-14 18:37:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-04-03 21:00:56

''' Run functionalities of the package and test validity of outputs. '''

import sys
import logging
from argparse import ArgumentParser
import numpy as np

from PySONIC.utils import logger
from PySONIC.core import BilayerSonophore, NeuronalBilayerSonophore
from PySONIC.neurons import getNeuronsDict

from PySONIC.constants import *

# Set logging level
logger.setLevel(logging.INFO)


def test_MECH():
    ''' Maximal negative and positive deflections of the BLS structure for a specific
        sonophore size, resting membrane properties and stimulation parameters. '''

    logger.info('Starting test: Mechanical simulation')

    # Create BLS instance
    a = 32e-9  # m
    Cm0 = 1e-2  # membrane resting capacitance (F/m2)
    Qm0 = -80e-5  # membrane resting charge density (C/m2)
    bls = BilayerSonophore(a, Cm0, Qm0)

    # Run mechanical simulation
    Fdrive = 350e3  # Hz
    Adrive = 100e3  # Pa
    Qm = 50e-5  # C/m2
    _, y, _ = bls.simulate(Fdrive, Adrive, Qm)

    # Check validity of deflection extrema
    Z, _ = y
    Zlast = Z[-NPC_FULL:]
    Zmin, Zmax = (Zlast.min(), Zlast.max())
    logger.info('Zmin = %.2f nm, Zmax = %.2f nm', Zmin * 1e9, Zmax * 1e9)
    Zmin_ref, Zmax_ref = (-0.116e-9, 5.741e-9)
    assert np.abs(Zmin - Zmin_ref) < 1e-12, 'Unexpected sonophore compression amplitude'
    assert np.abs(Zmax - Zmax_ref) < 1e-12, 'Unexpected sonophore expansion amplitude'

    logger.info('Passed test: Mechanical simulation')


def test_resting_potential():
    ''' Neurons membrane potential in free conditions should stabilize to their
        specified resting potential value. '''

    conv_err_msg = ('{} neuron membrane potential in free conditions does not converge to '
                    'stable value (gap after 20s: {:.2e} mV)')
    value_err_msg = ('{} neuron steady-state membrane potential in free conditions differs '
                     'significantly from specified resting potential (gap = {:.2f} mV)')

    logger.info('Starting test: neurons resting potential')

    for Neuron in getNeuronsDict().values():

        # Simulate each neuron in free conditions
        neuron = Neuron()

        logger.info('%s neuron simulation in free conditions', neuron.name)

        _, y, _ = neuron.simulate(Astim=0.0, tstim=20.0, toffset=0.0)
        Vm_free, *_ = y

        # Check membrane potential convergence
        Vm_free_last, Vm_free_beforelast = (Vm_free[-1], Vm_free[-2])
        Vm_free_conv = Vm_free_last - Vm_free_beforelast
        assert np.abs(Vm_free_conv) < 1e-5, conv_err_msg.format(neuron.name, Vm_free_conv)

        # Check membrane potential convergence to resting potential
        Vm_free_diff = Vm_free_last - neuron.Vm0
        assert np.abs(Vm_free_diff) < 0.1, value_err_msg.format(neuron.name, Vm_free_diff)

    logger.info('Passed test: neurons resting potential')


def test_ESTIM():
    ''' Threshold E-STIM amplitude and needed to obtain an action potential and response latency
        should match reference values. '''

    Athr_err_msg = ('{} neuron threshold amplitude for excitation does not match reference value'
                    '(gap = {:.2f} mA/m2)')
    latency_err_msg = ('{} neuron latency for excitation at threshold amplitude does not match '
                       'reference value (gap = {:.2f} ms)')

    logger.info('Starting test: E-STIM titration')

    # Stimulation parameters
    tstim = 100e-3  # s
    toffset = 50e-3  # s

    # Reference values
    Athr_refs = {'FS': 6.91, 'LTS': 1.54, 'RS': 5.03, 'RE': 3.61, 'TC': 4.05,
                 'LeechT': 4.66, 'LeechP': 13.72, 'IB': 3.08}

    for Neuron in getNeuronsDict().values():

        # Perform titration for each neuron
        neuron = Neuron()
        logger.info('%s neuron titration', neuron.name)
        Athr = neuron.titrate(tstim, toffset)

        # Check threshold amplitude
        Athr_diff = Athr - Athr_refs[neuron.name]
        assert np.abs(Athr_diff) < 0.1, Athr_err_msg.format(neuron.name, Athr_diff)

    logger.info('Passed test: E-STIM titration')


def test_ASTIM():
    ''' Threshold A-STIM amplitude and needed to obtain an action potential and response latency
        should match reference values. '''

    Athr_err_msg = ('{} neuron threshold amplitude for excitation does not match reference value'
                    '(gap = {:.2f} kPa)')
    latency_err_msg = ('{} neuron latency for excitation at threshold amplitude does not match '
                       'reference value (gap = {:.2f} ms)')

    logger.info('Starting test: A-STIM titration')

    # Sonophore radius
    a = 32e-9  # m

    # Stimulation parameters
    Fdrive = 350e3  # Hz
    tstim = 50e-3  # s
    toffset = 30e-3  # s

    # Reference values
    Athr_refs = {'FS': 38.96e3, 'LTS': 24.90e3, 'RS': 50.90e3, 'RE': 46.36e3, 'TC': 23.14e3,
                 'LeechT': 21.02e3, 'LeechP': 22.23e3, 'IB': 91.26e3}

    # Titration for each neuron
    for Neuron in getNeuronsDict().values():

        # Initialize sonic neuron
        neuron = Neuron()
        nbls = NeuronalBilayerSonophore(a, neuron)
        logger.info('%s neuron titration', neuron.name)

        # Perform titration
        Athr = nbls.titrate(Fdrive, tstim, toffset, method='sonic')

        # Check threshold amplitude
        Athr_diff = (Athr - Athr_refs[neuron.name]) * 1e-3
        assert np.abs(Athr_diff) < 0.1, Athr_err_msg.format(neuron.name, Athr_diff)

    logger.info('Passed test: A-STIM titration')


def test_all():
    logger.info('Starting tests')
    test_MECH()
    test_resting_potential()
    test_ESTIM()
    test_ASTIM()
    logger.info('All tests successfully passed')



def main():

    # Define valid test sets
    valid_testsets = [
        'MECH',
        'resting_potential',
        'ESTIM',
        'ASTIM',
        'all'
    ]

    # Define argument parser
    ap = ArgumentParser()

    ap.add_argument('-t', '--testset', type=str, default='all', choices=valid_testsets,
                    help='Specific test set')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')

    # Parse arguments
    args = ap.parse_args()
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Run test
    try:
        if args.testset == 'all':
            test_all()
        else:
            possibles = globals().copy()
            possibles.update(locals())
            method = possibles.get('test_{}'.format(args.testset))
            method()
        sys.exit(0)
    except AssertionError as e:
        logger.error(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
