#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-14 18:37:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-16 12:01:10

''' Run functionalities of the package and test validity of outputs. '''

import sys
import logging
from argparse import ArgumentParser
import numpy as np

from PointNICE.utils import logger, getNeuronsDict
from PointNICE import BilayerSonophore, SolverElec, SolverUS
from PointNICE.solvers import detectSpikes, titrateEStim, titrateAStim
from PointNICE.constants import *

# Set logging level
logger.setLevel(logging.INFO)

# List of implemented neurons
neurons = getNeuronsDict()
neurons = list(neurons.values())


def test_MECH():
    ''' Maximal negative and positive deflections of the BLS structure for a specific
        sonophore size, resting membrane properties and stimulation parameters. '''

    logger.info('Starting test: Mechanical simulation')

    # Create BLS instance
    a = 32e-9  # m
    Fdrive = 350e3  # Hz
    Cm0 = 1e-2  # membrane resting capacitance (F/m2)
    Qm0 = -80e-5  # membrane resting charge density (C/m2)
    bls = BilayerSonophore(a, Fdrive, Cm0, Qm0)

    # Run mechanical simulation
    Adrive = 100e3  # Pa
    Qm = 50e-5  # C/m2
    _, y, _ = bls.run(Fdrive, Adrive, Qm)

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

    # Initialize solver
    solver = SolverElec()
    for neuron_class in neurons:

        # Simulate each neuron in free conditions
        neuron = neuron_class()

        logger.info('%s neuron simulation in free conditions', neuron.name)

        _, y, _ = solver.run(neuron, Astim=0.0, tstim=20.0, toffset=0.0)
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

    # Initialize solver
    solver = SolverElec()
    Arange = (0.0, 2 * TITRATION_ESTIM_A_MAX)  # mA/m2

    # Stimulation parameters
    tstim = 100e-3  # s
    toffset = 50e-3  # s

    # Reference values
    Athr_refs = {'FS': 6.91, 'LTS': 1.54, 'RS': 5.03, 'RE': 3.61, 'TC': 4.05,
                 'LeechT': 4.66, 'LeechP': 13.72}
    latency_refs = {'FS': 101.00e-3, 'LTS': 128.56e-3, 'RS': 103.81e-3, 'RE': 148.50e-3,
                    'TC': 63.46e-3, 'LeechT': 21.32e-3, 'LeechP': 36.84e-3}

    for neuron_class in neurons:

        # Perform titration for each neuron
        neuron = neuron_class()
        logger.info('%s neuron titration', neuron.name)
        (Athr, t, y, _, latency) = titrateEStim(solver, neuron, Arange, tstim, toffset,
                                                PRF=None, DC=1.0)
        Vm = y[0, :]

        # Check that final number of spikes is 1
        n_spikes, _, _ = detectSpikes(t, Vm, SPIKE_MIN_VAMP, SPIKE_MIN_DT)
        assert n_spikes == 1, 'Number of spikes after titration should be exactly 1'

        # Check threshold amplitude
        Athr_diff = Athr - Athr_refs[neuron.name]
        assert np.abs(Athr_diff) < 0.1, Athr_err_msg.format(neuron.name, Athr_diff)

        # Check response latency
        lat_diff = (latency - latency_refs[neuron.name]) * 1e3
        assert np.abs(lat_diff) < 1.0, latency_err_msg.format(neuron.name, lat_diff)

    logger.info('Passed test: E-STIM titration')


def test_ASTIM():
    ''' Threshold A-STIM amplitude and needed to obtain an action potential and response latency
        should match reference values. '''

    Athr_err_msg = ('{} neuron threshold amplitude for excitation does not match reference value'
                    '(gap = {:.2f} kPa)')
    latency_err_msg = ('{} neuron latency for excitation at threshold amplitude does not match '
                       'reference value (gap = {:.2f} ms)')

    logger.info('Starting test: A-STIM titration')

    # Sonophore diameter
    a = 32e-9  # m

    # Stimulation parameters
    Fdrive = 350e3  # Hz
    tstim = 50e-3  # s
    toffset = 30e-3  # s

    Arange = (0.0, 2 * TITRATION_ASTIM_A_MAX)  # Pa

    # Reference values
    Athr_refs = {'FS': 38.66e3, 'LTS': 24.90e3, 'RS': 50.90e3, 'RE': 46.36e3, 'TC': 23.29e3,
                 'LeechT': 21.39e3, 'LeechP': 22.56e3}
    latency_refs = {'FS': 69.58e-3, 'LTS': 57.56e-3, 'RS': 71.59e-3, 'RE': 79.20e-3,
                    'TC': 63.67e-3, 'LeechT': 25.45e-3, 'LeechP': 54.76e-3}

    # Titration for each neuron
    for neuron_class in neurons:

        # Initialize neuron
        neuron = neuron_class()
        logger.info('%s neuron titration', neuron.name)

        # Initialize solver
        solver = SolverUS(a, neuron, Fdrive)

        # Perform titration
        (Athr, t, y, _, latency) = titrateAStim(solver, neuron, Fdrive, Arange, tstim, toffset,
                                                PRF=None, DC=1.0)
        Qm = y[2]

        # Check that final number of spikes is 1
        n_spikes, _, _ = detectSpikes(t, Qm, SPIKE_MIN_QAMP, SPIKE_MIN_DT)
        assert n_spikes == 1, 'Number of spikes after titration should be exactly 1'

        # Check threshold amplitude
        Athr_diff = (Athr - Athr_refs[neuron.name]) * 1e-3
        assert np.abs(Athr_diff) < 0.1, Athr_err_msg.format(neuron.name, Athr_diff)
        # Check response latency
        lat_diff = (latency - latency_refs[neuron.name]) * 1e3
        assert np.abs(lat_diff) < 1.0, latency_err_msg.format(neuron.name, lat_diff)


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
    if args.testset == 'all':
        test_all()
    else:
        possibles = globals().copy()
        possibles.update(locals())
        method = possibles.get('test_{}'.format(args.testset))
        method()
    sys.exit(0)


if __name__ == '__main__':
    main()
