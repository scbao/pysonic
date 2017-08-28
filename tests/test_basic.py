#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-14 18:37:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-28 14:18:56

''' Test the basic functionalities of the package. '''

import logging
import inspect
import numpy as np

from PointNICE.utils import load_BLS_params
from PointNICE import BilayerSonophore, channels, SolverElec, SolverUS
from PointNICE.solvers import detectSpikes, titrateEStim, titrateAStim
from PointNICE.constants import *

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# List of implemented neurons
neurons = []
for _, obj in inspect.getmembers(channels):
    if inspect.isclass(obj) and isinstance(obj.name, str):
        if obj.name != 'LeechT':
            neurons.append(obj)


def test_MECH():
    ''' Maximal negative and positive deflections of the BLS structure for a specific
        sonophore size, resting membrane properties and stimulation parameters. '''

    logger.info('Starting test: Mechanical simulation')

    # BLS geometry and parameters
    geom = {"a": 32e-9, "d": 0.0e-6}
    params = load_BLS_params()

    # Create BLS instance
    Fdrive = 350e3  # Hz
    Cm0 = 1e-2  # membrane resting capacitance (F/m2)
    Qm0 = -80e-5  # membrane resting charge density (C/m2)
    bls = BilayerSonophore(geom, params, Fdrive, Cm0, Qm0)

    # Run mechanical simulation
    Adrive = 100e3  # Pa
    Qm = 50e-5  # C/m2
    bls.runMech(Fdrive, Adrive, Qm)
    _, y, _ = bls.runMech(Fdrive, Adrive, Qm)

    # Check validity of deflection extrema
    Z, _ = y
    Zlast = Z[-NPC_FULL:]
    Zmin, Zmax = (Zlast.min(), Zlast.max())
    logger.info('Zmin = %.2f nm, Zmax = %.2f nm', Zmin * 1e9, Zmax * 1e9)
    Zmin_ref, Zmax_ref = (-0.116e-9, 5.741e-9)
    assert np.abs(Zmin - Zmin_ref) < 1e-12, 'Unexpected BLS compression amplitude'
    assert np.abs(Zmax - Zmax_ref) < 1e-12, 'Unexpected BLS expansion amplitude'

    logger.info('Passed test: Mechanical simulation')


def test_resting_potential():
    ''' Neurons membrane potential in free conditions should stabilize to their
        specified resting potential value. '''

    conv_err_msg = ('{} neuron membrane potential in free conditions does not converge to '
                    'stable value (gap after 20s: {:.2e} mV)')
    value_err_msg = ('{} neuron steady-state membrane potential in free conditions differs '
                     'significantly from specified resting potential (gap = {:.2f} mV)')

    logger.info('Starting test: neurons in free conditions')

    # Initialize solver
    solver = SolverElec()
    for neuron_class in neurons:

        # Simulate each neuron in free conditions
        neuron = neuron_class()
        _, y, _ = solver.run(neuron, Astim=0.0, tstim=20.0, toffset=0.0)
        Vm_free, *_ = y

        # Check membrane potential convergence
        Vm_free_last, Vm_free_beforelast = (Vm_free[-1], Vm_free[-2])
        Vm_free_conv = Vm_free_last - Vm_free_beforelast
        assert np.abs(Vm_free_conv) < 1e-5, conv_err_msg.format(neuron.name, Vm_free_conv)

        # Check membrane potential convergence to resting potential
        Vm_free_diff = Vm_free_last - neuron.Vm0
        assert np.abs(Vm_free_diff) < 0.1, value_err_msg.format(neuron.name, Vm_free_diff)
        logger.info('Passed test: neurons in free conditions')


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
    Athr_refs = {'FS': 6.91, 'LTS': 1.54, 'RS': 5.03, 'LeechT': 5.54, 'RE': 3.61, 'TC': 4.05}
    latency_refs = {'FS': 101.00e-3, 'LTS': 128.56e-3, 'RS': 103.81e-3, 'LeechT': 20.22e-3,
                    'RE': 148.50e-3, 'TC': 63.46e-3}

    for neuron_class in neurons:

        # Perform titration for each neuron
        neuron = neuron_class()
        print(neuron.name, 'neuron titration')
        (Athr, t, y, _, latency) = titrateEStim(solver, neuron, Arange, tstim, toffset,
                                                PRF=None, DF=1.0)
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

    # BLS geometry and parameters
    geom = {"a": 32e-9, "d": 0.0e-6}
    params = load_BLS_params()

    # Stimulation parameters
    Fdrive = 350e3  # Hz
    tstim = 50e-3  # s
    toffset = 30e-3  # s


    Arange = (0.0, 2 * TITRATION_ASTIM_A_MAX)  # Pa

    # Reference values
    Athr_refs = {'FS': 38.67e3, 'LTS': 24.80e3, 'RS': 51.17e3, 'RE': 46.36e3, 'TC': 23.24e3,
                 'LeechT': None}
    latency_refs = {'FS': 63.72e-3, 'LTS': 61.92e-3, 'RS': 62.52e-3, 'RE': 79.20e-3, 'TC': 68.53e-3,
                    'LeechT': None}

    # Titration for each neuron
    for neuron_class in neurons:

        # Initialize neuron
        neuron = neuron_class()
        print(neuron.name, 'neuron titration')

        # Initialize solver
        solver = SolverUS(geom, params, neuron, Fdrive)

        # Perform titration
        (Athr, t, y, _, latency) = titrateAStim(solver, neuron, Fdrive, Arange, tstim, toffset,
                                                PRF=None, DF=1.0)
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


def test_ASTIM_classic():
    ''' Short classic A-STIM simulation should not raise any errors. '''

    logger.info('Starting test: A-STIM classic simulation')

    # BLS geometry and parameters
    geom = {"a": 32e-9, "d": 0.0e-6}
    params = load_BLS_params()

    # Stimulation parameters
    Fdrive = 350e3  # Hz
    Adrive = 100e3  # Pa
    tstim = 1e-3  # s
    toffset = 1e-3  # s

    # Initialize RS neuron
    rs_neuron = channels.CorticalRS()

    # Initialize solver
    solver = SolverUS(geom, params, rs_neuron, Fdrive)

    # Run short classic simulation of the system
    solver.run(rs_neuron, Fdrive, Adrive, tstim, toffset, sim_type='classic')

    # If no error is raised, test is passed successfully
    logger.info('Passed test: A-STIM classic simulation')


def test_ASTIM_hybrid():
    ''' Short hybrid A-STIM simulation should not raise any errors. '''

    logger.info('Starting test: A-STIM hybrid simulation')

    # BLS geometry and parameters
    geom = {"a": 32e-9, "d": 0.0e-6}
    params = load_BLS_params()

    # Stimulation parameters
    Fdrive = 350e3  # Hz
    Adrive = 100e3  # Pa
    tstim = 10e-3  # s
    toffset = 1e-3  # s

    # Initialize RS neuron
    rs_neuron = channels.CorticalRS()

    # Initialize solver
    solver = SolverUS(geom, params, rs_neuron, Fdrive)

    # Run short classic simulation of the system
    solver.run(rs_neuron, Fdrive, Adrive, tstim, toffset, sim_type='hybrid')

    # If no error is raised, test is passed successfully
    logger.info('Passed test: A-STIM hybrid simulation')


if __name__ == '__main__':
    logger.info('Starting tests')

    test_MECH()
    test_resting_potential()
    test_ESTIM()
    test_ASTIM()
    test_ASTIM_classic()
    test_ASTIM_hybrid()

    logger.info('All tests successfully passed')
