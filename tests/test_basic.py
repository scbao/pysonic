#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-14 18:37:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-09-02 18:34:49

''' Test the basic functionalities of the package. '''

import sys

from PointNICE.utils import load_BLS_params, getNeuronsDict
from PointNICE import BilayerSonophore, SolverElec, SolverUS
from PointNICE.channels import *


# List of implemented neurons
neurons = getNeuronsDict()
del neurons['LeechT']
neurons = list(neurons.values())


def test_MECH():
    ''' Mechanical simulation. '''

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


def test_ESTIM():
    ''' Electrical simulation '''

    # Initialize neuron
    neuron = CorticalRS()

    # Initialize solver
    solver = SolverElec()

    # Stimulation parameters
    Astim = 10.0  # mA/m2
    tstim = 100e-3  # s
    toffset = 50e-3  # s

    # Run simulation
    solver.run(neuron, Astim, tstim, toffset, PRF=None, DF=1.0)


def test_ASTIM_effective():
    ''' Effective acoustic simulation '''

    # BLS geometry and parameters
    geom = {"a": 32e-9, "d": 0.0e-6}
    params = load_BLS_params()

    # Initialize neuron
    neuron = CorticalRS()

    # Stimulation parameters
    Fdrive = 350e3  # Hz
    Adrive = 100e3  # Pa
    tstim = 50e-3  # s
    toffset = 30e-3  # s

    solver = SolverUS(geom, params, neuron, Fdrive)
    solver.run(neuron, Fdrive, Adrive, tstim, toffset, PRF=None, DF=1.0, sim_type='effective')


def test_ASTIM_classic():
    ''' Classic acoustic simulation '''

    # BLS geometry and parameters
    geom = {"a": 32e-9, "d": 0.0e-6}
    params = load_BLS_params()

    # Initialize neuron
    neuron = CorticalRS()

    # Stimulation parameters
    Fdrive = 350e3  # Hz
    Adrive = 100e3  # Pa
    tstim = 1e-6  # s
    toffset = 0e-3  # s

    solver = SolverUS(geom, params, neuron, Fdrive)
    solver.run(neuron, Fdrive, Adrive, tstim, toffset, PRF=None, DF=1.0, sim_type='classic')


def test_ASTIM_hybrid():
    ''' Hybrid acoustic simulation '''

    # BLS geometry and parameters
    geom = {"a": 32e-9, "d": 0.0e-6}
    params = load_BLS_params()

    # Initialize neuron
    neuron = CorticalRS()

    # Stimulation parameters
    Fdrive = 350e3  # Hz
    Adrive = 100e3  # Pa
    tstim = 1e-3  # s
    toffset = 1e-3  # s

    solver = SolverUS(geom, params, neuron, Fdrive)
    solver.run(neuron, Fdrive, Adrive, tstim, toffset, PRF=None, DF=1.0, sim_type='hybrid')


def test_all():
    test_MECH()
    test_ESTIM()
    test_ASTIM_effective()
    test_ASTIM_classic()
    test_ASTIM_hybrid()


if __name__ == '__main__':

    valid_args = [
        'MECH',
        'ESTIM',
        'ASTIM_effective',
        'ASTIM_classic',
        'ASTIM_hybrid',
        'all'
    ]

    if len(sys.argv) > 2:
        print('tests script can only accept 1 command line argument')
        sys.exit()
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        if arg not in valid_args:
            print('tests script valid command line arguments are: ', str(valid_args))
            sys.exit()
        possibles = globals().copy()
        possibles.update(locals())
        method = possibles.get('test_{}'.format(arg))
        method()
    if len(sys.argv) == 1:
        test_all()
