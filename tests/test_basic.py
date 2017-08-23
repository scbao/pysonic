#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-06-14 18:37:45
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-23 16:22:41

''' Test the basic functionalities of the package. '''

import logging
import numpy as np
import PointNICE
from PointNICE.utils import LoadParams, detectSpikes
from PointNICE.channels import CorticalRS
from PointNICE.constants import *

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# Set geometry of NBLS structure
geom = {"a": 32e-9, "d": 0.0e-6}

# Defining general stimulation parameters
Fdrive = 3.5e5  # Hz
Adrive = 1e5  # Pa
PRF = 1.5e3  # Hz
DF = 1

logger.info('Starting basic tests')

logger.info('Test 1: Loading parameters')
params = LoadParams()

# logger.info('Test 2: Creating typical BLS instance')
# Cm0 = 1e-2  # membrane resting capacitance (F/m2)
# Qm0 = -89e-5  # membrane resting charge density (C/m2)

# bls = PointNICE.BilayerSonophore(geom, params, Fdrive, Cm0, Qm0)

# logger.info('Test 3: Running simulation of the mechanical system')
# charges = np.linspace(-100, 50, 10) * 1e-5  # C/m2
# for Qm in charges:
#     bls.runMech(Fdrive, 2e4, Qm)


logger.info('Test 4: Creating channel mechanism')
rs_mech = CorticalRS()


logger.info('Test 5: Creating typical SolverUS instance')
solver = PointNICE.SolverUS(geom, params, rs_mech, Fdrive)


# logger.info('Test 6: running short classic simulation of the full system')
# tstim = 1e-3  # s
# toffset = 1e-3  # s
# (t, y, _) = solver.runSim(rs_mech, Fdrive, Adrive, tstim, toffset, PRF, DF, 'classic')
# Qm = y[2]
# n_spikes, _, _ = detectSpikes(t, Qm, SPIKE_MIN_QAMP, SPIKE_MIN_DT)
# assert n_spikes == 0, 'Error: number of spikes should be 0'
# logger.info('0 spike detected --> OK')


# logger.info('Test 7: running hybrid simulation')
# tstim = 30e-3  # s
# toffset = 10e-3  # s
# (t, y, _) = solver.runSim(rs_mech, Fdrive, Adrive, tstim, toffset, PRF, DF, 'hybrid')
# Qm = y[2]
# n_spikes, _, _ = detectSpikes(t, Qm, SPIKE_MIN_QAMP, SPIKE_MIN_DT)
# assert n_spikes == 1, 'Error: number of spikes should be 1'
# logger.info('1 spike detected --> OK')


# logger.info('Test 8: creating dummy lookup file')
# amps = np.array([1, 2]) * 1e5  # Pa
# charges = np.array([-80.0, 30.0]) * 1e-5  # C/m2
# tmp = rs_mech.name
# rs_mech.name = 'test'
# solver.createLookup(rs_mech, Fdrive, amps, charges)
# rs_mech.name = tmp


# logger.info('Test 9: running effective simulation')
# tstim = 30e-3  # s
# toffset = 10e-3  # s
# (t, y, _) = solver.runSim(rs_mech, Fdrive, Adrive, tstim, toffset, PRF, DF, 'effective')
# Qm = y[2]
# n_spikes, _, _ = detectSpikes(t, Qm, SPIKE_MIN_QAMP, SPIKE_MIN_DT)
# assert n_spikes == 1, 'Error: number of spikes should be 1'
# logger.info('1 spike detected --> OK')


logger.info('Test 10: running effective amplitude titration')
tstim = 30e-3  # s
toffset = 10e-3  # s
Arange = (0.0, 2 * TITRATION_AMAX)  # Pa
(Athr, t, y, _, latency) = solver.titrateAmp(rs_mech, Fdrive, Arange, tstim, toffset,
                                             PRF, DF, 'effective')
Qm = y[2]
n_spikes, _, _ = detectSpikes(t, Qm, SPIKE_MIN_QAMP, SPIKE_MIN_DT)
assert n_spikes == 1, 'Error: number of spikes should be 1'
logger.info('1 spike detected --> OK')


# logger.info('Test 11: running effective duration titration')
# trange = (0.0, 2 * TITRATION_TMAX)  # s
# toffset = 10e-3  # s
# (tthr, t, y, _, latency) = solver.titrateDur(rs_mech, Fdrive, Adrive, trange, toffset,
#                                              PRF, DF, 'effective')
# Qm = y[2]
# n_spikes, _, _ = detectSpikes(t, Qm, SPIKE_MIN_QAMP, SPIKE_MIN_DT)
# assert n_spikes == 1, 'Error: number of spikes should be 1'
# logger.info('1 spike detected --> OK')


logger.info('All tests successfully completed')
