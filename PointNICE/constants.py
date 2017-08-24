#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-04 13:23:31
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-24 13:41:31

''' Algorithmic constants used in the core modules. '''

# Fitting and pre-processing
LJFIT_PM_MAX = 1e8  # intermolecular pressure at the deflection lower bound for LJ fitting (Pa)
PNET_EQ_MAX = 1e-1  # error threshold for net pressure at computed equilibrium position (Pa)
PMAVG_STD_ERR_MAX = 1500  # error threshold in nonlinear fit of molecular pressure (Pa)



# Generic integration constants
NPC_FULL = 1000  # nb of samples per acoustic period in full system
SOLVER_NSTEPS = 1000  # maximum number of steps allowed during one call to the LSODA/DOP853 solvers
CLASSIC_DS_FACTOR = 3  # time downsampling factor applied to output arrays of classic simulations

# Effective integration
DT_EFF = 5e-5  # time step for effective integration (s)
# DT_EFF = 1e-6  # time step for effective integration (s)

# Mechanical simulations
Z_ERR_MAX = 1e-11  # periodic convergence threshold for deflection gas content (m)
NG_ERR_MAX = 1e-24  # periodic convergence threshold for gas content (mol)

# Hybrid integration
NPC_HH = 40  # nb of samples per acoustic period in HH system
DQ_UPDATE = 1e-5  # charge evolution threshold between two hybrid integrations (C/m2)
DT_UPDATE = 5e-4  # time interval between two hybrid integrations (s)

# Titrations
TITRATION_AMAX = 2e5  # initial acoustic pressure upper bound for titration procedure (Pa)
TITRATION_TMAX = 2e-1  # initial stimulus duration upper bound for titration procedure (Pa)
TITRATION_DA_THR = 1e3  # acoustic pressure search range threshold for titration procedure (Pa)
TITRATION_DT_THR = 1e-3  # stimulus duration search range threshold for titration procedure (s)

# Spike detection
SPIKE_MIN_QAMP = 10e-5  # threshold amplitude for spike detection on charge signal (C/m2)
SPIKE_MIN_VAMP = 10.0  # threshold amplitude for spike detection on potential signal (mV)
SPIKE_MIN_DT = 1e-3  # minimal time interval for spike detection on charge signal (s)
