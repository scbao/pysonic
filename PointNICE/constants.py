#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-04 13:23:31
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-27 19:17:22

''' Algorithmic constants used in the package. '''


# Fitting and pre-processing
LJFIT_PM_MAX = 1e8  # intermolecular pressure at the deflection lower bound for LJ fitting (Pa)
PNET_EQ_MAX = 1e-1  # error threshold for net pressure at computed equilibrium position (Pa)
PMAVG_STD_ERR_MAX = 1500  # error threshold in nonlinear fit of molecular pressure (Pa)


# Mechanical simulations
Z_ERR_MAX = 1e-11  # periodic convergence threshold for deflection (m)
NG_ERR_MAX = 1e-24  # periodic convergence threshold for gas content (mol)
NCYCLES_MAX = 50  # max number of acoustic cycles in mechanical simulations
CHARGE_RANGE = (-120e-5, 70e-5)  # physiological charge range constraining the membrane (C/m2)

# E-STIM simulations
DT_ESTIM = 1e-4


# A-STIM simulations
SOLVER_NSTEPS = 1000  # maximum number of steps allowed during one call to the LSODA/DOP853 solvers
CLASSIC_TARGET_DT = 1e-8  # target temporal resolution for output arrays of classic simulations
NPC_FULL = 1000  # nb of samples per acoustic period in full system
NPC_HH = 40  # nb of samples per acoustic period in HH system
DQ_UPDATE = 1e-5  # charge evolution threshold between two hybrid integrations (C/m2)
DT_UPDATE = 5e-4  # time interval between two hybrid integrations (s)
DT_EFF = 5e-5  # time step for effective integration (s)


# Spike detection
SPIKE_MIN_QAMP = 10e-5  # threshold amplitude for spike detection on charge signal (C/m2)
SPIKE_MIN_QPROM = 2e-5  # threshold prominence for spike detection on charge signal (C/m2)
SPIKE_MIN_VAMP = 10.0  # threshold amplitude for spike detection on potential signal (mV)
SPIKE_MIN_DT = 1e-3  # minimal time interval for spike detection on charge signal (s)


# Titrations
TITRATION_T_OFFSET = 50e-3  # offset period for titration procedures (s)
TITRATION_ASTIM_A_MAX = 3e5  # initial acoustic pressure upper bound for titration (Pa)
TITRATION_ASTIM_DA_MAX = 1e3  # acoustic pressure search range threshold for titration (Pa)
TITRATION_ESTIM_A_MAX = 50.0  # initial current density upper bound for titration (mA/m2)
TITRATION_ESTIM_DA_MAX = 0.1  # current density search range threshold for titration (mA/m2)
TITRATION_T_MAX = 2e-1  # initial stimulus duration upper bound for titration (s)
TITRATION_DT_THR = 1e-3  # stimulus duration search range threshold for titration (s)
TITRATION_DDC_THR = 0.01  # stimulus duty cycle search range threshold for titration (-)
TITRATION_DC_MAX = 1.0  # initial stimulus duty cycle upper bound for titration (-)
