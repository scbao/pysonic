#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-04 13:23:31
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-29 02:45:41

''' Algorithmic constants used in the package. '''

# Biophysical constants
FARADAY = 9.64853e4  # Faraday constant (C/mol)
Rg = 8.31342  # Universal gas constant (Pa.m^3.mol^-1.K^-1 or J.mol^-1.K^-1)
Z_Ca = 2  # Calcium valence
Z_Na = 1  # Sodium valence
Z_K = 1  # Potassium valence
CELSIUS_2_KELVIN = 273.15  # Celsius to Kelvin conversion constant

# Fitting and pre-processing
LJFIT_PM_MAX = 1e8  # intermolecular pressure at the deflection lower bound for LJ fitting (Pa)
PNET_EQ_MAX = 1e-1  # error threshold for net pressure at computed equilibrium position (Pa)
PMAVG_STD_ERR_MAX = 3000  # error threshold in nonlinear fit of molecular pressure (Pa)

# Mechanical simulations
MAX_RMSE_PTP_RATIO = 1e-4  # threshold RMSE / peak-to-peak ratio for periodic convergence
Z_ERR_MAX = 1e-11  # periodic convergence threshold for deflection (m)
NG_ERR_MAX = 1e-24  # periodic convergence threshold for gas content (mol)
NCYCLES_MAX = 10  # max number of acoustic cycles in mechanical simulations
CHARGE_RANGE = (-200e-5, 150e-5)  # physiological charge range constraining the membrane (C/m2)

# E-STIM simulations
DT_ESTIM = 5e-5

# A-STIM simulations
SOLVER_NSTEPS = 1000  # maximum number of steps allowed during one call to the LSODA/DOP853 solvers
CLASSIC_TARGET_DT = 1e-8  # target temporal resolution for output arrays of classic simulations
NPC_FULL = 1000  # nb of samples per acoustic period in full system
NPC_HH = 40  # nb of samples per acoustic period in HH system
DQ_UPDATE = 1e-5  # charge evolution threshold between two hybrid integrations (C/m2)
DT_UPDATE = 5e-4  # time interval between two hybrid integrations (s)
DT_EFF = 5e-5  # time step for effective integration (s)
MIN_SAMPLES_PER_PULSE_INT = 1  # minimal number of time points per pulse interval (TON of TOFF)

# Spike detection
SPIKE_MIN_QAMP = 5e-5  # threshold amplitude for spike detection on charge signal (C/m2)
SPIKE_MIN_QPROM = 20e-5  # threshold prominence for spike detection on charge signal (C/m2)
SPIKE_MIN_VAMP = 10.0  # threshold amplitude for spike detection on potential signal (mV)
SPIKE_MIN_VPROM = 20.0  # threshold prominence for spike detection on potential signal (mV)
SPIKE_MIN_DT = 5e-4  # minimal time interval for spike detection on charge signal (s)
MIN_NSPIKES_SPECTRUM = 3  # minimum number of spikes to compute firing rate spectrum

# Titrations
TITRATION_T_OFFSET = 200e-3  # offset period for titration procedures (s)
TITRATION_ASTIM_DA_MAX = 1e2  # acoustic pressure search range threshold for titration (Pa)
TITRATION_ESTIM_A_MAX = 50.0  # initial current density upper bound for titration (mA/m2)
TITRATION_ESTIM_DA_MAX = 0.1  # current density search range threshold for titration (mA/m2)

# QSS stability analysis
QSS_REL_OFFSET = .05
QSS_HISTORY_INTERVAL = 30e-3  # recent history interval (s)
QSS_INTEGRATION_INTERVAL = 1e-3  # iterative integration interval (s)
QSS_MAX_INTEGRATION_DURATION = 1000e-3  # iterative integration interval (s)
QSS_Q_CONV_THR = 1e-7  # max. charge deviation to infer convergence (C/m2)
QSS_Q_DIV_THR = 1e-4  # min. charge deviation to infer divergence (C/m2)
TMIN_STABILIZATION = 500e-3  # time window for stabilization analysis (s)
