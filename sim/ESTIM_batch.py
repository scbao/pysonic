# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-24 18:16:37

""" Run batch electrical simulations of point-neuron models. """

import os
import logging
from PointNICE.solvers import checkBatchLog, runEStimBatch
from PointNICE.channels import *
from PointNICE.plt import plotBatch

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# Channels mechanisms
neurons = [LeechTouch()]

# Stimulation parameters
stim_params = {
    'amps': [20.0],  # mA/m2
    'durations': [0.5],  # s
    'PRFs': [1e2],  # Hz
    'DFs': [1.0]
}
stim_params['offsets'] = [1.0] * len(stim_params['durations'])  # s

# Select output directory
try:
    (batch_dir, log_filepath) = checkBatchLog('E-STIM')
except AssertionError as err:
    logger.error(err)
    quit()

# Run E-STIM batch
pkl_filepaths = runEStimBatch(batch_dir, log_filepath, neurons, stim_params)
pkl_dir, _ = os.path.split(pkl_filepaths[0])

vars_RS_FS = {
    'V_m': ['Vm'],
    'i_{Na}\ kin.': ['m', 'h'],
    'i_L\ kin.': ['n'],
    'i_M\ kin.': ['p'],
    'I': ['iNa', 'iK', 'iM', 'iL', 'iNet']
}

vars_LTS = {
    'V_m': ['Vm'],
    'i_{Na}\ kin.': ['m', 'h'],
    'i_K\ kin.': ['n'],
    'i_M\ kin.': ['p'],
    'i_T\ kin.': ['s', 'u'],
    'I': ['iNa', 'iK', 'iM', 'iT', 'iL', 'iNet']
}

vars_RE = {
    'V_m': ['Vm'],
    'i_{Na}\ kin.': ['m', 'h'],
    'i_K\ kin.': ['n'],
    'i_{TS}\ kin.': ['s', 'u'],
    'I': ['iNa', 'iK', 'iTs', 'iL', 'iNet']
}

vars_TC = {
    'V_m': ['Vm'],
    'i_{Na}\ kin.': ['m', 'h'],
    'i_K\ kin.': ['n'],
    'i_{T}\ kin.': ['s', 'u'],
    'i_{H}\ kin.': ['O', 'OL', 'O + 2OL'],
    'I': ['iNa', 'iK', 'iT', 'iH', 'iKL', 'iL', 'iNet']
}

vars_LeechT = {
    'V_m': ['Vm'],
    'i_{Na}\ kin.': ['m', 'h'],
    'i_K\ kin.': ['n'],
    'i_{Ca}\ kin.': ['s'],
    'pools': ['C_Na_arb', 'C_Na_arb_activation', 'C_Ca_arb', 'C_Ca_arb_activation'],
    'I': ['iNa', 'iK', 'iCa', 'iKCa', 'iPumpNa', 'iL', 'iNet']
}

plotBatch(vars_LeechT, pkl_dir, pkl_filepaths, lw=2)
