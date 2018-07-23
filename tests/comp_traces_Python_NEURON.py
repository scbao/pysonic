#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-07-23 11:35:07

""" Compare output of pure-Python and NEURON acoustic simulations. """

import logging
import pickle
import matplotlib.pyplot as plt

from PointNICE.utils import logger, getNeuronsDict, si_format
from PointNICE.solvers import SolverUS, checkBatchLog, AStimWorker, setBatchDir
from PointNICE.plt import plotComp

# Set logging level
logger.setLevel(logging.INFO)

# Set neurons
neurons = ['RS', 'FS', 'LTS', 'RE', 'TC']
# neurons = ['LTS']

# Set stimulation parameters
a = 32e-9  # m
Adrive = 100e3  # Pa
Fdrive = 500e3  # Hz
PRF = 100.  # Hz
DC = .5
tstim = 150e-3  # s
toffset = 100e-3  # s

ASTIM_CW_code = 'ASTIM_{}_CW_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms_{}'
ASTIM_PW_code = 'ASTIM_{}_PW_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms_PRF{:.2f}Hz_DC{:.2f}%_{}'
suffix = 'Python_vs_NEURON'

# Select output directory
batch_dir = setBatchDir()
mu_acc_factor = 0

# For each neuron
for nname in neurons:

    if DC == 1:
        simcode = ASTIM_CW_code.format(nname, a * 1e9, Fdrive * 1e-3, Adrive * 1e-3, tstim * 1e3,
                                       suffix)
    else:
        simcode = ASTIM_PW_code.format(nname, a * 1e9, Fdrive * 1e-3, Adrive * 1e-3, tstim * 1e3,
                                       PRF, DC * 1e2, suffix)

    neuron = getNeuronsDict()[nname]()

    # Initialize solver
    solver = SolverUS(a, neuron, Fdrive)

    log_filepath, _ = checkBatchLog(batch_dir, 'A-STIM')

    # Run NEURON and Python A-STIM point-neuron simulations
    logger.info('Running simulations for %s neuron', nname)
    NEURONsim_file = AStimWorker(1, batch_dir, log_filepath, solver, neuron, Fdrive, Adrive,
                                 tstim, toffset, PRF, DC, 'NEURON', 2).__call__()
    effsim_file = AStimWorker(1, batch_dir, log_filepath, solver, neuron, Fdrive, Adrive,
                              tstim, toffset, PRF, DC, 'effective', 2).__call__()

    # Compare computation times
    with open(effsim_file, 'rb') as fh:
        tcomp_eff = pickle.load(fh)['meta']['tcomp']
    logger.info('Python simulation completed in %.1f s', tcomp_eff)
    with open(NEURONsim_file, 'rb') as fh:
        tcomp_NEURON = pickle.load(fh)['meta']['tcomp']
    logger.info('NEURON simulation completed in %.1f ms (%.1f times faster)',
                tcomp_NEURON * 1e3, tcomp_eff / tcomp_NEURON)
    mu_acc_factor += tcomp_eff / tcomp_NEURON

    # Plot resulting profiles
    filepaths = [effsim_file, NEURONsim_file]
    fig = plotComp('Qm', filepaths, labels=['Python', 'NEURON'], showfig=False)
    ax = fig.gca()
    ax.set_title('{} neuron - {}Hz, {}Pa'.format(nname, *si_format([Fdrive, Adrive], space=' ')))
    fig.tight_layout()
    # fig.savefig('{}/{}.png'.format(batch_dir, simcode))

mu_acc_factor /= len(neurons)
logger.info('average acceleration factor = %.1f', mu_acc_factor)

plt.show()
