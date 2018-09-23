#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 18:16:09
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-23 15:27:47

''' Run A-STIM simulations of a specific point-neuron. '''

import os
import logging
import numpy as np
from argparse import ArgumentParser

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import logger, checkBatchLog, selectDirDialog
from PySONIC.neurons import getNeuronsDict
from PySONIC.batches import createSimQueue, runBatch
from PySONIC.plt import plotBatch

# Default parameters
defaults = dict(
    neuron='RS',
    diams=[32.0],  # nm
    freqs=[500.0],  # kHz
    amps=[100.0],  # kPa
    durations=[100.0],  # ms
    PRFs=[100.0],  # Hz
    DCs=[100.0],  # %
    offsets=[50.],  # ms
    method='sonic'
)


def runAStimBatch(outdir, logpath, nbls, stim_params, method, mpi=False):
    ''' Run batch A-STIM simulations of the system for various neuron types and
        stimulation parameters.

        :param outdir: full path to output directory
        :param logpath: full path log file
        :param stim_params: dictionary containing sweeps for all stimulation parameters
        :param method: numerical integration method ("classic", "hybrid" or "sonic")
        :param mpi: boolean statting wether or not to use multiprocessing
        :return: list of full paths to the output files
    '''

    mandatory_params = ['freqs', 'durations', 'offsets', 'PRFs', 'DCs']
    for mparam in mandatory_params:
        if mparam not in stim_params:
            raise ValueError('Missing stimulation parameter field: "{}"'.format(mparam))

    logger.info("Starting A-STIM simulation batch")

    # Generate queue
    nofreq_queue = createSimQueue(stim_params.get('amps', [None]), stim_params['durations'],
                                  stim_params['offsets'], stim_params['PRFs'], stim_params['DCs'])

    # Repeat queue for each US frequency
    nofreq_queue = np.array(nofreq_queue)
    freqs = stim_params['freqs']
    nf = len(freqs)
    nqueue = nofreq_queue.shape[0]
    queue = np.tile(nofreq_queue, (nf, 1))
    freqs_col = np.vstack([np.ones(nqueue) * f for f in freqs]).reshape(nf * nqueue, 1)
    queue = np.hstack((freqs_col, queue)).tolist()

    # Add method to queue items
    for item in queue:
        item.append(method)

    # Run batch
    return runBatch(nbls, 'runAndSave', queue, extra_params=[outdir, logpath], mpi=mpi)


if __name__ == '__main__':

    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('--mpi', default=False, action='store_true', help='Use multiprocessing')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-p', '--plot', default=False, action='store_true', help='Plot results')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')
    ap.add_argument('-t', '--titrate', default=False, action='store_true', help='Perform titration')
    ap.add_argument('-m', '--method', type=str, default=defaults['method'],
                    help='Numerical integration method ("classic", "hybrid" or "sonic")')

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-a', '--diams', nargs='+', type=float, help='Sonophore diameter (nm)')
    ap.add_argument('-f', '--freqs', nargs='+', type=float, help='US frequency (kHz)')
    ap.add_argument('-A', '--amps', nargs='+', type=float, help='Acoustic pressure amplitude (kPa)')
    ap.add_argument('-d', '--durations', nargs='+', type=float, help='Stimulus duration (ms)')
    ap.add_argument('--offset', nargs='+', type=float, help='Offset duration (ms)')
    ap.add_argument('--PRF', nargs='+', type=float, help='PRF (Hz)')
    ap.add_argument('--DC', nargs='+', type=float, help='Duty cycle (%%)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    loglevel = logging.DEBUG if args['verbose'] is True else logging.INFO
    logger.setLevel(loglevel)
    outdir = args['outputdir'] if 'outputdir' in args else selectDirDialog()
    mpi = args['mpi']
    plot = args['plot']
    titrate = args['titrate']
    method = args['method']
    neuron_str = args['neuron']
    diams = np.array(args.get('diams', defaults['diams'])) * 1e-9  # m
    stim_params = dict(
        freqs=np.array(args.get('freqs', defaults['freqs'])) * 1e3,  # Hz
        amps=np.array(args.get('amps', defaults['amps'])) * 1e3,  # Pa
        durations=np.array(args.get('durations', defaults['durations'])) * 1e-3,  # s
        PRFs=np.array(args.get('PRFs', defaults['PRFs'])),  # Hz
        DCs=np.array(args.get('DCs', defaults['DCs'])) * 1e-2,  # (-)
        offsets=np.array(args.get('offsets', defaults['offsets'])) * 1e-3  # s
    )
    if titrate:
        stim_params['amps'] = [None]

    # Run A-STIM batch
    logpath, _ = checkBatchLog(outdir, 'A-STIM')
    if neuron_str not in getNeuronsDict():
        raise ValueError('Unknown neuron type: "{}"'.format(neuron_str))
    neuron = getNeuronsDict()[neuron_str]()
    pkl_filepaths = []
    for a in diams:
        nbls = NeuronalBilayerSonophore(a, neuron)
        pkl_filepaths += runAStimBatch(outdir, logpath, nbls, stim_params, method, mpi=mpi)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    if plot:
        yvars = {'Q_m': ['Qm']}
        plotBatch(pkl_dir, pkl_filepaths, yvars)
