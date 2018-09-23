# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-23 14:32:39

''' Run E-STIM simulations of a specific point-neuron. '''

import os
import logging
import numpy as np
from argparse import ArgumentParser

from PySONIC.utils import logger, checkBatchLog, selectDirDialog
from PySONIC.neurons import *
from PySONIC.batches import createSimQueue, runBatch
from PySONIC.plt import plotBatch

# Default parameters
defaults = dict(
    neuron='RS',
    amps=[10.0],  # mA/m2
    durations=[100.0],  # ms
    PRFs=[100.0],  # Hz
    DCs=[100.0],  # %
    offsets=[50.],  # ms
    method='sonic'
)


def runEStimBatch(outdir, logpath, neuron, stim_params, mpi=False):
    ''' Run batch E-STIM simulations of the system for various neuron types and
        stimulation parameters.

        :param outdir: full path to output directory
        :param logpath: full path log file
        :param stim_params: dictionary containing sweeps for all stimulation parameters
        :param mpi: boolean statting wether or not to use multiprocessing
        :return: list of full paths to the output files
    '''
    mandatory_params = ['durations', 'offsets', 'PRFs', 'DCs']
    for mparam in mandatory_params:
        if mparam not in stim_params:
            raise ValueError('Missing stimulation parameter field: "{}"'.format(mparam))

    logger.info("Starting E-STIM simulation batch")

    # Generate simulations queue
    queue = createSimQueue(stim_params.get('amps', [None]), stim_params['durations'],
                           stim_params['offsets'], stim_params['PRFs'], stim_params['DCs'])

    # Run batch
    return runBatch(neuron, 'runAndSave', queue, extra_params=[outdir, logpath], mpi=mpi)


if __name__ == '__main__':

    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('--mpi', default=False, action='store_true', help='Use multiprocessing')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-p', '--plot', default=False, action='store_true', help='Plot results')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')
    ap.add_argument('-t', '--titrate', default=False, action='store_true', help='Perform titration')

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-A', '--amps', nargs='+', type=float, help='Injected current density (mA/m2)')
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
    neuron_str = args['neuron']
    stim_params = dict(
        amps=np.array(args.get('amps', defaults['amps'])),  # mA/m2
        durations=np.array(args.get('durations', defaults['durations'])) * 1e-3,  # s
        PRFs=np.array(args.get('PRFs', defaults['PRFs'])),  # Hz
        DCs=np.array(args.get('DCs', defaults['DCs'])) * 1e-2,  # (-)
        offsets=np.array(args.get('offsets', defaults['offsets'])) * 1e-3  # s
    )
    if titrate:
        stim_params['amps'] = [None]

    # Run E-STIM batch
    logpath, _ = checkBatchLog(outdir, 'E-STIM')
    if neuron_str not in getNeuronsDict():
        raise ValueError('Unknown neuron type: "{}"'.format(neuron_str))
    neuron = getNeuronsDict()[neuron_str]()
    pkl_filepaths = runEStimBatch(outdir, logpath, neuron, stim_params, mpi=mpi)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    if plot:
        yvars = {'V_m': ['Vm']}
        plotBatch(pkl_dir, pkl_filepaths, yvars)
