# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-11-30 10:46:34

''' Run E-STIM simulations of a specific point-neuron. '''

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.utils import logger, selectDirDialog
from PySONIC.neurons import *
from PySONIC.batches import createEStimQueue, runBatch
from PySONIC.plt import plotBatch

# Default parameters
defaults = dict(
    neuron='RS',
    amp=[10.0],  # mA/m2
    duration=[100.0],  # ms
    PRF=[100.0],  # Hz
    DC=[100.0],  # %
    offset=[50.],  # ms
    method='sonic'
)


def runEStimBatch(outdir, neuron, stim_params, mpi=False):
    ''' Run batch E-STIM simulations of the system for various neuron types and
        stimulation parameters.

        :param outdir: full path to output directory
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
    queue = createEStimQueue(
        stim_params.get('amps', [None]),
        stim_params['durations'],
        stim_params['offsets'],
        stim_params['PRFs'],
        stim_params['DCs']
    )

    # Run batch
    return runBatch(neuron, 'runAndSave', queue, extra_params=[outdir], mpi=mpi)


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('--mpi', default=False, action='store_true', help='Use multiprocessing')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-p', '--plotVm', default=False, action='store_true', help='Plot Vm')
    ap.add_argument('--plotall', default=False, action='store_true', help='Plot all variables')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')
    ap.add_argument('-t', '--titrate', default=False, action='store_true', help='Perform titration')

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-A', '--amp', nargs='+', type=float, help='Injected current density (mA/m2)')
    ap.add_argument('-d', '--duration', nargs='+', type=float, help='Stimulus duration (ms)')
    ap.add_argument('--offset', nargs='+', type=float, help='Offset duration (ms)')
    ap.add_argument('--PRF', nargs='+', type=float, help='PRF (Hz)')
    ap.add_argument('--DC', nargs='+', type=float, help='Duty cycle (%%)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    loglevel = logging.DEBUG if args['verbose'] is True else logging.INFO
    logger.setLevel(loglevel)
    outdir = args['outputdir'] if 'outputdir' in args else selectDirDialog()
    plot = True if (args['plotVm'] or args['plotall']) else False
    titrate = args['titrate']
    neuron_str = args['neuron']
    stim_params = dict(
        amps=np.array(args.get('amp', defaults['amp'])),  # mA/m2
        durations=np.array(args.get('duration', defaults['duration'])) * 1e-3,  # s
        PRFs=np.array(args.get('PRF', defaults['PRF'])),  # Hz
        DCs=np.array(args.get('DC', defaults['DC'])) * 1e-2,  # (-)
        offsets=np.array(args.get('offset', defaults['offset'])) * 1e-3  # s
    )
    if titrate:
        stim_params['amps'] = [None]

    # Run E-STIM batch
    if neuron_str not in getNeuronsDict():
        logger.error('Unknown neuron type: "%s"', neuron_str)
        return
    neuron = getNeuronsDict()[neuron_str]()
    pkl_filepaths = runEStimBatch(outdir, neuron, stim_params, mpi=args['mpi'])
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    if plot:
        if args['plotVm']:
            vars_dict = {'V_m': ['Vm']}
        elif args['plotall']:
            vars_dict = None
        plotBatch(pkl_filepaths, vars_dict=vars_dict)
        plt.show()


if __name__ == '__main__':
    main()
