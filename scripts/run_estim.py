# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-02 15:49:00

''' Run E-STIM simulations of a specific point-neuron. '''

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.core import Batch
from PySONIC.utils import logger, selectDirDialog, parseElecAmps, getInDict
from PySONIC.neurons import *
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


def runEStimBatch(outdir, neuron, stim_params, mpi=False, loglevel=logging.INFO):
    ''' Run batch E-STIM simulations of the system for various neuron types and
        stimulation parameters.

        :param outdir: full path to output directory
        :param stim_params: dictionary containing sweeps for all stimulation parameters
        :param mpi: boolean statting wether or not to use multiprocessing
        :param loglevel: logging level
        :return: list of full paths to the output files
    '''
    mandatory_params = ['durations', 'offsets', 'PRFs', 'DCs']
    for mparam in mandatory_params:
        if mparam not in stim_params:
            raise ValueError('Missing stimulation parameter field: "{}"'.format(mparam))

    logger.info("Starting E-STIM simulation batch")

    # Generate simulations queue
    queue = neuron.simQueue(
        stim_params.get('amps', None),
        stim_params['durations'],
        stim_params['offsets'],
        stim_params['PRFs'],
        stim_params['DCs']
    )
    for item in queue:
        item.insert(0, outdir)

    # Run batch
    batch = Batch(neuron.runAndSave, queue)
    return batch(mpi=mpi, loglevel=loglevel)


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('--mpi', default=False, action='store_true', help='Use multiprocessing')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-p', '--plot', type=str, nargs='+', help='Variables to plot')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')
    ap.add_argument('-t', '--titrate', default=False, action='store_true', help='Perform titration')

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-A', '--amp', nargs='+', type=float, help='Injected current density (mA/m2)')
    ap.add_argument('--Arange', type=str, nargs='+',
                    help='Amplitude range [scale min max n] (mA/m2)')
    ap.add_argument('-d', '--duration', nargs='+', type=float, help='Stimulus duration (ms)')
    ap.add_argument('--offset', nargs='+', type=float, help='Offset duration (ms)')
    ap.add_argument('--PRF', nargs='+', type=float, help='PRF (Hz)')
    ap.add_argument('--DC', nargs='+', type=float, help='Duty cycle (%%)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    loglevel = logging.DEBUG if args['verbose'] is True else logging.INFO
    logger.setLevel(loglevel)
    outdir = getInDict(args, 'outputdir', selectDirDialog)
    if outdir == '':
        logger.error('No output directory selected')
        quit()
    titrate = args['titrate']
    neuron_str = args['neuron']

    try:
        amps = parseElecAmps(args, defaults)
    except ValueError as err:
        logger.error(err)
        quit()

    stim_params = dict(
        amps=amps,
        durations=np.array(args.get('duration', defaults['duration'])) * 1e-3,  # s
        PRFs=np.array(args.get('PRF', defaults['PRF'])),  # Hz
        DCs=np.array(args.get('DC', defaults['DC'])) * 1e-2,  # (-)
        offsets=np.array(args.get('offset', defaults['offset'])) * 1e-3  # s
    )
    if titrate:
        stim_params['amps'] = None

    # Run E-STIM batch
    if neuron_str not in getNeuronsDict():
        logger.error('Unknown neuron type: "%s"', neuron_str)
        return
    neuron = getNeuronsDict()[neuron_str]()
    pkl_filepaths = runEStimBatch(outdir, neuron, stim_params, mpi=args['mpi'], loglevel=loglevel)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    if 'plot' in args:
        if args['plot'] == ['all']:
            pltscheme = None
        else:
            pltscheme = {x: [x] for x in args['plot']}
        plotBatch(pkl_filepaths, pltscheme=pltscheme)
        plt.show()


if __name__ == '__main__':
    main()
