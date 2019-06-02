# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-24 11:55:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-02 22:10:34

''' Run E-STIM simulations of a specific point-neuron. '''

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
    mpi = args['mpi']
    outdir = getInDict(args, 'outputdir', selectDirDialog)
    if outdir == '':
        logger.error('No output directory selected')
        return
    titrate = args['titrate']
    neuron_str = args['neuron']
    if neuron_str not in getNeuronsDict():
        logger.error('Unknown neuron type: "%s"', neuron_str)
        return
    neuron = getNeuronsDict()[neuron_str]()
    try:
        amps = parseElecAmps(args, defaults)  # mA/m2
    except ValueError as err:
        logger.error(err)
        return
    if titrate:
        amps = None
    durations = np.array(args.get('duration', defaults['duration'])) * 1e-3  # s
    offsets = np.array(args.get('offset', defaults['offset'])) * 1e-3  # s
    PRFs = np.array(args.get('PRF', defaults['PRF']))  # Hz
    DCs = np.array(args.get('DC', defaults['DC'])) * 1e-2  # (-)

    # Run E-STIM batch
    logger.info("Starting E-STIM simulation batch")
    queue = neuron.simQueue(amps, durations, offsets, PRFs, DCs)
    for item in queue:
        item.insert(0, outdir)
    batch = Batch(neuron.runAndSave, queue)
    pkl_filepaths = batch(mpi=mpi, loglevel=loglevel)

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
