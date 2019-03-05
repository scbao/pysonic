#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-15 15:59:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-05 11:00:33

''' Plot the effective variables as a function of charge density with color code. '''

import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.plt import plotEffectiveVariables
from PySONIC.utils import logger, Intensity2Pressure
from PySONIC.neurons import getNeuronsDict

# Set logging level
logger.setLevel(logging.INFO)


def main():
    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default='RS',
                    help='Neuron name (string)')
    ap.add_argument('-a', '--radius', type=float, default=None,
                    help='Sonophore radius (nm)')
    ap.add_argument('-f', '--freq', type=float, default=None,
                    help='US frequency (kHz)')
    ap.add_argument('-A', '--amp', type=float, default=None,
                    help='Acoustic pressure amplitude (kPa)')
    ap.add_argument('--log', action='store_true', default=False,
                    help='Log color scale')
    ap.add_argument('-c', '--cmap', type=str, default=None,
                    help='Colormap name')
    ap.add_argument('--ncol', type=int, default=1,
                    help='Number of columns in figure')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    neuron_str = args['neuron']
    a = args['radius'] * 1e-9 if 'radius' in args else None  # m
    Fdrive = args['freq'] * 1e3 if 'freq' in args else None  # Hz
    Adrive = args['amp'] * 1e3 if 'amp' in args else None  # Pa

    # Range of intensities
    if neuron_str == 'STN':
        intensities = np.hstack((
            np.arange(10, 101, 10),
            np.arange(101, 131, 1),
            np.array([140])
        ))  # W/m2
        Adrive = np.array([Intensity2Pressure(I) for I in intensities])  # Pa

    zscale = 'log' if args['log'] else 'lin'
    cmap = args.get('cmap', None)
    ncol = args['ncol']

    loglevel = logging.DEBUG if args['verbose'] is True else logging.INFO
    logger.setLevel(loglevel)

    # Plot effective variables
    if neuron_str not in getNeuronsDict():
        logger.error('Unknown neuron type: "%s"', neuron_str)
        return
    neuron = getNeuronsDict()[neuron_str]()
    plotEffectiveVariables(neuron, a=a, Fdrive=Fdrive, Adrive=Adrive,
                           zscale=zscale, cmap=cmap, ncolmax=ncol)
    plt.show()


if __name__ == '__main__':
    main()
