#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-15 15:59:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-11-21 14:48:49

''' Plot the effective variables as a function of charge density with color code. '''

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.plt import plotEffectiveVariables
from PySONIC.utils import logger
from PySONIC.neurons import getNeuronsDict


# Default parameters
defaults = dict(
    neuron='RS',
    radius=32.0,
    freq=500.0,
    amps=np.logspace(np.log10(1), np.log10(600), 10),  # kPa
)


def main():
    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
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

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    neuron_str = args['neuron']
    a = args['radius'] * 1e-9 if 'radius' in args else None  # m
    Fdrive = args['freq'] * 1e3 if 'freq' in args else None  # Hz
    Adrive = args['amp'] * 1e3 if 'amp' in args else None  # Pa

    zscale = 'log' if args['log'] else 'lin'
    cmap = args.get('cmap', None)
    ncol = args['ncol']

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
