#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-15 15:59:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-02 01:46:25

''' Plot the effective variables as a function of charge density with amplitude color code. '''

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.plt import plotEffectiveVariables
from PySONIC.utils import logger
from PySONIC.neurons import getNeuronsDict


# Default parameters
defaults = dict(
    neuron='RS',
    diam=32.0,
    freq=500.0,
    amps=np.logspace(np.log10(1), np.log10(600), 10),  # kPa
)


def main():
    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-a', '--diam', type=float, default=defaults['diam'],
                    help='Sonophore diameter (nm)')
    ap.add_argument('-f', '--freq', type=float, default=defaults['freq'],
                    help='US frequency (kHz)')
    ap.add_argument('-A', '--amps', nargs='+', type=float, help='Acoustic pressure amplitude (kPa)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    neuron_str = args['neuron']
    diam = args['diam'] * 1e-9  # m
    Fdrive = args['freq'] * 1e3  # Hz
    amps = np.array(args.get('amps', defaults['amps'])) * 1e3  # Pa

    # Plot effective variables
    if neuron_str not in getNeuronsDict():
        logger.error('Unknown neuron type: "%s"', neuron_str)
        return
    neuron = getNeuronsDict()[neuron_str]()
    plotEffectiveVariables(neuron, diam, Fdrive, amps=amps)
    plt.show()


if __name__ == '__main__':
    main()
