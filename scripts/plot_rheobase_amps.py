# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-04-30 21:06:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-12 12:23:44

''' Plot duty-cycle dependent rheobase acoustic amplitudes of various neurons
    for a specific US frequency and PRF. '''

import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.utils import logger
from PySONIC.neurons import getPointNeuron
from PySONIC.plt import plotAstimRheobaseAmps, plotEstimRheobaseAmps

# Set logging level
logger.setLevel(logging.INFO)


# Default parameters
defaults = dict(
    neuron='RS',
    radii=[32.0],
    freqs=[500.0]
)


def main():
    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-a', '--radii', type=float, nargs='+', default=defaults['radii'],
                    help='Sonophore radius (nm)')
    ap.add_argument('-f', '--freqs', type=float, nargs='+', default=defaults['freqs'],
                    help='US frequency (kHz)')
    ap.add_argument('-m', '--mode', type=str, default='US',
                    help='Stimulation modality (US or elec)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    mode = args['mode']

    # Get neurons objects from names
    neuron_str = args.get('neuron', defaults['neuron'])
    try:
        pneuron = getPointNeuron(neuron_str)
    except ValueError as err:
        logger.error(err)
        return

    if mode == 'US':
        radii = np.array(args['radii']) * 1e-9  # m
        freqs = np.array(args['freqs']) * 1e3  # Hz
        plotAstimRheobaseAmps(pneuron, radii, freqs)
    elif mode == 'elec':
        plotEstimRheobaseAmps(pneuron)
    else:
        logger.error('Invalid stimulation type: "%s"', mode)
        return
    plt.show()


if __name__ == '__main__':
    main()
