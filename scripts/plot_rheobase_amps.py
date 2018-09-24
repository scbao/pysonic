# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-04-30 21:06:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-24 20:48:04

''' Plot neuron-specific rheobase acoustic amplitudes for various duty cycles. '''

import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.utils import logger, si_format
from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.neurons import getNeuronsDict

# Set logging level
logger.setLevel(logging.INFO)


# Default parameters
defaults = dict(
    neurons=['RS', 'FS', 'RE'],
    diam=32.0,
    freq=500.0,
    amps=np.logspace(np.log10(1), np.log10(600), 10),  # kPa
)


def plotRheobaseAmps(a, Fdrive, neurons):

    print(a, Fdrive, neurons)

    # Initialize figure
    fs = 15  # font size
    fig, ax = plt.subplots()
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('Rheobase amplitude (kPa)', fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.set_yscale('log')
    ax.set_ylim([10, 600])

    # Loop through neuron types
    DCs = np.arange(1, 101) / 1e2
    for neuron in neurons:
        nbls = NeuronalBilayerSonophore(a, neuron)
        logger.info('Computing %s neuron rheobase amplitudes at %sHz', neuron.name, si_format(Fdrive))
        Athrs = nbls.findRheobaseAmps(Fdrive, DCs, neuron.VT)
        ax.plot(DCs * 1e2, Athrs * 1e-3, label='{} neuron'.format(neuron.name))

    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    plt.show()

    return fig


def main():
    ap = ArgumentParser()


    # Stimulation parameters
    ap.add_argument('-n', '--neurons', type=str, nargs='+', default=defaults['neurons'],
                    help='Neuron name (string)')
    ap.add_argument('-a', '--diam', type=float, default=defaults['diam'],
                    help='Sonophore diameter (nm)')
    ap.add_argument('-f', '--freq', type=float, default=defaults['freq'],
                    help='US frequency (kHz)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    neurons_str = args['neurons']
    diam = args['diam'] * 1e-9  # m
    Fdrive = args['freq'] * 1e3  # Hz

    neurons = []
    for n in neurons_str:
        if n not in getNeuronsDict():
            logger.error('Unknown neuron type: "%s"', n)
            return
        neurons.append(getNeuronsDict()[n]())

    plotRheobaseAmps(diam, Fdrive, neurons)
    plt.show()


if __name__ == '__main__':
    main()
