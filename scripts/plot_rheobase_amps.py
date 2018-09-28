# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-04-30 21:06:10
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-28 14:12:31

''' Plot duty-cycle dependent rheobase acoustic amplitudes of various neurons
    for a specific US frequency and PRF. '''

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
    freq=500.0
)


def plotAstimRheobaseAmps(neurons, a, Fdrive, fs=15):
    fig, ax = plt.subplots()
    ax.set_title('Rheobase amplitudes @ {}Hz ({:.0f} nm sonophore)'.format(
        si_format(Fdrive, 1, space=' '), a * 1e9), fontsize=fs)
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('Threshold amplitude (kPa)', fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.set_yscale('log')
    ax.set_ylim([10, 600])
    DCs = np.arange(1, 101) / 1e2
    for neuron in neurons:
        nbls = NeuronalBilayerSonophore(a, neuron)
        Athrs = nbls.findRheobaseAmps(DCs, Fdrive, neuron.VT)
        ax.plot(DCs * 1e2, Athrs * 1e-3, label='{} neuron'.format(neuron.name))
    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    return fig


def plotEstimRheobaseAmps(neurons, fs=15):
    fig, ax = plt.subplots()
    ax.set_title('Rheobase amplitudes', fontsize=fs)
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('Threshold amplitude (mA/m2)', fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.set_yscale('log')
    ax.set_ylim([1e0, 1e3])
    DCs = np.arange(1, 101) / 1e2
    for neuron in neurons:
        Athrs = neuron.findRheobaseAmps(DCs, neuron.VT)
        ax.plot(DCs * 1e2, Athrs, label='{} neuron'.format(neuron.name))
    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
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
    ap.add_argument('-m', '--mode', type=str, default='US',
                    help='Stimulation modality (US or elec)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    neurons_str = args.get('neurons', defaults['neurons'])
    neurons = []
    for n in neurons_str:
        if n not in getNeuronsDict():
            logger.error('Invalid neuron type: "%s"', n)
            return
        neurons.append(getNeuronsDict()[n]())
    mode = args['mode']
    if mode == 'US':
        diam = args.get('diam', defaults['diam']) * 1e-9  # m
        Fdrive = args.get('freq', defaults['freq']) * 1e3  # Hz
        plotAstimRheobaseAmps(neurons, diam, Fdrive)
    elif mode == 'elec':
        plotEstimRheobaseAmps(neurons)
    else:
        logger.error('Invalid stimulation type: "%s"', mode)
        return
    plt.show()


if __name__ == '__main__':
    main()
