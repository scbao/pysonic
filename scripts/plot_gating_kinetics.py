#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-11 20:35:38
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-14 23:37:42

''' Plot the voltage-dependent steady-states and time constants of activation and inactivation
    gates of the different ionic currents involved in the neuron's membrane dynamics. '''


import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.utils import logger
from PySONIC.neurons import getNeuronsDict


# Default parameters
defaults = dict(
    neuron='RS'
)


def plotGatingKinetics(neuron, fs=15):
    ''' Plot the voltage-dependent steady-states and time constants of activation and
        inactivation gates of the different ionic currents involved in a specific
        neuron's membrane.

        :param neuron: specific channel mechanism object
        :param fs: labels and title font size
    '''

    # Input membrane potential vector
    Vm = np.linspace(-100, 50, 300)

    xinf_dict = {}
    taux_dict = {}

    logger.info('Computing %s neuron gating kinetics', neuron.name)
    names = neuron.states
    for xname in names:
        Vm_state = True

        # Names of functions of interest
        xinf_func_str = xname.lower() + 'inf'
        taux_func_str = 'tau' + xname.lower()
        alphax_func_str = 'alpha' + xname.lower()
        betax_func_str = 'beta' + xname.lower()
        # derx_func_str = 'der' + xname.upper()

        # 1st choice: use xinf and taux function
        if hasattr(neuron, xinf_func_str) and hasattr(neuron, taux_func_str):
            xinf_func = getattr(neuron, xinf_func_str)
            taux_func = getattr(neuron, taux_func_str)
            xinf = np.array([xinf_func(v) for v in Vm])
            if isinstance(taux_func, float):
                taux = taux_func * np.ones(len(Vm))
            else:
                taux = np.array([taux_func(v) for v in Vm])

        # 2nd choice: use alphax and betax functions
        elif hasattr(neuron, alphax_func_str) and hasattr(neuron, betax_func_str):
            alphax_func = getattr(neuron, alphax_func_str)
            betax_func = getattr(neuron, betax_func_str)
            alphax = np.array([alphax_func(v) for v in Vm])
            if isinstance(betax_func, float):
                betax = betax_func * np.ones(len(Vm))
            else:
                betax = np.array([betax_func(v) for v in Vm])
            taux = 1.0 / (alphax + betax)
            xinf = taux * alphax

        # # 3rd choice: use derX choice
        # elif hasattr(neuron, derx_func_str):
        #     derx_func = getattr(neuron, derx_func_str)
        #     xinf = brentq(lambda x: derx_func(neuron.Vm, x), 0, 1)
        else:
            Vm_state = False
        if not Vm_state:
            logger.error('no function to compute %s-state gating kinetics', xname)
        else:
            xinf_dict[xname] = xinf
            taux_dict[xname] = taux

    fig, axes = plt.subplots(2)
    fig.suptitle('{} neuron: gating dynamics'.format(neuron.name))

    ax = axes[0]
    ax.get_xaxis().set_ticklabels([])
    ax.set_ylabel('$X_{\infty}$', fontsize=fs)
    for xname in names:
        if xname in xinf_dict:
            ax.plot(Vm, xinf_dict[xname], lw=2, label='$' + xname + '_{\infty}$')
    ax.legend(fontsize=fs, loc=7)

    ax = axes[1]
    ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
    ax.set_ylabel('$\\tau_X\ (ms)$', fontsize=fs)
    for xname in names:
        if xname in taux_dict:
            ax.plot(Vm, taux_dict[xname] * 1e3, lw=2, label='$\\tau_{' + xname + '}$')
    ax.legend(fontsize=fs, loc=7)

    return fig


def main():
    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
                    help='Neuron name (string)')

    # Parse arguments
    args = ap.parse_args()
    neuron_str = args.neuron

    # Plot gating kinetics variables
    if neuron_str not in getNeuronsDict():
        logger.error('Unknown neuron type: "%s"', neuron_str)
        return
    neuron = getNeuronsDict()[neuron_str]()
    plotGatingKinetics(neuron)
    plt.show()


if __name__ == '__main__':
    main()
