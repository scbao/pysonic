#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-15 15:59:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-02 02:53:08

''' Plot the effective variables as a function of charge density with amplitude color code. '''

import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging

from PySONIC.core import BilayerSonophore
from PySONIC.plt import plotEffectiveVariables, plotEffectiveCapacitance
from PySONIC.utils import logger, selectDirDialog
from PySONIC.neurons import getNeuronsDict
from PySONIC.constants import NPC_FULL


def fig4_main(neuron, a, Fdrive, amps):
    neuron = getNeuronsDict()[neuron]()
    mainfig = plotEffectiveVariables(neuron, a, Fdrive, amps=amps)
    mainfig.canvas.set_window_title('fig4 main')
    Cmfig = plotEffectiveCapacitance(neuron, a, Fdrive, amps=amps)
    Cmfig.canvas.set_window_title('fig4 Cmeff')
    return mainfig, Cmfig


def fig4_gas(a, neuron, Fdrive, Adrive, charges, fs=8):
    neuron = getNeuronsDict()[neuron]()
    bls = BilayerSonophore(a, neuron.Cm0, neuron.Cm0 * neuron.Vm0 * 1e-3)
    t = np.linspace(0, 1 / Fdrive, NPC_FULL) * 1e6
    fig, ax = plt.subplots(figsize=(3, 3))
    insetfig, insetax = plt.subplots(figsize=(3, 3))
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    ax.set_xticks([0, 1e6 / Fdrive])
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.set_xlabel('$\\rm {:.0f}\ \mu s$'.format(t[-1]), fontsize=fs)
    ax.set_ylabel('$\\rm 10^{22}\ mol$', fontsize=fs)
    insetax.set_xlim([1e6 / Fdrive - 0.05, 1e6 / Fdrive])
    insetax.set_ylim([1.4, 1.7])
    insetax.set_xticks([])
    insetax.set_yticks([])
    ngmin, ngmax = np.inf, -np.inf
    for Qm in charges:
        _, (_, ng), _ = bls.simulate(Fdrive, Adrive, Qm)
        ng = ng[-NPC_FULL:]
        ngmin, ngmax = min(ng.min(), ngmin), max(ng.max(), ngmax)
        ax.plot(t, ng * 1e22, label='{:.0f} nC/cm2'.format(Qm * 1e5))
        insetax.plot(t, ng * 1e22, label='{:.0f} nC/cm2'.format(Qm * 1e5))
    ax.set_yticks([np.floor(ngmin * 1e22), np.ceil(ngmax * 1e22)])
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()
    fig.canvas.set_window_title('fig4 gas')
    insetfig.canvas.set_window_title('fig4 inset gas')
    return fig, insetfig


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-i', '--inputdir', type=str, help='Input directory')
    ap.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save output figures as pdf')

    args = ap.parse_args()
    loglevel = logging.DEBUG if args.verbose is True else logging.INFO
    logger.setLevel(loglevel)

    # Parameters
    neuron = 'RS'
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    amps = np.logspace(np.log10(1), np.log10(600), 10) * 1e3  # Pa

    # Generate figures
    figs = []
    figs += fig4_main(neuron, a, Fdrive, amps)
    figs += fig4_gas(a, neuron, Fdrive, 600e3, [0., 13e-5])

    if args.save:
        inputdir = selectDirDialog() if args.inputdir is None else args.inputdir
        if inputdir == '':
            logger.error('No input directory chosen')
            return
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(inputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
