# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-26 09:51:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-06 21:01:54

''' Sub-panels of (duty-cycle x amplitude) US activation maps and related Q-V traces. '''

import os
import numpy as np
import logging
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import logger, selectDirDialog, si_format
from PySONIC.plt import ActivationMap
from PySONIC.neurons import getPointNeuron

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def plotMapAndTraces(inputdir, neuron, a, Fdrive, tstim, amps, PRF, DCs, FRbounds,
                     insets, tmax, Vbounds, prefix):
    # Activation map
    mapcode = '{} {}Hz PRF{}Hz 1s'.format(neuron.name, *si_format([Fdrive, PRF, tstim], space=''))
    subdir = os.path.join(inputdir, mapcode)
    actmap = ActivationMap(subdir, neuron, a, Fdrive, tstim, PRF, amps, DCs)
    mapfig = actmap.render(FRbounds=FRbounds, thresholds=True)
    mapfig.canvas.set_window_title('{} map {}'.format(prefix, mapcode))
    ax = mapfig.axes[0]
    DC_insets, A_insets = zip(*insets)
    ax.scatter(DC_insets, A_insets, s=80, facecolors='none', edgecolors='k', linestyle='--')

    # Related inset traces
    tracefigs = []
    nbls = NeuronalBilayerSonophore(a, neuron)
    for inset in insets:
        DC = inset[0] * 1e-2
        Adrive = inset[1] * 1e3
        fname = '{}.pkl'.format(nbls.filecode(
            Fdrive, actmap.correctAmp(Adrive), tstim, 0., PRF, DC, 'sonic'))
        fpath = os.path.join(subdir, fname)
        tracefig = actmap.plotQVeff(fpath, tmax=tmax, ybounds=Vbounds)
        figcode = '{} VQ trace {} {:.1f}kPa {:.0f}%DC'.format(
            prefix, neuron.name, Adrive * 1e-3, DC * 1e2)
        tracefig.canvas.set_window_title(figcode)
        tracefigs.append(tracefig)

    return mapfig, tracefigs


def panel(inputdir, neurons, a, tstim, PRF, amps, DCs, FRbounds, tmax, Vbounds, insets, prefix):

    mapfigs, tracefigs = [], []
    for n in neurons:
        out = plotMapAndTraces(
            inputdir, n, a, 500e3, tstim, amps, PRF, DCs,
            FRbounds, insets[n.name], tmax, Vbounds, prefix)
        mapfigs.append(out[0])
        tracefigs += out[1]

    return mapfigs + tracefigs


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-i', '--inputdir', type=str, help='Input directory')
    ap.add_argument('-f', '--figset', type=str, nargs='+', help='Figure set', default='all')
    ap.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save output figures as pdf')

    args = ap.parse_args()
    loglevel = logging.DEBUG if args.verbose is True else logging.INFO
    logger.setLevel(loglevel)
    inputdir = selectDirDialog() if args.inputdir is None else args.inputdir
    if inputdir == '':
        logger.error('No input directory chosen')
        return
    figset = args.figset
    if figset == 'all':
        figset = ['a', 'b', 'c']

    logger.info('Generating panel {} of {}'.format(figset, figbase))

    # Parameters
    neurons = [getPointNeuron(n) for n in ['RS', 'LTS']]
    a = 32e-9  # m
    tstim = 1.0  # s
    amps = np.logspace(np.log10(10), np.log10(600), num=30) * 1e3  # Pa
    DCs = np.arange(1, 101) * 1e-2
    FRbounds = (1e0, 1e3)  # Hz

    tmax = 240  # ms
    Vbounds = -150, 50  # mV

    # Generate figures
    try:

        figs = []
        if 'a' in figset:
            PRF = 1e1
            insets = {
                'RS': [(28, 127.0), (37, 168.4)],
                'LTS': [(8, 47.3), (30, 146.2)]
            }
            figs += panel(inputdir, neurons, a, tstim, PRF, amps, DCs, FRbounds, tmax, Vbounds,
                          insets, figbase + 'a')
        if 'b' in figset:
            PRF = 1e2
            insets = {
                'RS': [(51, 452.4), (56, 452.4)],
                'LTS': [(13, 193.9), (43, 257.2)]
            }
            figs += panel(inputdir, neurons, a, tstim, PRF, amps, DCs, FRbounds, tmax, Vbounds,
                          insets, figbase + 'b')
        if 'c' in figset:
            PRF = 1e3
            insets = {
                'RS': [(40, 110.2), (64, 193.9)],
                'LTS': [(10, 47.3), (53, 168.4)]
            }
            figs += panel(inputdir, neurons, a, tstim, PRF, amps, DCs, FRbounds, tmax, Vbounds,
                          insets, figbase + 'c')

    except Exception as e:
        logger.error(e)
        quit()

    if args.save:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(inputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
