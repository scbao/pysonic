# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-26 09:51:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-02 01:39:49

''' Plot (duty-cycle x amplitude) US activation map of a neuron at a given frequency and PRF. '''

import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.utils import logger, selectDirDialog, si_format, ASTIM_filecode
from PySONIC.plt import plotActivationMap, plotQVeff


def plot_actmap(inputdir, neuron, a, Fdrive, tstim, amps, PRF, DCs, FRbounds, prefix):
    mapcode = '{} {}Hz PRF{}Hz 1s'.format(neuron, *si_format([Fdrive, PRF, tstim], space=''))
    subdir = os.path.join(inputdir, mapcode)
    fig = plotActivationMap(
        subdir, neuron, a, Fdrive, tstim, PRF, amps, DCs, FRbounds=FRbounds)
    fig.canvas.set_window_title('{} map {}'.format(prefix, mapcode))
    return fig


def plot_traces(inputdir, neuron, a, Fdrive, Adrive, tstim, PRF, DC, tmax, Vbounds, prefix):
    mapcode = '{} {}Hz PRF{}Hz 1s'.format(neuron, *si_format([Fdrive, PRF, tstim], space=''))
    subdir = os.path.join(inputdir, mapcode)
    fname = '{}.pkl'.format(ASTIM_filecode(neuron, a, Fdrive, Adrive, tstim, PRF, DC, 'sonic'))
    fpath = os.path.join(subdir, fname)
    fig = plotQVeff(fpath, tmax=tmax, ybounds=Vbounds)
    figcode = '{} VQ trace {}kPa {}%DC'.format(prefix, Adrive * 1e-3, DC * 1e2)
    fig.canvas.set_window_title(figcode)
    return fig


def fig7a(inputdir, neuron, a, tstim, amps, DCs, FRbounds, tmax, Vbounds):
    prefix = 'fig7a'
    mapfigs = [
        plot_actmap(inputdir, neuron, a, 500e3, tstim, amps, 1e1, DCs, FRbounds, prefix),
        plot_actmap(inputdir, neuron, a, 500e3, tstim, amps, 1e2, DCs, FRbounds, prefix),
        plot_actmap(inputdir, neuron, a, 500e3, tstim, amps, 1e3, DCs, FRbounds, prefix),
        plot_actmap(inputdir, neuron, a, 4e6, tstim, amps, 1e2, DCs, FRbounds, prefix)
    ]
    tracefigs = [
        plot_traces(inputdir, neuron, a, 500e3, 41.0e3, tstim, 1e2, 1.0, tmax, Vbounds, prefix),
        plot_traces(inputdir, neuron, a, 500e3, 62.7e3, tstim, 1e2, 0.29, tmax, Vbounds, prefix),
        plot_traces(inputdir, neuron, a, 500e3, 452.4e3, tstim, 1e2, 0.51, tmax, Vbounds, prefix),
        plot_traces(inputdir, neuron, a, 500e3, 452.4e3, tstim, 1e2, 0.56, tmax, Vbounds, prefix)
    ]
    return mapfigs + tracefigs


def fig7b(inputdir, neuron, a, tstim, amps, DCs, FRbounds, tmax, Vbounds):
    prefix = 'fig7b'
    mapfig = plot_actmap(inputdir, neuron, a, 500e3, tstim, amps, 1e2, DCs, FRbounds, prefix)
    tracefigs = [
        plot_traces(inputdir, neuron, a, 500e3, 26.9e3, tstim, 1e2, 1.0, tmax, Vbounds, prefix),
        plot_traces(inputdir, neuron, a, 500e3, 127.0e3, tstim, 1e2, 0.04, tmax, Vbounds, prefix)
    ]
    return [mapfig] + tracefigs



def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
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
        figset = ['a', 'b']

    # Parameters
    a = 32e-9  # m
    tstim = 1.0  # s
    amps = np.logspace(np.log10(10), np.log10(600), num=30) * 1e3  # Pa
    DCs = np.arange(1, 101) * 1e-2
    FRbounds = (1e0, 1e3)  # Hz

    tmax = 240  # ms
    Vbounds = -150, 50  # mV

    # Generate figures
    figs = []
    if 'a' in figset:
        figs += fig7a(inputdir, 'RS', a, tstim, amps, DCs, FRbounds, tmax, Vbounds)
    if 'b' in figset:
        figs += fig7b(inputdir, 'LTS', a, tstim, amps, DCs, FRbounds, tmax, Vbounds)

    if args.save:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(inputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
