# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-26 09:51:43
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-12-14 15:59:36

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
        subdir, neuron, a, Fdrive, tstim, PRF, amps, DCs, FRbounds=FRbounds, thrs=True)
    fig.canvas.set_window_title('{} map {}'.format(prefix, mapcode))
    return fig


def plot_traces(inputdir, neuron, a, Fdrive, Adrive, tstim, PRF, DC, tmax, Vbounds, prefix):
    mapcode = '{} {}Hz PRF{}Hz 1s'.format(neuron, *si_format([Fdrive, PRF, tstim], space=''))
    subdir = os.path.join(inputdir, mapcode)
    fname = '{}.pkl'.format(ASTIM_filecode(neuron, a, Fdrive, Adrive, tstim, PRF, DC, 'sonic'))
    fpath = os.path.join(subdir, fname)
    fig = plotQVeff(fpath, tmax=tmax, ybounds=Vbounds)
    figcode = '{} VQ trace {:.1f}kPa {:.0f}%DC'.format(prefix, Adrive * 1e-3, DC * 1e2)
    fig.canvas.set_window_title(figcode)
    return fig


def fig7a(inputdir, a, tstim, amps, DCs, FRbounds, tmax, Vbounds):
    prefix = 'fig7a'
    mapfigs = [
        plot_actmap(inputdir, n, a, 500e3, tstim, amps, 1e1, DCs, FRbounds, prefix)
        for n in ['RS', 'LTS']
    ]

    tracefigs = [
        plot_traces(inputdir, 'RS', a, 500e3, 127.0e3, tstim, 1e1, 0.28, tmax, Vbounds, prefix),
        plot_traces(inputdir, 'RS', a, 500e3, 168.4e3, tstim, 1e1, 0.37, tmax, Vbounds, prefix),
        plot_traces(inputdir, 'LTS', a, 500e3, 47.3e3, tstim, 1e1, 0.08, tmax, Vbounds, prefix),
        plot_traces(inputdir, 'LTS', a, 500e3, 146.2e3, tstim, 1e1, 0.30, tmax, Vbounds, prefix)
    ]

    return mapfigs + tracefigs


def fig7b(inputdir, a, tstim, amps, DCs, FRbounds, tmax, Vbounds):
    prefix = 'fig7b'
    mapfigs = [
        plot_actmap(inputdir, n, a, 500e3, tstim, amps, 1e2, DCs, FRbounds, prefix)
        for n in ['RS', 'LTS']
    ]

    tracefigs = [
        plot_traces(inputdir, 'RS', a, 500e3, 452.4e3, tstim, 1e2, 0.51, tmax, Vbounds, prefix),
        plot_traces(inputdir, 'RS', a, 500e3, 452.4e3, tstim, 1e2, 0.56, tmax, Vbounds, prefix),
        plot_traces(inputdir, 'LTS', a, 500e3, 193.9e3, tstim, 1e2, 0.13, tmax, Vbounds, prefix),
        plot_traces(inputdir, 'LTS', a, 500e3, 257.2e3, tstim, 1e2, 0.43, tmax, Vbounds, prefix)
    ]

    return mapfigs + tracefigs


def fig7c(inputdir, a, tstim, amps, DCs, FRbounds, tmax, Vbounds):
    prefix = 'fig7c'
    mapfigs = [
        plot_actmap(inputdir, n, a, 500e3, tstim, amps, 1e3, DCs, FRbounds, prefix)
        for n in ['RS', 'LTS']
    ]

    tracefigs = [
        plot_traces(inputdir, 'RS', a, 500e3, 110.2e3, tstim, 1e3, 0.40, tmax, Vbounds, prefix),
        plot_traces(inputdir, 'RS', a, 500e3, 193.9e3, tstim, 1e3, 0.64, tmax, Vbounds, prefix),
        plot_traces(inputdir, 'LTS', a, 500e3, 47.3e3, tstim, 1e3, 0.10, tmax, Vbounds, prefix),
        plot_traces(inputdir, 'LTS', a, 500e3, 168.4e3, tstim, 1e3, 0.53, tmax, Vbounds, prefix)
    ]

    return mapfigs + tracefigs



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
    try:

        figs = []
        if 'a' in figset:
            figs += fig7a(inputdir, a, tstim, amps, DCs, FRbounds, tmax, Vbounds)
        if 'b' in figset:
            figs += fig7b(inputdir, a, tstim, amps, DCs, FRbounds, tmax, Vbounds)
        if 'c' in figset:
            figs += fig7c(inputdir, a, tstim, amps, DCs, FRbounds, tmax, Vbounds)
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
