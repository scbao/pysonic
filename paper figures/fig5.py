# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-06-06 18:38:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-01 21:55:48


import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.utils import *
from PySONIC.neurons import *
from PySONIC.batches import createAStimQueue
from PySONIC.plt import plotComp, plotSpikingMetrics

from utils import *

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'


def fig5aleft(neuron, a, Fdrive, CW_Athrs, tstim, toffset, inputdir):
    ''' Comparison of resulting charge profiles for CW stimuli at sub-threshold,
        threshold and supra-threshold amplitudes. '''
    Athr = CW_Athrs[neuron].loc[Fdrive * 1e-3]  # kPa
    amps = np.array([Athr - 5., Athr, Athr + 20.]) * 1e3  # Pa
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths = getSims(subdir, neuron, a, createAStimQueue(
        [Fdrive], amps, [tstim], [toffset], [None], [1.], 'sonic'))
    full_fpaths = getSims(subdir, neuron, a, createAStimQueue(
        [Fdrive], amps, [tstim], [toffset], [None], [1.], 'full'))
    regimes = ['threshold - 5 kPa', 'threshold', 'threshold + 20 kPa']
    fig = plotComp(
        sum([[x, y] for x, y in zip(full_fpaths, sonic_fpaths)], []),
        'Qm',
        labels=sum([['', x] for x in regimes], []),
        lines=['-', '--'] * len(regimes),
        colors=plt.get_cmap('Paired').colors[:2 * len(regimes)],
        fs=8, patches='one', xticks=[0, 250], yticks=[getNeuronsDict()[neuron].Vm0, 25],
        straightlegend=True, figsize=cm2inch(12.5, 5.8)
    )
    fig.axes[0].get_xaxis().set_label_coords(0.5, -0.05)
    fig.subplots_adjust(bottom=0.2, right=0.95, top=0.95)
    fig.canvas.set_window_title('fig5a left')
    return fig


def fig5aright(neuron, a, Fdrive, amps, tstim, toffset, inputdir):
    ''' Comparison of spiking metrics for CW stimuli at various supra-threshold amplitudes. '''
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths = getSims(subdir, neuron, a, createAStimQueue(
        [Fdrive], amps, [tstim], [toffset], [None], [1.], 'sonic'))
    full_fpaths = getSims(subdir, neuron, a, createAStimQueue(
        [Fdrive], amps, [tstim], [toffset], [None], [1.], 'full'))
    metrics_files = {x: '{}_spikemetrics_vs_amplitude_{}.csv'.format(neuron, x)
                     for x in ['full', 'sonic']}
    metrics_fpaths = {key: os.path.join(inputdir, value) for key, value in metrics_files.items()}
    xlabel = 'Amplitude (kPa)'
    full_metrics, sonic_metrics = getSpikingMetrics(
        subdir, neuron, amps * 1e-3, xlabel, full_fpaths, sonic_fpaths, metrics_fpaths)
    metrics_dict = {neuron: {'full': full_metrics, 'sonic': sonic_metrics}}
    fig = plotSpikingMetrics(amps * 1e-3, xlabel, metrics_dict, logscale=True)
    fig.canvas.set_window_title('fig5a right')
    return fig


def fig5bleft(neuron, a, freqs, CW_Athrs, tstim, toffset, inputdir):
    ''' Comparison of resulting charge profiles for supra-threshold CW stimuli
        at low and high US frequencies. '''
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths, full_fpaths = [], []
    for Fdrive in freqs:
        Athr = CW_Athrs[neuron].loc[Fdrive * 1e-3]  # kPa
        Adrive = (Athr + 20.) * 1e3  # Pa
        sonic_fpaths += getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'sonic'))
        full_fpaths += getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'full'))
    fig = plotComp(
        sum([[x, y] for x, y in zip(full_fpaths, sonic_fpaths)], []),
        'Qm',
        labels=sum([['', '{}Hz'.format(si_format(f))] for f in freqs], []),
        lines=['-', '--'] * len(freqs), colors=plt.get_cmap('Paired').colors[6:10], fs=8,
        patches='one', xticks=[0, 250], yticks=[getNeuronsDict()[neuron].Vm0, 25],
        straightlegend=True, figsize=cm2inch(12.5, 5.8),
        inset={'xcoords': [5, 40], 'ycoords': [-35, 45], 'xlims': [57.5, 58.5], 'ylims': [10, 35]}
    )
    fig.axes[0].get_xaxis().set_label_coords(0.5, -0.05)
    fig.subplots_adjust(bottom=0.2, right=0.95, top=0.95)
    fig.canvas.set_window_title('fig5b left')
    return fig


def fig5bright(neuron, a, freqs, CW_Athrs, tstim, toffset, inputdir):
    ''' Comparison of spiking metrics for supra-threshold CW stimuli at various US frequencies. '''
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths, full_fpaths = [], []
    for Fdrive in freqs:
        Athr = CW_Athrs[neuron].loc[Fdrive * 1e-3]  # kPa
        Adrive = (Athr + 20.) * 1e3  # Pa
        sonic_fpaths += getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'sonic'))
        full_fpaths += getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'full'))
    metrics_files = {x: '{}_spikemetrics_vs_frequency_{}.csv'.format(neuron, x)
                     for x in ['full', 'sonic']}
    metrics_fpaths = {key: os.path.join(inputdir, value) for key, value in metrics_files.items()}
    xlabel = 'Frequency (kHz)'
    full_metrics, sonic_metrics = getSpikingMetrics(
        subdir, neuron, freqs * 1e-3, xlabel, full_fpaths, sonic_fpaths, metrics_fpaths)
    metrics_dict = {neuron: {'full': full_metrics, 'sonic': sonic_metrics}}
    fig = plotSpikingMetrics(freqs * 1e-3, xlabel, metrics_dict, logscale=True)
    fig.canvas.set_window_title('fig5b right')
    return fig


def fig5cleft(neurons, a, Fdrive, Adrive, tstim, toffset, PRF, DC, inputdir):
    ''' Comparison of resulting charge profiles for PW stimuli at 5% duty cycle
        for different neuron types. '''
    sonic_fpaths, full_fpaths = [], []
    for neuron in neurons:
        subdir = os.path.join(inputdir, neuron)
        sonic_fpaths += getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [PRF], [DC], 'sonic'))
        full_fpaths += getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [PRF], [DC], 'full'))
    colors = list(plt.get_cmap('Paired').colors[:6])
    del colors[2:4]
    fig = plotComp(
        sum([[x, y] for x, y in zip(full_fpaths, sonic_fpaths)], []),
        'Qm',
        labels=sum([['', '{}, {:.0f}% DC'.format(x, DC * 1e2)] for x in neurons], []),
        lines=['-', '--'] * len(neurons), colors=colors, fs=8, patches='one',
        xticks=[0, 250], yticks=[min(getNeuronsDict()[n].Vm0 for n in neurons), 50],
        straightlegend=True, figsize=cm2inch(12.5, 5.8),
        inset={'xcoords': [130, 210], 'ycoords': [-35, 45],
               'xlims': [109.5, 121.0], 'ylims': [-60.0, 49.0]}
    )
    fig.axes[0].get_xaxis().set_label_coords(0.5, -0.05)
    fig.subplots_adjust(bottom=0.2, right=0.95, top=0.95)
    fig.canvas.set_window_title('fig5c left')
    return fig


def fig5cright(neurons, a, Fdrive, Adrive, tstim, toffset, PRF, DCs, inputdir):
    ''' Comparison of spiking metrics for PW stimuli at various duty cycle for
        different neuron types. '''
    metrics_dict = {}
    xlabel = 'Duty cycle (%)'
    colors = list(plt.get_cmap('Paired').colors[:6])
    del colors[2:4]
    colors_dict = {}
    for i, neuron in enumerate(neurons):
        subdir = os.path.join(inputdir, neuron)
        sonic_fpaths = getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [PRF], DCs, 'sonic'))
        full_fpaths = getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [PRF], DCs, 'full'))
        metrics_files = {x: '{}_spikemetrics_vs_DC_{}.csv'.format(neuron, x)
                         for x in ['full', 'sonic']}
        metrics_fpaths = {key: os.path.join(inputdir, value) for key, value in metrics_files.items()}
        sonic_fpaths = sonic_fpaths[1:] + [sonic_fpaths[0]]
        full_fpaths = full_fpaths[1:] + [full_fpaths[0]]
        full_metrics, sonic_metrics = getSpikingMetrics(
            subdir, neuron, DCs * 1e2, xlabel, full_fpaths, sonic_fpaths, metrics_fpaths)
        metrics_dict[neuron] = {'full': full_metrics, 'sonic': sonic_metrics}
        colors_dict[neuron] = {'full': colors[2 * i], 'sonic': colors[2 * i + 1]}
    fig = plotSpikingMetrics(DCs * 1e2, xlabel, metrics_dict, spikeamp=False, colors=colors_dict)
    fig.canvas.set_window_title('fig5c right')
    return fig


def fig5dleft(neuron, a, Fdrive, Adrive, tstim, toffset, PRFs, DC, inputdir):
    ''' Comparison of resulting charge profiles for PW stimuli at 5% duty cycle
        with different pulse repetition frequencies. '''
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths = getSims(subdir, neuron, a, createAStimQueue(
        [Fdrive], [Adrive], [tstim], [toffset], PRFs, [DC], 'sonic'))
    full_fpaths = getSims(subdir, neuron, a, createAStimQueue(
        [Fdrive], [Adrive], [tstim], [toffset], PRFs, [DC], 'full'))
    patches = [False, True] * len(PRFs)
    patches[-1] = False
    fig = plotComp(
        sum([[x, y] for x, y in zip(full_fpaths, sonic_fpaths)], []),
        'Qm',
        labels=sum([['', '{}Hz PRF'.format(si_format(PRF, space=' '))] for PRF in PRFs], []),
        lines=['-', '--'] * len(PRFs), colors=plt.get_cmap('Paired').colors[4:12], fs=8,
        patches=patches,
        xticks=[0, 250], yticks=[getNeuronsDict()[neuron].Vm0, 50],
        straightlegend=True, figsize=cm2inch(12.5, 5.8)
    )
    fig.axes[0].get_xaxis().set_label_coords(0.5, -0.05)
    fig.subplots_adjust(bottom=0.2, right=0.95, top=0.95)
    fig.canvas.set_window_title('fig5d left')
    return fig


def fig5dright(neuron, a, Fdrive, Adrive, tstim, toffset, PRFs, DC, inputdir):
    ''' Comparison of spiking metrics for PW stimuli at 5% duty cycle
        with different pulse repetition frequencies. '''
    xlabel = 'PRF (Hz)'
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths = getSims(subdir, neuron, a, createAStimQueue(
        [Fdrive], [Adrive], [tstim], [toffset], PRFs, [DC], 'sonic'))
    full_fpaths = getSims(subdir, neuron, a, createAStimQueue(
        [Fdrive], [Adrive], [tstim], [toffset], PRFs, [DC], 'full'))
    metrics_files = {x: '{}_spikemetrics_vs_PRF_{}.csv'.format(neuron, x)
                     for x in ['full', 'sonic']}
    metrics_fpaths = {key: os.path.join(inputdir, value) for key, value in metrics_files.items()}
    full_metrics, sonic_metrics = getSpikingMetrics(
        subdir, neuron, PRFs, xlabel, full_fpaths, sonic_fpaths, metrics_fpaths)
    metrics_dict = {neuron: {'full': full_metrics, 'sonic': sonic_metrics}}
    fig = plotSpikingMetrics(PRFs, xlabel, metrics_dict, spikeamp=False, logscale=True)
    fig.canvas.set_window_title('fig5d right')
    return fig


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-i', '--inputdir', type=str, help='Input directory')
    ap.add_argument('-f', '--figset', type=str, nargs='+', help='Figure set', default='a')
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

    # Parameters
    a = 32e-9  # m
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    freqs = np.array([20e3, 100e3, 500e3, 1e6, 2e6, 3e6, 4e6])  # Hz
    Fdrive = 500e3  # Hz
    amps = np.array([50, 100, 300, 600]) * 1e3  # Pa
    Adrive = 100e3  # Pa
    PRFs_sparse = np.array([1e1, 1e2, 1e3, 1e4])  # Hz
    PRFs_dense = sum([[x, 2 * x, 5 * x] for x in PRFs_sparse[:-1]], []) + [PRFs_sparse[-1]]  # Hz
    PRF = 100  # Hz
    DCs = np.array([5, 10, 25, 50, 75, 100]) * 1e-2
    DC = 0.05

    # Get threshold amplitudes if needed
    if 'a' in figset or 'b' in figset:
        CW_Athrs = getCWtitrations(['RS'], a, freqs, tstim, toffset,
                                   os.path.join(inputdir, 'CW_Athrs.csv'))

    # Generate figures
    figs = []
    if 'a' in figset:
        figs.append(fig5aleft('RS', a, Fdrive, CW_Athrs, tstim, toffset, inputdir))
        figs.append(fig5aright('RS', a, Fdrive, amps, tstim, toffset, inputdir))
    if 'b' in figset:
        figs.append(fig5bleft('RS', a, [freqs.min(), freqs.max()], CW_Athrs, tstim, toffset,
                              inputdir))
        figs.append(fig5bright('RS', a, freqs, CW_Athrs, tstim, toffset, inputdir))
    if 'c' in figset:
        figs.append(fig5cleft(['RS', 'LTS'], a, Fdrive, Adrive, tstim, toffset, PRF, DC, inputdir))
        figs.append(fig5cright(['RS', 'LTS'], a, Fdrive, Adrive, tstim, toffset, PRF, DCs, inputdir))
    if 'd' in figset:
        figs.append(fig5dleft('LTS', a, Fdrive, Adrive, tstim, toffset, PRFs_sparse, DC, inputdir))
        figs.append(fig5dright('LTS', a, Fdrive, Adrive, tstim, toffset, PRFs_dense, DC, inputdir))

    if args.save:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(inputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
