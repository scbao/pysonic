# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-06-06 18:38:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-11-22 20:08:35


import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.utils import *
from PySONIC.neurons import *
from PySONIC.batches import createAStimQueue

from utils import *

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'


def fig6a(neurons, a, Fdrive, CW_Athrs, tstim, toffset, inputdir, width=0.35, fs=8):
    ''' Plot comparative bar chart of computation rates for the different regimes. '''

    regimes = ['thr - 5 kPa', 'thr + 20 kPa']

    # Get filepaths
    sonic_fpaths = {r: [] for r in regimes}
    full_fpaths = {r: [] for r in regimes}
    for neuron in neurons:
        full_fpaths[neuron], sonic_fpaths[neuron] = {}, {}
        subdir = os.path.join(inputdir, neuron)
        Athr = CW_Athrs.loc[Fdrive * 1e-3, neuron] * 1e3  # Pa
        sonic_fpaths[neuron]['thr - 5 kPa'] = getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Athr - 5e3], [tstim], [toffset], [None], [1.], 'sonic'))
        sonic_fpaths[neuron]['thr + 20 kPa'] = getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Athr + 20e3], [tstim], [toffset], [None], [1.], 'sonic'))
        full_fpaths[neuron]['thr - 5 kPa'] = getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Athr - 5e3], [tstim], [toffset], [None], [1.], 'full'))
        full_fpaths[neuron]['thr + 20 kPa'] = getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Athr + 20e3], [tstim], [toffset], [None], [1.], 'full'))
    data_fpaths = {'full': full_fpaths, 'sonic': sonic_fpaths}

    # Extract computation rates (s comp / ms stimulus)
    comptimes_fpaths = {x: os.path.join(inputdir, 'comptimes_vs_regimes_{}.csv'.format(x))
                        for x in ['sonic', 'full']}
    comptimes = getCompTimesQual(
        inputdir, neurons, regimes, data_fpaths, comptimes_fpaths)
    full_comptimes = comptimes['full']
    sonic_comptimes = comptimes['sonic']

    full_comprates = tstim * 1e3 / full_comptimes  # ms simulation / s computation
    sonic_comprates = tstim * 1e3 / sonic_comptimes  # ms simulation / s computation

    # Comutation means and standard deviations
    mu_full_comprates = [full_comprates[reg].mean() for reg in regimes]
    std_full_comprates = [full_comprates[reg].std() for reg in regimes]
    mu_sonic_comprates = [sonic_comprates[reg].mean() for reg in regimes]
    std_sonic_comprates = [sonic_comprates[reg].std() for reg in regimes]

    # Plot
    fig, ax = plt.subplots(figsize=cm2inch(5.5, 5.8))
    plt.subplots_adjust(bottom=0.2, left=0.25, right=0.95, top=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel('Comp. rate (ms stimulus / s)', fontsize=fs)
    ax.set_yscale('log')
    ax.set_ylim((1e-4, 1e2))
    ax.get_yaxis().set_tick_params(which='minor', size=0)
    ax.get_yaxis().set_tick_params(which='minor', width=0)
    ind = np.arange(len(regimes))
    ax.bar(ind, mu_full_comprates, width, color='silver',
           yerr=std_full_comprates, label='full')
    ax.bar(ind + width, mu_sonic_comprates, width, color='dimgray',
           yerr=std_sonic_comprates, label='sonic')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(regimes)
    for item in ax.get_yticklabels():
            item.set_fontsize(fs)
    for item in ax.get_xticklabels():
        item.set_fontsize(fs)
    fig.canvas.set_window_title('fig6a')
    return fig


def fig6b(neuron, a, Fdrive, amps, tstim, toffset, inputdir, fs=8, lw=2, ps=4):
    ''' Plot comparative bar chart of computation rates for different supra-threshold amplitudes. '''

    # Get filepaths
    xlabel = 'Amplitude (kPa)'
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths = getSims(subdir, neuron, a, createAStimQueue(
        [Fdrive], amps, [tstim], [toffset], [None], [1.], 'sonic'))
    full_fpaths = getSims(subdir, neuron, a, createAStimQueue(
        [Fdrive], amps, [tstim], [toffset], [None], [1.], 'full'))
    data_fpaths = {'full': full_fpaths, 'sonic': sonic_fpaths}

    # Extract computation rates (s comp / ms stimulus)
    comptimes_fpath = os.path.join(inputdir, '{}_comptimes_vs_amps.csv'.format(neuron))
    comptimes = getCompTimesQuant(
        inputdir, neuron, amps * 1e-3, xlabel, data_fpaths, comptimes_fpath)
    comprates = tstim * 1e3 / comptimes  # ms simulation / s computation

    # Plot comparative profiles of computation rate vs. amplitude
    fig, ax = plt.subplots(figsize=cm2inch(5.5, 5.8))
    plt.subplots_adjust(bottom=0.2, left=0.25, right=0.95, top=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=fs, labelpad=1)
    ax.set_ylabel('Comp. rate (ms sim / s comp)', fontsize=fs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim((1e-4, 1e2))
    ax.get_yaxis().set_tick_params(which='minor', size=0)
    ax.get_yaxis().set_tick_params(which='minor', width=0)
    colors = ['silver', 'dimgrey']
    for i, key in enumerate(comprates):
        ax.plot(amps * 1e-3, comprates[key], 'o--', color=colors[i],
                linewidth=lw, label=key, markersize=ps)
    for item in ax.get_yticklabels():
            item.set_fontsize(fs)
    for item in ax.get_xticklabels():
        item.set_fontsize(fs)
    fig.canvas.set_window_title('fig6b')
    return fig


def fig6c(neuron, a, freqs, CW_Athrs, tstim, toffset, inputdir, fs=8, lw=2, ps=4):
    ''' Plot comparative bar chart of computation rates for different US frequencies. '''

    # Get filepaths
    xlabel = 'Frequency (kHz)'
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths, full_fpaths = [], []
    for Fdrive in freqs:
        Athr = CW_Athrs[neuron].loc[Fdrive * 1e-3]  # kPa
        Adrive = (Athr + 20.) * 1e3  # Pa
        sonic_fpaths += getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'sonic'))
        full_fpaths += getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'full'))
    data_fpaths = {'full': full_fpaths, 'sonic': sonic_fpaths}

    # Extract computation rates (s comp / ms stimulus)
    comptimes_fpath = os.path.join(inputdir, '{}_comptimes_vs_freqs.csv'.format(neuron))
    comptimes = getCompTimesQuant(
        inputdir, neuron, freqs * 1e-3, xlabel, data_fpaths, comptimes_fpath)
    comprates = tstim * 1e3 / comptimes  # ms simulation / s computation

    # Plot comparative profiles of computation rate vs. frequency
    fig, ax = plt.subplots(figsize=cm2inch(5.5, 5.8))
    plt.subplots_adjust(bottom=0.2, left=0.25, right=0.95, top=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=fs, labelpad=1)
    ax.set_ylabel('Comp. rate (ms sim / s comp)', fontsize=fs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim((1e-4, 1e2))
    ax.get_yaxis().set_tick_params(which='minor', size=0)
    ax.get_yaxis().set_tick_params(which='minor', width=0)
    colors = ['silver', 'dimgrey']
    for i, key in enumerate(comprates):
        ax.plot(freqs * 1e-3, comprates[key], 'o--', color=colors[i],
                linewidth=lw, label=key, markersize=ps)
    for item in ax.get_yticklabels():
            item.set_fontsize(fs)
    for item in ax.get_xticklabels():
        item.set_fontsize(fs)
    fig.canvas.set_window_title('fig6c')
    return fig


def fig6d(neurons, a, Fdrive, Adrive, tstim, toffset, PRF, DCs, inputdir, fs=8, lw=2, ps=4):

    xlabel = 'Duty cycle (%)'
    colors = list(plt.get_cmap('Paired').colors[:6])
    del colors[2:4]

    # Create figure
    fig, ax = plt.subplots(figsize=cm2inch(5.5, 5.8))
    plt.subplots_adjust(bottom=0.2, left=0.25, right=0.95, top=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=fs, labelpad=-7)
    ax.set_ylabel('Comp. rate (ms stimulus / s)', fontsize=fs)
    ax.set_xticks([DCs.min() * 1e2, DCs.max() * 1e2])
    ax.set_yscale('log')
    ax.set_ylim((1e-4, 1e2))
    ax.get_yaxis().set_tick_params(which='minor', size=0)
    ax.get_yaxis().set_tick_params(which='minor', width=0)


    # Loop through neurons
    for i, neuron in enumerate(neurons):
        # Get filepaths
        subdir = os.path.join(inputdir, neuron)
        sonic_fpaths = getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [PRF], DCs, 'sonic'))
        full_fpaths = getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [PRF], DCs, 'full'))
        sonic_fpaths = sonic_fpaths[1:] + [sonic_fpaths[0]]
        full_fpaths = full_fpaths[1:] + [full_fpaths[0]]
        data_fpaths = {'full': full_fpaths, 'sonic': sonic_fpaths}

        # Extract computation rates (s comp / ms stimulus)
        comptimes_fpath = os.path.join(inputdir, '{}_comptimes_vs_DC.csv'.format(neuron))
        comptimes = getCompTimesQuant(
            inputdir, neuron, DCs * 1e2, xlabel, data_fpaths, comptimes_fpath)
        comprates = tstim * 1e3 / comptimes  # ms simulation / s computation

        # Plot
        ax.plot(DCs * 1e2, comprates['full'], 'o--', color=colors[2 * i], linewidth=lw, markersize=ps)
        ax.plot(DCs * 1e2, comprates['sonic'], 'o--', color=colors[2 * i + 1], linewidth=lw,
                markersize=ps, label=neuron)

    fig.canvas.set_window_title('fig6d')
    return fig


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
        figset = ['a', 'b', 'c', 'd']

    # Parameters
    a = 32e-9  # m
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    freqs = np.array([20e3, 100e3, 500e3, 1e6, 2e6, 3e6, 4e6])  # Hz
    Fdrive = 500e3  # Hz
    amps = np.array([50, 100, 300, 600]) * 1e3  # Pa
    Adrive = 100e3  # Pa
    PRF = 100  # Hz
    DCs = np.array([5, 10, 25, 50, 75, 100]) * 1e-2


    # Get threshold amplitudes if needed
    if 'a' in figset or 'c' in figset:
        allneurons = ['RS', 'FS', 'LTS', 'RE', 'TC']
        CW_Athr_vs_Fdrive = getCWtitrations_vs_Fdrive(
            allneurons, a, [Fdrive], tstim, toffset, os.path.join(inputdir, 'CW_Athrs_vs_freqs.csv'))

    # Generate figures
    figs = []
    if 'a' in figset:
        figs.append(fig6a(allneurons, a, Fdrive, CW_Athr_vs_Fdrive, tstim, toffset, inputdir))
    if 'b' in figset:
        figs.append(fig6b('RS', a, Fdrive, amps, tstim, toffset, inputdir))
    if 'c' in figset:
        figs.append(fig6c('RS', a, freqs, CW_Athr_vs_Fdrive, tstim, toffset, inputdir))
    if 'd' in figset:
        figs.append(fig6d(['RS', 'LTS'], a, Fdrive, Adrive, tstim, toffset, PRF, DCs, inputdir))


    if args.save:
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(inputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
