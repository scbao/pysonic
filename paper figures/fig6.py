# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-06-06 18:38:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-11-26 21:38:18


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


def fig6a(neuron, a, Fdrive, amps, tstim, toffset, inputdir, fs=8, lw=2, ps=4):
    ''' Plot comparative bar chart of computation rates for different amplitudes. '''

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
    tcomp_lookup = getLookupsCompTime(neuron)

    # Plot comparative profiles of computation rate vs. amplitude
    fig, ax = plt.subplots(figsize=cm2inch(5.5, 5.8))
    plt.subplots_adjust(bottom=0.2, left=0.25, right=0.95, top=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=fs, labelpad=1)
    ax.set_ylabel('Computation time', fontsize=fs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    yticks = np.array([1, 60, 60**2, 60**2 * 24, 60**2 * 24 * 7])
    yticklabels = ['1 s', '1 min', '1 hour', '1 day', '1 week']
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim((1e-1, 1e6))
    ax.get_yaxis().set_tick_params(which='minor', size=0)
    ax.get_yaxis().set_tick_params(which='minor', width=0)
    ax.axhline(tcomp_lookup, color='k', linewidth=lw)
    colors = ['silver', 'dimgrey']
    for i, key in enumerate(comptimes):
        ax.plot(amps * 1e-3, comptimes[key], 'o--', color=colors[i],
                linewidth=lw, label=key, markersize=ps)
    ax.plot(amps * 1e-3, comptimes['sonic'] + tcomp_lookup, 'o--', color='k',
            linewidth=lw, label=key, markersize=ps)
    for item in ax.get_yticklabels():
            item.set_fontsize(fs)
    for item in ax.get_xticklabels():
        item.set_fontsize(fs)
    fig.canvas.set_window_title('fig6b')
    return fig


def fig6b(neuron, a, freqs, CW_Athrs, tstim, toffset, inputdir, fs=8, lw=2, ps=4):
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
    tcomp_lookup = getLookupsCompTime(neuron)

    # Plot comparative profiles of computation rate vs. frequency
    fig, ax = plt.subplots(figsize=cm2inch(5.5, 5.8))
    plt.subplots_adjust(bottom=0.2, left=0.25, right=0.95, top=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=fs, labelpad=1)
    ax.set_ylabel('Computation time', fontsize=fs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    yticks = np.array([1, 60, 60**2, 60**2 * 24, 60**2 * 24 * 7])
    yticklabels = ['1 s', '1 min', '1 hour', '1 day', '1 week']
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim((1e-1, 1e6))
    ax.get_yaxis().set_tick_params(which='minor', size=0)
    ax.get_yaxis().set_tick_params(which='minor', width=0)
    ax.axhline(tcomp_lookup, color='k', linewidth=lw)
    colors = ['silver', 'dimgrey']
    for i, key in enumerate(comptimes):
        ax.plot(freqs * 1e-3, comptimes[key], 'o--', color=colors[i],
                linewidth=lw, label=key, markersize=ps)
    for item in ax.get_yticklabels():
            item.set_fontsize(fs)
    for item in ax.get_xticklabels():
        item.set_fontsize(fs)
    fig.canvas.set_window_title('fig6c')
    return fig


def fig6c(neuron, radii, Fdrive, CW_Athrs, tstim, toffset, inputdir, fs=8, lw=2, ps=4):
    ''' Plot comparative bar chart of computation rates for different sonophore radii. '''

    # Get filepaths
    xlabel = 'Sonophore radius (nm)'
    subdir = os.path.join(inputdir, neuron)
    sonic_fpaths, full_fpaths = [], []
    for a in radii:
        Athr = CW_Athrs[neuron].loc[np.round(a * 1e9, 1)]  # kPa
        Adrive = (Athr + 20.) * 1e3  # Pa
        sonic_fpaths += getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'sonic'))
        full_fpaths += getSims(subdir, neuron, a, createAStimQueue(
            [Fdrive], [Adrive], [tstim], [toffset], [None], [1.], 'full'))
    data_fpaths = {'full': full_fpaths, 'sonic': sonic_fpaths}

    # Extract computation rates (s comp / ms stimulus)
    comptimes_fpath = os.path.join(inputdir, '{}_comptimes_vs_radius.csv'.format(neuron))
    comptimes = getCompTimesQuant(
        inputdir, neuron, radii * 1e9, xlabel, data_fpaths, comptimes_fpath)
    tcomp_lookup = getLookupsCompTime(neuron)

    # Plot comparative profiles of computation rate vs. frequency
    fig, ax = plt.subplots(figsize=cm2inch(5.5, 5.8))
    plt.subplots_adjust(bottom=0.2, left=0.25, right=0.95, top=0.95)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel(xlabel, fontsize=fs, labelpad=1)
    ax.set_ylabel('Computation time', fontsize=fs)
    ax.set_xscale('log')
    ax.set_yscale('log')
    yticks = np.array([1, 60, 60**2, 60**2 * 24, 60**2 * 24 * 7])
    yticklabels = ['1 s', '1 min', '1 hour', '1 day', '1 week']
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim((1e-1, 1e6))
    ax.get_yaxis().set_tick_params(which='minor', size=0)
    ax.get_yaxis().set_tick_params(which='minor', width=0)
    ax.axhline(tcomp_lookup, color='k', linewidth=lw)
    colors = ['silver', 'dimgrey']
    for i, key in enumerate(comptimes):
        ax.plot(radii * 1e9, comptimes[key], 'o--', color=colors[i],
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
    ax.set_ylabel('Computation time', fontsize=fs)
    ax.set_xticks([DCs.min() * 1e2, DCs.max() * 1e2])
    ax.set_yscale('log')
    yticks = np.array([1, 60, 60**2, 60**2 * 24, 60**2 * 24 * 7])
    yticklabels = ['1 s', '1 min', '1 hour', '1 day', '1 week']
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim((1e-1, 1e6))
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

        # Plot
        ax.plot(DCs * 1e2, comptimes['full'], 'o--', color=colors[2 * i], linewidth=lw, markersize=ps)
        ax.plot(DCs * 1e2, comptimes['sonic'], 'o--', color=colors[2 * i + 1], linewidth=lw,
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
    radii = np.array([16, 22.6, 32, 45.3, 64]) * 1e-9  # nm
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    freqs = np.array([20e3, 100e3, 500e3, 1e6, 2e6, 3e6, 4e6])  # Hz
    Fdrive = 500e3  # Hz
    CW_Athr_vs_Fdrive = getCWtitrations_vs_Fdrive(
        ['RS'], a, freqs, tstim, toffset, os.path.join(inputdir, 'CW_Athrs_vs_freqs.csv'))
    Athr = CW_Athr_vs_Fdrive['RS'].loc[Fdrive * 1e-3]
    amps1 = np.array([Athr - 5, Athr, Athr + 20]) * 1e3
    amps2 = np.array([50, 100, 300, 600]) * 1e3  # Pa
    amps = np.sort(np.hstack([amps1, amps2]))

    CW_Athr_vs_radius = getCWtitrations_vs_radius(
        ['RS'], radii, Fdrive, tstim, toffset, os.path.join(inputdir, 'CW_Athrs_vs_radius.csv'))

    Adrive = 100e3  # Pa
    PRF = 100  # Hz
    DCs = np.array([5, 10, 25, 50, 75, 100]) * 1e-2

    # Generate figures
    figs = []
    if 'a' in figset:
        figs.append(fig6a('RS', a, Fdrive, amps, tstim, toffset, inputdir))
    if 'b' in figset:
        figs.append(fig6b('RS', a, freqs, CW_Athr_vs_Fdrive, tstim, toffset, inputdir))
    if 'c' in figset:
        figs.append(fig6c('RS', radii, Fdrive, CW_Athr_vs_radius, tstim, toffset, inputdir))
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
