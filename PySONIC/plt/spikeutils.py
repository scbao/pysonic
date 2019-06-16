# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-10-01 20:40:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-16 22:46:04

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from ..utils import *
from ..constants import *
from ..postpro import findPeaks
from .pltutils import cm2inch, figtitle


# Plot parameters
phaseplotvars = {
    'Vm': {
        'label': 'V_m\ (mV)',
        'dlabel': 'dV/dt\ (V/s)',
        'factor': 1e0,
        'lim': (-80.0, 50.0),
        'dfactor': 1e-3,
        'dlim': (-300, 700),
        'thr_amp': SPIKE_MIN_VAMP,
        'thr_prom': SPIKE_MIN_VPROM
    },
    'Qm': {
        'label': 'Q_m\ (nC/cm^2)',
        'dlabel': 'I\ (A/m^2)',
        'factor': 1e5,
        'lim': (-80.0, 50.0),
        'dfactor': 1e0,
        'dlim': (-2, 5),
        'thr_amp': SPIKE_MIN_QAMP,
        'thr_prom': SPIKE_MIN_QPROM
    }
}


def plotPhasePlane(filepaths, varname, no_offset=False, no_first=False,
                   labels=None, colors=None, fs=15, lw=2, tbounds=None, pretty=True):
    ''' Plot phase-plane diagrams of spiking dynamics from simulation results.

        :param filepaths: list of full paths to data files
        :param varname: name of output variable of interest ('Qm' or Vm')
        :param no_offset: boolean stating whether or not to discard post-offset spikes
        :param no_first: boolean stating whether or not to discard first spike
        :param tbounds: spike interval bounds (ms)
        :return: figure handle
    '''

    # Preprocess parameters
    if tbounds is None:
        tbounds = (-1.5, 1.5)

    pltvar = phaseplotvars[varname]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 1st axis: variable as function of time
    ax = axes[0]
    ax.set_xlabel('$\\rm time\ (ms)$', fontsize=fs)
    ax.set_ylabel('$\\rm {}$'.format(pltvar['label']), fontsize=fs)
    ax.set_xlim(tbounds)
    ax.set_ylim(pltvar['lim'])
    if pretty:
        ax.set_xticks(tbounds)
        ax.set_yticks(pltvar['lim'])
        ax.set_xticklabels(['{:+.1f}'.format(x) for x in ax.get_xticks()])
        ax.set_yticklabels(['{:+.0f}'.format(x) for x in ax.get_yticks()])

    # 2nd axis: phase plot (derivative of variable vs variable)
    ax = axes[1]
    ax.set_xlabel('$\\rm {}$'.format(pltvar['label']), fontsize=fs)
    ax.set_ylabel('$\\rm {}$'.format(pltvar['dlabel']), fontsize=fs)
    ax.set_xlim(pltvar['lim'])
    ax.set_ylim(pltvar['dlim'])
    ax.plot([0, 0], [pltvar['dlim'][0], pltvar['dlim'][1]], '--', color='k', linewidth=1)
    ax.plot([pltvar['lim'][0], pltvar['lim'][1]], [0, 0], '--', color='k', linewidth=1)
    if pretty:
        ax.set_xticks(pltvar['lim'])
        ax.set_yticks(pltvar['dlim'])
        ax.set_xticklabels(['{:+.0f}'.format(x) for x in ax.get_xticks()])
        ax.set_yticklabels(['{:+.0f}'.format(x) for x in ax.get_yticks()])

    for ax in axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)

    handles = []
    autolabels = []

    # For each file
    for i, filepath in enumerate(filepaths):

        # Load data
        logger.info('loading data from file "{}"'.format(filepath))
        with open(filepath, 'rb') as fh:
            frame = pickle.load(fh)
        df = frame['data']
        meta = frame['meta']

        tstim = meta['tstim']
        t = df['t'].values
        y = df[varname].values
        dt = t[1] - t[0]
        dydt = np.diff(y) / dt

        # Prominence-based spike detection
        ispikes, *_, ibounds = findPeaks(
            y,
            mph=pltvar['thr_amp'],
            # mpd=int(np.ceil(SPIKE_MIN_DT / dt)),
            mpp=pltvar['thr_prom']
        )

        if len(ispikes) > 0:
            # Discard potential irrelevant spikes
            if no_offset:
                ibounds_right = [x[1] for x in ibounds]
                inds = np.where(t[ibounds_right] < tstim)[0]
                ispikes = ispikes[inds]
                ibounds = ibounds[inds]
            if no_first:
                ispikes = ispikes[1:]
                ibounds = ibounds[1:]

        # Store spikes in dedicated lists
        tspikes = []
        yspikes = []
        dydtspikes = []
        for ispike, ibound in zip(ispikes, ibounds):
            tmin = max(t[ibound[0]], tbounds[0] * 1e-3 + t[ispike])
            tmax = min(t[ibound[1]], tbounds[1] * 1e-3 + t[ispike])
            inds = np.where((t > tmin) & (t < tmax))[0]
            tspikes.append(t[inds] - t[ispike])
            yspikes.append(y[inds])
            dinds = np.hstack(([inds[0] - 1], inds, [inds[-1] + 1]))
            dydt = np.diff(y[dinds]) / np.diff(t[dinds])
            dydtspikes.append((dydt[:-1] + dydt[1:]) / 2)

        if len(tspikes) == 0:
            logger.warning('No spikes detected')
        else:
            # Plot spikes temporal profiles and phase-plane diagrams
            for j in range(len(tspikes)):
                if colors is None:
                    color = 'C{}'.format(i if len(filepaths) > 1 else j % 10)
                else:
                    color = colors[i]
                lh = axes[0].plot(tspikes[j] * 1e3, yspikes[j] * pltvar['factor'],
                                  linewidth=lw, c=color)[0]
                axes[1].plot(yspikes[j] * pltvar['factor'], dydtspikes[j] * pltvar['dfactor'],
                             linewidth=lw, c=color)

            # Populate legend
            handles.append(lh)
            autolabels.append(figtitle(meta))

    fig.tight_layout()

    if labels is None:
        labels = autolabels

    # Add legend
    fig.subplots_adjust(top=0.8)
    if len(filepaths) > 1:
        axes[0].legend(handles, labels, fontsize=fs, frameon=False,
                       loc='upper center', bbox_to_anchor=(1.0, 1.35))
    else:
        fig.suptitle(labels[0], fontsize=fs)

    # Return
    return fig



def plotSpikingMetrics(xvar, xlabel, metrics_dict, logscale=False, spikeamp=True, colors=None,
                       fs=8, lw=2, ps=4, figsize=cm2inch(7.25, 5.8)):
    ''' Plot the evolution of key spiking metrics as function of a specific stimulation parameter. '''

    ls = {'full': 'o-', 'sonic': 'o--'}
    cdefault = {'full': 'silver', 'sonic': 'k'}

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    ibase = 0 if spikeamp else 1
    axes[ibase].set_ylabel('Latency\n (ms)', fontsize=fs, rotation=0, ha='right', va='center')
    axes[ibase + 1].set_ylabel('Firing\n rate (Hz)', fontsize=fs, rotation=0, ha='right', va='center')
    if spikeamp:
        axes[2].set_ylabel('Spike amp.\n ($\\rm nC/cm^2$)', fontsize=fs, rotation=0, ha='right',
                           va='center')
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if logscale:
            ax.set_xscale('log')
        for item in ax.get_yticklabels():
            item.set_fontsize(fs)
    for ax in axes[:-1]:
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        plt.setp(ax.get_xticklabels(minor=True), visible=False)
        ax.get_xaxis().set_tick_params(which='minor', size=0)
        ax.get_xaxis().set_tick_params(which='minor', width=0)
    axes[-1].set_xlabel(xlabel, fontsize=fs)
    if not logscale:
        axes[-1].set_xticks([min(xvar), max(xvar)])
    for item in axes[-1].get_xticklabels():
        item.set_fontsize(fs)

    # Plot metrics for each neuron
    for i, neuron in enumerate(metrics_dict.keys()):
        full_metrics = metrics_dict[neuron]['full']
        sonic_metrics = metrics_dict[neuron]['sonic']
        c = colors[neuron] if colors is not None else cdefault

        # Latency
        rf = 10
        ax = axes[ibase]
        ax.plot(xvar, full_metrics['latencies (ms)'].values, ls['full'], color=c['full'],
                linewidth=lw, markersize=ps)
        ax.plot(xvar, sonic_metrics['latencies (ms)'].values, ls['sonic'], color=c['sonic'],
                linewidth=lw, markersize=ps, label=neuron)

        # Firing rate
        rf = 10
        ax = axes[ibase + 1]
        ax.errorbar(xvar, full_metrics['mean firing rates (Hz)'].values,
                    yerr=full_metrics['std firing rates (Hz)'].values,
                    fmt=ls['full'], color=c['full'], linewidth=lw, markersize=ps)
        ax.errorbar(xvar, sonic_metrics['mean firing rates (Hz)'].values,
                    yerr=sonic_metrics['std firing rates (Hz)'].values,
                    fmt=ls['sonic'], color=c['sonic'], linewidth=lw, markersize=ps)

        # Spike amplitudes
        if spikeamp:
            ax = axes[2]
            rf = 10
            ax.errorbar(xvar, full_metrics['mean spike amplitudes (nC/cm2)'].values,
                        yerr=full_metrics['std spike amplitudes (nC/cm2)'].values,
                        fmt=ls['full'], color=c['full'], linewidth=lw, markersize=ps)
            ax.errorbar(xvar, sonic_metrics['mean spike amplitudes (nC/cm2)'].values,
                        yerr=sonic_metrics['std spike amplitudes (nC/cm2)'].values,
                        fmt=ls['sonic'], color=c['sonic'], linewidth=lw, markersize=ps)

    # Adapt axes y-limits
    rf = 10
    for ax in axes:
        ax.set_ylim([np.floor(ax.get_ylim()[0] / rf) * rf, np.ceil(ax.get_ylim()[1] / rf) * rf])
        ax.set_yticks([max(ax.get_ylim()[0], 0), ax.get_ylim()[1]])

    # Legend
    if len(metrics_dict.keys()) > 1:
        leg = axes[0].legend(fontsize=fs, frameon=False, bbox_to_anchor=(0., 0.9, 1., .102),
                             loc=8, ncol=2, borderaxespad=0.)
        for l in leg.get_lines():
            l.set_linestyle('-')

    fig.subplots_adjust(hspace=.3, bottom=0.2, left=0.35, right=0.95, top=0.95)
    return fig
