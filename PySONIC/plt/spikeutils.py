# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-10-01 20:40:28
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-25 14:48:28


import pickle
import numpy as np
import matplotlib.pyplot as plt

from ..utils import cm2inch, logger, figtitle
from ..constants import *
from ..postpro import findPeaks

# Plot parameters
phaseplotvars = {
    'Vm': {
        'label': 'V_m\ (mV)',
        'dlabel': 'dV/dt\ (V/s)',
        'factor': 1e0,
        'lim': (-80.0, 50.0),
        'dfactor': 1e-3,
        'dlim': (-200, 400),
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


def plotPhasePlane(filepath, varname, no_offset=False, no_first=False,
                   title=None, label=None, fs=15, lw=2):
    ''' Plot phase-plane diagrams of spiking dynamics from simulation results.

        :param filepath: full path to data file
        :param varname: name of output variable of interest ('Qm' or Vm')
        :return: figure handle
    '''

    pltvar = phaseplotvars[varname]
    dt_spike = 3e-3  # s
    tbounds = np.array([-dt_spike / 2, dt_spike / 2])  # s

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 1st axis: variable as function of time
    ax = axes[0]
    ax.set_xlabel('$\\rm time\ (ms)$', fontsize=fs)
    ax.set_ylabel('$\\rm {}$'.format(pltvar['label']), fontsize=fs)
    ax.set_xlim(tbounds * 1.05 * 1e3)
    ax.set_xticks(tbounds * 1e3)
    ax.set_ylim(pltvar['lim'])
    ax.set_yticks(pltvar['lim'])
    ax.set_xticklabels(['{:+.1f}'.format(x) for x in ax.get_xticks()])
    ax.set_yticklabels(['{:+.0f}'.format(x) for x in ax.get_yticks()])

    # 2nd axis: phase plot (derivative of variable vs variable)
    ax = axes[1]
    ax.set_xlabel('$\\rm {}$'.format(pltvar['label']), fontsize=fs)
    ax.set_ylabel('$\\rm {}$'.format(pltvar['dlabel']), fontsize=fs)
    ax.set_xlim(pltvar['lim'])
    ax.set_xticks(pltvar['lim'])
    ax.set_ylim(pltvar['dlim'])
    ax.set_yticks(pltvar['dlim'])
    ax.plot([0, 0], [pltvar['dlim'][0], pltvar['dlim'][1]], '--', color='k', linewidth=1)
    ax.plot([pltvar['lim'][0], pltvar['lim'][1]], [0, 0], '--', color='k', linewidth=1)
    ax.set_xticklabels(['{:+.0f}'.format(x) for x in ax.get_xticks()])
    ax.set_yticklabels(['{:+.0f}'.format(x) for x in ax.get_yticks()])

    for ax in axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)

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

    # Detect spikes on charge profile
    mpd = int(np.ceil(SPIKE_MIN_DT / dt))
    ispikes, prominences, widths, ibounds = findPeaks(
        y, pltvar['thr_amp'], mpd, pltvar['thr_prom'])

    # Discard potential irrelevant spikes
    if no_offset:
        ispikes = ispikes[t[ispikes] < tstim]  # discard post-offset spikes
    if no_first:
        ispikes = ispikes[1:]

    # Store spikes
    tspikes = []
    yspikes = []
    dydtspikes = []
    for ispike in ispikes:
        inds = np.where((t > t[ispike] + tbounds[0]) & (t < t[ispike] + tbounds[1]))[0]
        tspikes.append(t[inds] - t[ispike])
        yspikes.append(y[inds])
        dinds = np.hstack(([inds[0] - 1], inds, [inds[-1] + 1]))
        dydt = np.diff(y[dinds]) / np.diff(t[dinds])
        dydtspikes.append((dydt[:-1] + dydt[1:]) / 2)

    # Plot spikes temporal profiles
    for tspike, yspike in zip(tspikes, yspikes):
        axes[0].plot(tspike * 1e3, yspike * pltvar['factor'], linewidth=lw, color='C0')

    # Plot spikes phase-plane-diagrams
    for yspike, dydtspike in zip(yspikes, dydtspikes):
        axes[1].plot(yspike * pltvar['factor'], dydtspike * pltvar['dfactor'], linewidth=lw,
                     color='C0')

    if title is None:
        title = figtitle(meta)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.suptitle(title, fontsize=fs)
    # if label is None:
    #     label = figtitle(meta)
    # ax.legend([lh], [label], fontsize=fs, frameon=False, bbox_to_anchor=(0.6, 1.25))

    return fig



def plotSpikingMetrics(xvar, xlabel, metrics_dict, logscale=False, spikeamp=True, colors=None,
                       fs=8, lw=2, ps=4, figsize=cm2inch(7.25, 5.8)):
    ''' Plot the evolution of key spiking metrics as function of a specific stimulation parameter. '''

    ls = {'full': 'o-', 'sonic': 'o--'}
    cdefault = {'full': 'silver', 'sonic': 'k'}

    # Create figure
    naxes = 3 if spikeamp else 2
    fig, axes = plt.subplots(naxes, 1, figsize=figsize)
    axes[0].set_ylabel('Latency\n (ms)', fontsize=fs, rotation=0, ha='right', va='center')
    axes[1].set_ylabel('Firing\n rate (Hz)', fontsize=fs, rotation=0, ha='right', va='center')
    if naxes == 3:
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
        ax = axes[0]
        ax.plot(xvar, full_metrics['latencies (ms)'].values, ls['full'], color=c['full'],
                linewidth=lw, markersize=ps)
        ax.plot(xvar, sonic_metrics['latencies (ms)'].values, ls['sonic'], color=c['sonic'],
                linewidth=lw, markersize=ps, label=neuron)

        # Firing rate
        rf = 10
        ax = axes[1]
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
