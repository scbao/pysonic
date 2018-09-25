# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-23 14:55:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-25 17:20:05

''' Plotting utilities '''

import os
import pickle
import ntpath
import re
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter

from ..utils import si_format
from .pltvars import pltvars

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Get package logger
logger = logging.getLogger('PySONIC')

# Define global variables
neuron = None
bls = None
timeunits = {'ASTIM': 't_ms', 'ESTIM': 't_ms', 'MECH': 't_us'}

# Regular expression for input files
rgxp = re.compile('(ESTIM|ASTIM)_([A-Za-z]*)_(.*).pkl')
rgxp_mech = re.compile('(MECH)_(.*).pkl')


# Figure naming conventions
def ESTIM_title(name, A, t, PRF, DC):
    return '{} neuron: {} E-STIM {:.2f}mA/m2, {:.0f}ms{}'.format(
        name, 'PW' if DC < 1. else 'CW', A, t,
        ', {:.2f}Hz PRF, {:.0f}% DC'.format(PRF, DC) if DC < 1. else '')


def ASTIM_title(name, f, A, t, PRF, DC):
    return '{} neuron: {} A-STIM {:.0f}kHz {:.0f}kPa, {:.0f}ms{}'.format(
        name, 'PW' if DC < 1. else 'CW', f, A, t,
        ', {:.2f}Hz PRF, {:.0f}% DC'.format(PRF, DC) if DC < 1. else '')


def MECH_title(a, f, A):
    return '{:.0f}nm BLS structure: MECH-STIM {:.0f}kHz, {:.0f}kPa'.format(a, f, A)


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def computeMeshEdges(x, scale='lin'):
    ''' Compute the appropriate edges of a mesh that quads a linear or logarihtmic distribution.

        :param x: the input vector
        :param scale: the type of distribution ('lin' for linear, 'log' for logarihtmic)
        :return: the edges vector
    '''
    if scale is 'log':
        x = np.log10(x)
    dx = x[1] - x[0]
    if scale is 'lin':
        y = np.linspace(x[0] - dx / 2, x[-1] + dx / 2, x.size + 1)
    elif scale is 'log':
        y = np.logspace(x[0] - dx / 2, x[-1] + dx / 2, x.size + 1)
    return y


def getPatchesLoc(t, states):
    ''' Determine the location of stimulus patches.

        :param t: simulation time vector (s).
        :param states: a vector of stimulation state (ON/OFF) at each instant in time.
        :return: 3-tuple with number of patches, timing of STIM-ON an STIM-OFF instants.
    '''

    # Compute states derivatives and identify bounds indexes of pulses
    dstates = np.diff(states)
    ipatch_on = np.insert(np.where(dstates > 0.0)[0] + 1, 0, 0)
    ipatch_off = np.where(dstates < 0.0)[0] + 1
    if ipatch_off.size < ipatch_on.size:
        ioff = t.size - 1
        if ipatch_off.size == 0:
            ipatch_off = np.array([ioff])
        else:
            ipatch_off = np.insert(ipatch_off, ipatch_off.size - 1, ioff)

    # Get time instants for pulses ON and OFF
    npatches = ipatch_on.size
    tpatch_on = t[ipatch_on]
    tpatch_off = t[ipatch_off]

    # return 3-tuple with #patches, pulse ON and pulse OFF instants
    return (npatches, tpatch_on, tpatch_off)


def plotActivationMap(DCs, amps, actmap, FRlims, title=None, Ascale='log', FRscale='log', fs=8):
    ''' Plot a neuron's activation map over the amplitude x duty cycle 2D space.

        :param DCs: duty cycle vector
        :param amps: amplitude vector
        :param actmap: 2D activation matrix
        :param FRlims: lower and upper bounds of firing rate color-scale
        :param title: figure title
        :param Ascale: scale to use for the amplitude dimension ('lin' or 'log')
        :param FRscale: scale to use for the firing rate coloring ('lin' or 'log')
        :param fs: fontsize to use for the title and labels
        :return: 3-tuple with the handle to the generated figure and the mesh x and y coordinates
    '''

    # Check firing rate bounding
    minFR, maxFR = (actmap[actmap > 0].min(), actmap.max())
    logger.info('FR range: %.0f - %.0f Hz', minFR, maxFR)
    if minFR < FRlims[0]:
        logger.warning('Minimal firing rate (%.0f Hz) is below defined lower bound (%.0f Hz)',
                       minFR, FRlims[0])
    if maxFR > FRlims[1]:
        logger.warning('Maximal firing rate (%.0f Hz) is above defined upper bound (%.0f Hz)',
                       maxFR, FRlims[1])

    # Plot activation map
    if FRscale == 'lin':
        norm = matplotlib.colors.Normalize(*FRlims)
    elif FRscale == 'log':
        norm = matplotlib.colors.LogNorm(*FRlims)
    fig, ax = plt.subplots(figsize=cm2inch(8, 5.8))
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.8, top=0.92)
    if title is not None:
        ax.set_title(title, fontsize=fs)
    if Ascale == 'log':
        ax.set_yscale('log')
    ax.set_xlabel('Duty cycle (%)', fontsize=fs, labelpad=-0.5)
    ax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    ax.set_xlim(np.array([DCs.min(), DCs.max()]) * 1e2)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    xedges = computeMeshEdges(DCs)
    yedges = computeMeshEdges(amps, scale='log')
    actmap[actmap == -1] = np.nan
    actmap[actmap == 0] = 1e-3
    cmap = plt.get_cmap('viridis')
    cmap.set_bad('silver')
    cmap.set_under('k')
    ax.pcolormesh(xedges * 1e2, yedges * 1e-3, actmap, cmap=cmap, norm=norm)

    # Plot firing rate colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    pos1 = ax.get_position()  # get the map axis position
    cbarax = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.03, pos1.height])
    fig.colorbar(sm, cax=cbarax)
    cbarax.set_ylabel('Firing rate (Hz)', fontsize=fs)
    for item in cbarax.get_yticklabels():
        item.set_fontsize(fs)

    return (fig, xedges, yedges)


def plotRawTrace(fpath, key, ybounds):
    '''  Plot the raw signal of a given variable within specified bounds.

        :param foath: full path to the data file
        :param key: key to the target variable
        :param ybounds: y-axis bounds
        :return: handle to the generated figure
    '''

    # Check file existence
    fname = ntpath.basename(fpath)
    if not os.path.isfile(fpath):
        raise FileNotFoundError('Error: "{}" file does not exist'.format(fname))

    # Load data
    logger.debug('Loading data from "%s"', fname)
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
        df = frame['data']
    t = df['t'].values
    y = df[key].values * pltvars[key]['factor']

    Δy = y.max() - y.min()
    logger.info('d%s = %.1f %s', key, Δy, pltvars[key]['unit'])

    # Plot trace
    fig, ax = plt.subplots(figsize=cm2inch(12.5, 5.8))
    fig.canvas.set_window_title(fname)
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(ybounds)
    ax.plot(t, y, color='k', linewidth=1)
    fig.tight_layout()

    return fig


def plotTraces(fpath, keys, tbounds):
    '''  Plot the raw signal of sevral variables within specified bounds.

        :param foath: full path to the data file
        :param key: key to the target variable
        :param tbounds: x-axis bounds
        :return: handle to the generated figure
    '''

    # Check file existence
    fname = ntpath.basename(fpath)
    if not os.path.isfile(fpath):
        raise FileNotFoundError('Error: "{}" file does not exist'.format(fname))

    # Load data
    logger.debug('Loading data from "%s"', fname)
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
        df = frame['data']
    t = df['t'].values * 1e3

    # Plot trace
    fs = 8
    fig, ax = plt.subplots(figsize=cm2inch(7, 3))
    fig.canvas.set_window_title(fname)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_position(('axes', -0.03))
        ax.spines[s].set_linewidth(2)
    ax.yaxis.set_tick_params(width=2)

    # ax.spines['bottom'].set_linewidth(2)
    ax.set_xlim(tbounds)
    ax.set_xticks([])
    ymin = np.nan
    ymax = np.nan
    dt = tbounds[1] - tbounds[0]
    ax.set_xlabel('{}s'.format(si_format(dt * 1e-3, space=' ')), fontsize=fs)
    ax.set_ylabel('mV - $\\rm nC/cm^2$', fontsize=fs, labelpad=-15)

    colors = {'Vm': 'darkgrey', 'Qm': 'k'}
    for key in keys:
        y = df[key].values * pltvars[key]['factor']
        ymin = np.nanmin([ymin, y.min()])
        ymax = np.nanmax([ymax, y.max()])
        # if key == 'Qm':
            # y0 = y[0]
            # ax.plot(t, y0 * np.ones(t.size), '--', color='k', linewidth=1)
        Δy = y.max() - y.min()
        logger.info('d%s = %.1f %s', key, Δy, pltvars[key]['unit'])
        ax.plot(t, y, color=colors[key], linewidth=1)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # ax.set_yticks([ymin, ymax])
    ax.set_ylim([-200, 100])
    ax.set_yticks([-200, 100])
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)
    # fig.tight_layout()
    return fig


def plotSignals(t, signals, states=None, ax=None, onset=None, lbls=None, fs=10, cmode='qual'):
    ''' Plot several signals on one graph.

        :param t: time vector
        :param signals: list of signal vectors
        :param states (optional): stimulation state vector
        :param ax (optional): handle to figure axis
        :param onset (optional): onset to add to signals on graph
        :param lbls (optional): list of legend labels
        :param fs (optional): font size to use on graph
        :param cmode: color mode ('seq' for sequentiual or 'qual' for qualitative)
        :return: figure handle (if no axis provided as input)
    '''


    # If no axis provided, create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        argout = fig
    else:
        argout = None

    # Set axis aspect
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)

    # Compute number of signals
    nsignals = len(signals)

    # Adapt labels for sequential color mode
    if cmode == 'seq' and lbls is not None:
        lbls[1:-1] = ['.'] * (nsignals - 2)

    # Add stimulation patches if states provided
    if states is not None:
        npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)
        for i in range(npatches):
            ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                       facecolor='#8A8A8A', alpha=0.2)

    # Add onset of provided
    if onset is not None:
        t0, y0 = onset
        t = np.hstack((np.array([t0, 0.]), t))
        signals = np.hstack((np.ones((nsignals, 2)) * y0, signals))

    # Determine colorset
    nlevels = nsignals
    if cmode == 'seq':
        norm = matplotlib.colors.Normalize(0, nlevels - 1)
        sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('viridis'))
        sm._A = []
        colors = [sm.to_rgba(i) for i in range(nlevels)]
    elif cmode == 'qual':
        nlevels_max = 10
        if nlevels > nlevels_max:
            raise Warning('Number of signals higher than number of color levels')
        colors = ['C{}'.format(i) for i in range(nlevels)]
    else:
        raise ValueError('Unknown color mode')

    # Plot signals
    for i, var in enumerate(signals):
        ax.plot(t, var, label=lbls[i] if lbls is not None else None, c=colors[i])

    # Add legend
    if lbls is not None:
        ax.legend(fontsize=fs, frameon=False)

    return argout
