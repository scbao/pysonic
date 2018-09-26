# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-23 14:55:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-26 16:51:55

''' Plotting utilities '''

import os
import pickle
import ntpath
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ..utils import getStimPulses, logger
from .pltvars import pltvars

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

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
        npatches, tpatch_on, tpatch_off = getStimPulses(t, states)
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
        sm = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('viridis'))
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
