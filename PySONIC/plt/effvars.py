# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-10-02 01:44:59
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-11-19 19:58:51

import numpy as np
from scipy.interpolate import interp1d, interp2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from PySONIC.plt import pltvars
from PySONIC.utils import logger, getLookups2D


def setGrid(n, ncolmax=3):
    ''' Determine number of rows and columns in figure grid, based on number of
        variables to plot. '''
    if n <= ncolmax:
        return (1, n)
    else:
        return ((n - 1) // ncolmax + 1, ncolmax)


def plotEffectiveVariables(neuron, a, Fdrive, amps=None, fs=12, ncolmax=2):
    ''' Plot the profiles of effective variables of a specific neuron for a given frequency.
        For each variable, one line chart per amplitude is plotted, using charge as the
        input variable on the abscissa and a linear color code for the amplitude value.

        :param neuron: channel mechanism object
        :param a: sonophore diameter (m)
        :param Fdrive: acoustic drive frequency (Hz)
        :param amps: vector of amplitudes at which variables must be plotted (Pa)
        :param fs: figure fontsize
        :param ncolmax: max number of columns on the figure
        :return: handle to the created figure
    '''

    # Get 2D lookups at specific (a, Fdrive) combination
    Aref, Qref, lookups2D = getLookups2D(neuron.name, a, Fdrive)
    lookups2D['Vm'] = lookups2D.pop('V')
    keys = ['Vm'] + list(lookups2D.keys())[:-1]

    #  Define log-amplitude color code
    if amps is None:
        amps = Aref
    mymap = cm.get_cmap('Oranges')
    norm = matplotlib.colors.LogNorm(amps.min(), amps.max())
    sm = cm.ScalarMappable(norm=norm, cmap=mymap)
    sm._A = []

    # Plot
    logger.info('plotting')
    nrows, ncols = setGrid(len(lookups2D), ncolmax=ncolmax)
    xvar = pltvars['Qm']
    Qbounds = np.array([Qref.min(), Qref.max()]) * xvar['factor']

    fig, _ = plt.subplots(figsize=(3 * ncols, 1 * nrows), squeeze=False)
    for j, key in enumerate(keys):
        ax = plt.subplot2grid((nrows, ncols), (j // ncols, j % ncols))
        for s in ['right', 'top']:
            ax.spines[s].set_visible(False)
        yvar = pltvars[key]
        if j // ncols == nrows - 1:
            ax.set_xlabel('$\\rm {}\ ({})$'.format(xvar['label'], xvar['unit']), fontsize=fs)
            ax.set_xticks(Qbounds)
        else:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)

        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.yaxis.set_label_coords(-0.02, 0.5)

        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)

        ymin = np.inf
        ymax = -np.inf

        # Plot effective variable for each selected amplitude
        y0 = np.squeeze(interp2d(Aref, Qref, lookups2D[key].T)(0, Qref))
        for Adrive in amps:
            y = np.squeeze(interp2d(Aref, Qref, lookups2D[key].T)(Adrive, Qref))
            if 'alpha' in key or 'beta' in key:
                y[y > y0.max() * 2] = np.nan
            ax.plot(Qref * xvar['factor'], y * yvar['factor'], c=sm.to_rgba(Adrive))
            ymin = min(ymin, y.min())
            ymax = max(ymax, y.max())

        # Plot reference variable
        ax.plot(Qref * xvar['factor'], y0 * yvar['factor'], '--', c='k')
        ymax = max(ymax, y0.max())
        ymin = min(ymin, y0.min())

        # Set axis y-limits
        if 'alpha' in key or 'beta' in key:
            ymax = y0.max() * 2
        ylim = [ymin * yvar['factor'], ymax * yvar['factor']]
        if key == 'ng':
            ylim = [np.floor(ylim[0] * 1e2) / 1e2, np.ceil(ylim[1] * 1e2) / 1e2]
        else:
            factor = 1 / np.power(10, np.floor(np.log10(ylim[1])))
            ylim = [np.floor(ylim[0] * factor) / factor, np.ceil(ylim[1] * factor) / factor]
        dy = ylim[1] - ylim[0]
        ax.set_yticks(ylim)
        ax.set_ylim(ylim)
        # ax.set_ylim([ylim[0] - 0.05 * dy, ylim[1] + 0.05 * dy])

        # Annotate variable and unit
        xlim = ax.get_xlim()
        if np.argmax(y0) < np.argmin(y0):
            xtext = xlim[0] + 0.6 * (xlim[1] - xlim[0])
        else:
            xtext = xlim[0] + 0.01 * (xlim[1] - xlim[0])
        if key in ['Vm', 'ng']:
            ytext = ylim[0] + 0.85 * dy
        else:
            ytext = ylim[0] + 0.15 * dy
        ax.text(xtext, ytext, '$\\rm {}\ ({})$'.format(yvar['label'], yvar['unit']), fontsize=fs)

    fig.suptitle('{} neuron: original vs. effective variables @ {:.0f} kHz'.format(
        neuron.name, Fdrive * 1e-3))

    # Plot colorbar
    fig.subplots_adjust(left=0.10, bottom=0.05, top=0.9, right=0.85)
    cbarax = fig.add_axes([0.87, 0.05, 0.04, 0.85])
    fig.colorbar(sm, cax=cbarax)
    cbarax.set_ylabel('amplitude (Pa)', fontsize=fs)
    for item in cbarax.get_yticklabels():
        item.set_fontsize(fs)

    return fig



def plotEffectiveCapacitance(neuron, a, Fdrive, amps=None, fs=12):
    ''' Plot the profiles of effective membrane capacitance of a specific neuron for a given frequency.
        One line chart per amplitude is plotted, using charge as the input variable on the abscissa
        and a linear color code for the amplitude value.

        :param neuron: channel mechanism object
        :param a: sonophore diameter (m)
        :param Fdrive: acoustic drive frequency (Hz)
        :param amps: vector of amplitudes at which variables must be plotted (Pa)
        :param fs: figure fontsize
        :param ncolmax: max number of columns on the figure
        :return: handle to the created figure
    '''

    # Get 2D lookups at specific (a, Fdrive) combination
    Aref, Qref, lookups2D = getLookups2D(neuron.name, a, Fdrive)

    # Compute effective capacitance
    Cmeff = Qref / lookups2D.pop('V') * 1e5  # uF/cm2

    #  Define log-amplitude color code
    if amps is None:
        amps = Aref
    mymap = cm.get_cmap('Oranges')
    norm = matplotlib.colors.LogNorm(amps.min(), amps.max())
    sm = cm.ScalarMappable(norm=norm, cmap=mymap)
    sm._A = []

    # Plot
    logger.info('plotting')
    xvar = pltvars['Qm']
    Qbounds = np.array([Qref.min(), Qref.max()]) * xvar['factor']

    # Create figure
    fig, ax = plt.subplots(figsize=(3, 3))
    for s in ['right', 'top']:
        ax.spines[s].set_visible(False)
    ax.set_xlabel('$\\rm {}\ ({})$'.format(xvar['label'], xvar['unit']), fontsize=fs)
    ax.set_ylabel('$\\rm C_{m,eff}\ (\mu F /cm^2)$', fontsize=fs)
    ax.set_xticks(Qbounds)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)

    # Plot effective variable for each selected amplitude, and reference variable
    y0 = interp1d(Aref, Cmeff, axis=0)(0)
    for Adrive in amps:
        y = interp1d(Aref, Cmeff, axis=0)(Adrive)
        ax.plot(Qref * xvar['factor'], y, c=sm.to_rgba(Adrive))
    ax.plot(Qref * xvar['factor'], y0, '--', c='k')

    fig.tight_layout()

    return fig
