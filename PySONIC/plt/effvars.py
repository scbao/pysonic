# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-10-02 01:44:59
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-14 22:03:18

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib

from PySONIC.utils import logger, si_prefixes, isWithin, getLookups2D, getLookupsOff


def setGrid(n, ncolmax=3):
    ''' Determine number of rows and columns in figure grid, based on number of
        variables to plot. '''
    if n <= ncolmax:
        return (1, n)
    else:
        return ((n - 1) // ncolmax + 1, ncolmax)


def plotEffectiveVariables(neuron, a=None, Fdrive=None, Adrive=None,
                           nlevels=10, zscale='lin', cmap=None, fs=12, ncolmax=1):
    ''' Plot the profiles of effective variables of a specific neuron as a function of charge density
        and another reference variable (z-variable). For each effective variable, one charge-profile
        per z-value is plotted, with a color code based on the z-variable value.

        :param neuron: channel mechanism object
        :param a: sonophore radius (m)
        :param Fdrive: acoustic drive frequency (Hz)
        :param Adrive: acoustic pressure amplitude (Pa)
        :param nlevels: number of levels for the z-variable
        :param zscale: scale type for the z-variable ('lin' or 'log')
        :param cmap: colormap name
        :param fs: figure fontsize
        :param ncolmax: max number of columns on the figure
        :return: handle to the created figure
    '''

    if sum(isinstance(x, float) for x in [a, Fdrive, Adrive]) < 2:
        raise ValueError('at least 2 parameters in (a, Fdrive, Adrive) must be fixed')

    if cmap is None:
        cmap = 'viridis'

    # Get reference US-OFF lookups (1D)
    _, lookupsoff = getLookupsOff(neuron.name)

    # Get 2D lookups at specific combination
    zref, Qref, lookups2D, zvar = getLookups2D(neuron.name, a=a, Fdrive=Fdrive, Adrive=Adrive)
    _, lookupsoff = getLookupsOff(neuron.name)
    for lookups in [lookups2D, lookupsoff]:
        lookups.pop('ng')
        lookups['Cm'] = Qref / lookups['V'] * 1e5  # uF/cm2

    zref *= zvar['factor']
    prefix = {value: key for key, value in si_prefixes.items()}[1 / zvar['factor']]

    # Optional: interpolate along z dimension if nlevels specified
    if zscale is 'log':
        znew = np.logspace(np.log10(zref.min()), np.log10(zref.max()), nlevels)
    elif zscale is 'lin':
        znew = np.linspace(zref.min(), zref.max(), nlevels)
    else:
        raise ValueError('unknown scale type (should be "lin" or "log")')
    znew = np.array([isWithin(zvar['label'], z, (zref.min(), zref.max())) for z in znew])
    lookups2D = {key: interp1d(zref, y2D, axis=0)(znew) for key, y2D in lookups2D.items()}
    zref = znew

    for lookups in [lookups2D, lookupsoff]:
        lookups['Vm'] = lookups.pop('V')  # mV
        lookups['Cm'] = Qref / lookups['Vm'] * 1e3  # uF/cm2
    keys = ['Cm', 'Vm'] + list(lookups2D.keys())[:-2]

    #  Define color code
    mymap = cm.get_cmap(cmap)
    if zscale == 'lin':
        norm = matplotlib.colors.Normalize(zref.min(), zref.max())
    elif zscale == 'log':
        norm = matplotlib.colors.LogNorm(zref.min(), zref.max())
    sm = cm.ScalarMappable(norm=norm, cmap=mymap)
    sm._A = []

    # Plot
    logger.info('plotting')
    nrows, ncols = setGrid(len(lookups2D), ncolmax=ncolmax)
    xvar = pltvars['Qm']
    Qbounds = np.array([Qref.min(), Qref.max()]) * xvar['factor']

    fig, _ = plt.subplots(figsize=(3.5 * ncols, 1 * nrows), squeeze=False)
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

        # Plot effective variable for each selected z-value
        y0 = lookupsoff[key]
        for i, z in enumerate(zref):
            y = lookups2D[key][i]
            if 'alpha' in key or 'beta' in key:
                y[y > y0.max() * 2] = np.nan
            ax.plot(Qref * xvar['factor'], y * yvar['factor'], c=sm.to_rgba(z))
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
        if key == 'Cm':
            factor = 1e1
            ylim = [np.floor(ylim[0] * factor) / factor, np.ceil(ylim[1] * factor) / factor]
        else:
            factor = 1 / np.power(10, np.floor(np.log10(ylim[1])))
            ylim = [np.floor(ylim[0] * factor) / factor, np.ceil(ylim[1] * factor) / factor]
        ax.set_yticks(ylim)
        ax.set_ylim(ylim)
        ax.set_ylabel('$\\rm {}\ ({})$'.format(yvar['label'], yvar['unit']), fontsize=fs,
                      rotation=0, ha='right', va='center')

    fig.suptitle('{} neuron: {} \n modulated effective variables'.format(neuron.name, zvar['label']))

    # Plot colorbar
    fig.subplots_adjust(left=0.20, bottom=0.05, top=0.8, right=0.80, hspace=0.5)
    cbarax = fig.add_axes([0.10, 0.90, 0.80, 0.02])
    fig.colorbar(sm, cax=cbarax, orientation='horizontal')
    cbarax.set_xlabel('{} ({}{})'.format(
        zvar['label'], prefix, zvar['unit']), fontsize=fs)
    for item in cbarax.get_yticklabels():
        item.set_fontsize(fs)

    return fig
