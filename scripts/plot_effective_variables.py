#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-15 15:59:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-28 14:10:11

''' Plot the effective variables as a function of charge density with amplitude color code. '''

import numpy as np
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from argparse import ArgumentParser

from PySONIC.plt import pltvars
from PySONIC.utils import logger, getLookups2D
from PySONIC.neurons import getNeuronsDict


# Default parameters
defaults = dict(
    neuron='RS',
    diam=32.0,
    freq=500.0,
    amps=np.logspace(np.log10(1), np.log10(600), 10),  # kPa
)


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

    if 'V' in lookups2D:
        lookups2D['Vm'] = lookups2D.pop('V')

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
    for j, key in enumerate(lookups2D.keys()):
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


def main():
    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default=defaults['neuron'],
                    help='Neuron name (string)')
    ap.add_argument('-a', '--diam', type=float, default=defaults['diam'],
                    help='Sonophore diameter (nm)')
    ap.add_argument('-f', '--freq', type=float, default=defaults['freq'],
                    help='US frequency (kHz)')
    ap.add_argument('-A', '--amps', nargs='+', type=float, help='Acoustic pressure amplitude (kPa)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    neuron_str = args['neuron']
    diam = args['diam'] * 1e-9  # m
    Fdrive = args['freq'] * 1e3  # Hz
    amps = np.array(args.get('amps', defaults['amps'])) * 1e3  # Pa

    # Plot effective variables
    if neuron_str not in getNeuronsDict():
        logger.error('Unknown neuron type: "%s"', neuron_str)
        return
    neuron = getNeuronsDict()[neuron_str]()
    plotEffectiveVariables(neuron, diam, Fdrive, amps=amps)
    plt.show()


if __name__ == '__main__':
    main()
