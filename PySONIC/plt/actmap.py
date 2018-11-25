# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-26 16:47:18
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-11-22 19:24:58

import os
import ntpath
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

from ..core import NeuronalBilayerSonophore
from ..utils import logger, si_format, ASTIM_filecode, cm2inch
from ..postpro import findPeaks
from ..constants import *
from ..neurons import getNeuronsDict


def getActivationMap(root, neuron, a, Fdrive, tstim, PRF, amps, DCs):
    ''' Compute the activation map of a neuron with specific sonophore radius
        at a given frequency and PRF, by computing the spiking metrics of simulation
        results over a 2D space (amplitude x duty cycle).

        :param root: directory containing the input data files
        :param neuron: neuron name
        :param a: sonophore radius
        :param Fdrive: US frequency (Hz)
        :param tstim: duration of US stimulation (s)
        :param PRF: pulse repetition frequency (Hz)
        :param amps: vector of acoustic amplitudes (Pa)
        :param DCs: vector of duty cycles (-)
        :return the activation matrix
    '''

    # Load activation map from file if it exists
    actmap_filename = 'actmap {} {}Hz PRF{}Hz {}s.csv'.format(
        neuron, *si_format([Fdrive, PRF, tstim], space=''))
    actmap_filepath = os.path.join(root, actmap_filename)
    if os.path.isfile(actmap_filepath):
        logger.info('Loading activation map for %s neuron', neuron)
        return np.loadtxt(actmap_filepath, delimiter=',')

    # Otherwise generate it
    logger.info('Generating activation map for %s neuron', neuron)
    actmap = np.empty((amps.size, DCs.size))
    nfiles = DCs.size * amps.size
    for i, A in enumerate(amps):
        for j, DC in enumerate(DCs):
            fname = '{}.pkl'.format(ASTIM_filecode(neuron, a, Fdrive, A, tstim, PRF, DC, 'sonic'))
            fpath = os.path.join(root, fname)

            if not os.path.isfile(fpath):
                logger.error('"{}" file not found'.format(fname))
                actmap[i, j] = np.nan
            else:
                # Load data
                logger.debug('Loading file {}/{}: "{}"'.format(i * amps.size + j + 1, nfiles, fname))
                with open(fpath, 'rb') as fh:
                    frame = pickle.load(fh)
                df = frame['data']
                meta = frame['meta']
                tstim = meta['tstim']
                t = df['t'].values
                Qm = df['Qm'].values
                dt = t[1] - t[0]

                # Detect spikes on charge profile during stimulus
                mpd = int(np.ceil(SPIKE_MIN_DT / dt))
                ispikes, *_ = findPeaks(Qm[t <= tstim], SPIKE_MIN_QAMP, mpd, SPIKE_MIN_QPROM)

                # Compute firing metrics
                if ispikes.size == 0:  # if no spike, assign -1
                    actmap[i, j] = -1
                elif ispikes.size == 1:  # if only 1 spike, assign 0
                    actmap[i, j] = 0
                else:  # if more than 1 spike, assign firing rate
                    FRs = 1 / np.diff(t[ispikes])
                    actmap[i, j] = np.mean(FRs)

    # Save activation map to file
    np.savetxt(actmap_filepath, actmap, delimiter=',')

    return actmap


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


def onClick(event, root, neuron, a, Fdrive, tstim, PRF, amps, DCs, meshedges, tmax, Vbounds):
    ''' Retrieve the specific input parameters of the x and y dimensions when the user clicks
        on a cell in the 2D map, and define filename from it.
    '''

    # Get DC and A from x and y coordinates
    x, y = event.xdata, event.ydata
    DC = DCs[np.searchsorted(meshedges[0], x * 1e-2) - 1]
    Adrive = amps[np.searchsorted(meshedges[1], y * 1e3) - 1]

    # Define filepath
    fname = '{}.pkl'.format(ASTIM_filecode(neuron, a, Fdrive, Adrive, tstim, PRF, DC, 'sonic'))
    filepath = os.path.join(root, fname)

    # Plot Q-trace
    try:
        plotQVeff(filepath, tmax=tmax, ybounds=Vbounds)
        plt.show()
    except FileNotFoundError as err:
        logger.error(err)


def plotQVeff(filepath, tonset=10, tmax=None, ybounds=None, fs=8, lw=1):
    '''  Plot superimposed profiles of membrane charge density and effective membrane potential.

        :param filepath: full path to the data file
        :param tonset: pre-stimulus onset to add to profiles (ms)
        :param tmax: max time value showed on graph (ms)
        :param ybounds: y-axis bounds (mV / nC/cm2)
        :return: handle to the generated figure
    '''
    # Check file existence
    fname = ntpath.basename(filepath)
    if not os.path.isfile(filepath):
        raise FileNotFoundError('Error: "{}" file does not exist'.format(fname))

    # Load data
    logger.debug('Loading data from "%s"', fname)
    with open(filepath, 'rb') as fh:
        frame = pickle.load(fh)
        df = frame['data']
        meta = frame['meta']
    t = df['t'].values * 1e3  # ms
    Qm = df['Qm'].values * 1e5  # nC/cm2
    Vm = df['Vm'].values  # mV

    # Add onset to profiles
    t = np.hstack((np.array([-tonset, t[0]]), t))
    Vm = np.hstack((np.array([getNeuronsDict()[meta['neuron']]().Vm0] * 2), Vm))
    Qm = np.hstack((np.array([Qm[0]] * 2), Qm))

    # Determine axes bounds
    if tmax is None:
        tmax = t.max()
    if ybounds is None:
        ybounds = (min(Vm.min(), Qm.min()), max(Vm.max(), Qm.max()))

    # Create figure
    fig, ax = plt.subplots(figsize=cm2inch(7, 3))
    fig.canvas.set_window_title(fname)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    for key in ['bottom', 'left']:
        ax.spines[key].set_position(('axes', -0.03))
        ax.spines[key].set_linewidth(2)
    ax.yaxis.set_tick_params(width=2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    print(-tonset, tmax)
    ax.set_xlim((-tonset, tmax))
    ax.set_xticks([])
    ax.set_xlabel('{}s'.format(si_format((tonset + tmax) * 1e-3, space=' ')), fontsize=fs)
    ax.set_ylabel('mV - $\\rm nC/cm^2$', fontsize=fs, labelpad=-15)
    ax.set_ylim(ybounds)
    ax.set_yticks(ybounds)
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)

    # Plot Qm and Vmeff profiles
    ax.plot(t, Vm, color='darkgrey', linewidth=lw)
    ax.plot(t, Qm, color='k', linewidth=lw)

    # fig.tight_layout()
    return fig


def plotActivationMap(root, neuron, a, Fdrive, tstim, PRF, amps, DCs, Ascale='log', FRscale='log',
                      FRbounds=None, title=None, fs=8, rheobase=True, connect=False,
                      tmax=None, Vbounds=None):
    ''' Plot a neuron's activation map over the amplitude x duty cycle 2D space.

        :param root: directory containing the input data files
        :param neuron: neuron name
        :param a: sonophore radius
        :param Fdrive: US frequency (Hz)
        :param tstim: duration of US stimulation (s)
        :param PRF: pulse repetition frequency (Hz)
        :param amps: vector of acoustic amplitudes (Pa)
        :param DCs: vector of duty cycles (-)
        :param Ascale: scale to use for the amplitude dimension ('lin' or 'log')
        :param FRscale: scale to use for the firing rate coloring ('lin' or 'log')
        :param FRbounds: lower and upper bounds of firing rate color-scale
        :param title: figure title
        :param fs: fontsize to use for the title and labels
        :return: 3-tuple with the handle to the generated figure and the mesh x and y coordinates
    '''

    # Get activation map
    actmap = getActivationMap(root, neuron, a, Fdrive, tstim, PRF, amps, DCs)

    # Check firing rate bounding
    minFR, maxFR = (actmap[actmap > 0].min(), actmap.max())
    logger.info('FR range: %.0f - %.0f Hz', minFR, maxFR)
    if FRbounds is None:
        FRbounds = (minFR, maxFR)
    else:
        if minFR < FRbounds[0]:
            logger.warning('Minimal firing rate (%.0f Hz) is below defined lower bound (%.0f Hz)',
                           minFR, FRbounds[0])
        if maxFR > FRbounds[1]:
            logger.warning('Maximal firing rate (%.0f Hz) is above defined upper bound (%.0f Hz)',
                           maxFR, FRbounds[1])

    # Plot activation map
    if FRscale == 'lin':
        norm = matplotlib.colors.Normalize(*FRbounds)
    elif FRscale == 'log':
        norm = matplotlib.colors.LogNorm(*FRbounds)
    fig, ax = plt.subplots(figsize=cm2inch(8, 5.8))
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.8, top=0.92)
    if title is None:
        title = '{} neuron @ {}Hz, {}Hz PRF ({}m sonophore)'.format(
            neuron, *si_format([Fdrive, PRF, a]))
    ax.set_title(title, fontsize=fs)
    if Ascale == 'log':
        ax.set_yscale('log')
    ax.set_xlabel('Duty cycle (%)', fontsize=fs, labelpad=-0.5)
    ax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    ax.set_xlim(np.array([DCs.min(), DCs.max()]) * 1e2)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    xedges = computeMeshEdges(DCs)
    yedges = computeMeshEdges(amps, scale=Ascale)
    actmap[actmap == -1] = np.nan
    actmap[actmap == 0] = 1e-3
    cmap = plt.get_cmap('viridis')
    cmap.set_bad('silver')
    cmap.set_under('k')
    ax.pcolormesh(xedges * 1e2, yedges * 1e-3, actmap, cmap=cmap, norm=norm)

    # Plot  rheobase amplitudes if specified
    if rheobase:
        logger.info('Computing rheobase amplitudes')
        dDC = 0.01
        DCs_dense = np.arange(dDC, 100 + dDC / 2, dDC) / 1e2
        neuronobj = getNeuronsDict()[neuron]()
        nbls = NeuronalBilayerSonophore(a, neuronobj)
        Athrs = nbls.findRheobaseAmps(DCs_dense, Fdrive, neuronobj.VT)[0]
        ax.plot(DCs_dense * 1e2, Athrs * 1e-3, '-', color='#F26522', linewidth=2,
                label='threshold amplitudes')
        ax.legend(loc='lower center', frameon=False, fontsize=8)

    # Plot firing rate colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    pos1 = ax.get_position()  # get the map axis position
    cbarax = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.03, pos1.height])
    fig.colorbar(sm, cax=cbarax)
    cbarax.set_ylabel('Firing rate (Hz)', fontsize=fs)
    for item in cbarax.get_yticklabels():
        item.set_fontsize(fs)

    # Link callback to figure
    if connect:
        fig.canvas.mpl_connect(
            'button_press_event',
            lambda event: onClick(event, root, neuron, a, Fdrive, tstim, PRF, amps, DCs,
                                  (xedges, yedges), tmax, Vbounds)
        )

    return fig


def plotAstimRheobaseAmps(neuron, radii, freqs, fs=12):
    ''' Plot threshold excitation amplitudes (determined by quasi-steady approximation)
        of a specific neuron as a function of duty cycle, for various combinations of
        sonophore radius and US frequency.

        :param neuron: neuron object
        :param radii: list of sonophore radii (m)
        :param freqs: list US frequencies (Hz)
        :return: figure handle
    '''
    linestyles = ['-', '--', ':', '-.']
    assert len(freqs) <= len(linestyles), 'too many frequencies'
    fig, ax = plt.subplots()
    ax.set_title('{} neuron: rheobase amplitude profiles'.format(neuron.name), fontsize=fs)
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('Threshold amplitude (kPa)', fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.set_yscale('log')
    ax.set_xlim([0, 100])
    ax.set_ylim([10, 600])
    DCs = np.arange(1, 101) / 1e2
    for i, a in enumerate(radii):
        nbls = NeuronalBilayerSonophore(a, neuron)
        for j, Fdrive in enumerate(freqs):
            Athrs, Aref = nbls.findRheobaseAmps(DCs, Fdrive, neuron.VT)
            color = 'C{}'.format(i)
            lbl = '{:.0f} nm radius sonophore, {}Hz'.format(a * 1e9, si_format(Fdrive, 1, space=' '))
            ax.plot(DCs * 1e2, Athrs * 1e-3, linestyles[j], c=color, label=lbl)
            # inans = np.where(np.isnan(Athrs))[0]
            # if len(inans) > 0:
            #     i1 = inans[-1] + 1
            # else:
            #     i1 = 0

            # i2 = i1 + 1
            # m = (np.log10(Athrs[i2]) - np.log10(Athrs[i1])) / (DCs[i2] - DCs[i1])
            # p = np.log10(Athrs[i1]) - m * DCs[i1]
            # DClim = max(0, (np.log10(Aref.max()) - p) / m)
            # ax.plot(np.array([DClim, DCs[i1]]) * 1e2,
            #         np.array([Aref.max(), Athrs[i1]]) * 1e-3, '--', c=color)
    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    return fig


def plotEstimRheobaseAmps(neurons, fs=15):
    fig, ax = plt.subplots()
    ax.set_title('Rheobase amplitudes', fontsize=fs)
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('Threshold amplitude (mA/m2)', fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.set_yscale('log')
    ax.set_ylim([1e0, 1e3])
    DCs = np.arange(1, 101) / 1e2
    for neuron in neurons:
        Athrs = neuron.findRheobaseAmps(DCs, neuron.VT)
        ax.plot(DCs * 1e2, Athrs, label='{} neuron'.format(neuron.name))
    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    return fig
