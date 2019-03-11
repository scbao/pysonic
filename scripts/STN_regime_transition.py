# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-11 21:48:12

''' Script to study STN transitions between different behavioral regimesl. '''

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from argparse import ArgumentParser
import logging

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import *
from PySONIC.neurons import getNeuronsDict

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Set logging level
logger.setLevel(logging.INFO)


def plotQSSvars_vs_Qm(neuron, a, Fdrive, Adrive, fs=12):

    # Get quasi-steady states and effective membrane potential profiles at this amplitude
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    _, Qref, Vmeff, QS_states = nbls.getQSSvars(Fdrive, amps=Adrive)

    # Compute QSS currents
    currents = neuron.currents(Vmeff, QS_states)
    iNet = sum(currents.values())

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(7, 9))
    axes[-1].set_xlabel('Charge Density (nC/cm2)', fontsize=fs)
    for ax in axes:
        for skey in ['top', 'right']:
            ax.spines[skey].set_visible(False)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
        for item in ax.get_xticklabels(minor=True):
            item.set_visible(False)
    figname = '{} neuron QSS dynamics @ {:.2f}kPa'.format(neuron.name, Adrive * 1e-3)
    fig.suptitle(figname, fontsize=fs)

    # Subplot 1: Vmeff
    ax = axes[0]
    ax.set_ylabel('$V_m^*$ (mV)', fontsize=fs)
    ax.plot(Qref * 1e5, Vmeff, color='C0')
    ax.axhline(neuron.Vm0, linewidth=0.5, color='k')

    # Subplot 2: quasi-steady states
    ax = axes[1]
    ax.set_ylabel('$X_\infty$', fontsize=fs)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([-0.05, 1.05])
    for label, qsstate in zip(neuron.states_names[:-1], QS_states[:-1]):
        ax.plot(Qref * 1e5, qsstate, label=label)

    # Subplot 3: currents
    ax = axes[2]
    ax.set_ylabel('QSS currents (A/m2)', fontsize=fs)
    for k, I in currents.items():
        ax.plot(Qref * 1e5, I * 1e-3, label=k)
    ax.plot(Qref * 1e5, iNet * 1e-3, color='k', label='iNet')
    ax.axhline(0, color='k', linewidth=0.5)

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    for ax in axes[1:]:
        ax.legend(loc='center right', fontsize=fs, frameon=False, bbox_to_anchor=(1.3, 0.5))

    return fig


def plotInetQSS_vs_Qm(neuron, a, Fdrive, amps, fs=12, cmap='viridis', zscale='lin'):

    # Compute net current profile for each amplitude, from QSS states and Vmeff profiles
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    _, Qref, Vmeff, QS_states = nbls.getQSSvars(Fdrive, amps=amps)
    iNet = neuron.iNet(Vmeff, QS_states)

    #  Define color code
    mymap = plt.get_cmap(cmap)
    zref = amps * 1e-3
    if zscale == 'lin':
        norm = matplotlib.colors.Normalize(zref.min(), zref.max())
    elif zscale == 'log':
        norm = matplotlib.colors.LogNorm(zref.min(), zref.max())
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=mymap)
    sm._A = []

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel('$\\rm Q_m\ (nC/cm^2)$', fontsize=fs)
    ax.set_ylabel('$\\rm I_{net, QSS}\ (A/m^2)$', fontsize=fs)
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    figname = '{} neuron - QSS current imbalance vs. amplitude'.format(neuron.name)
    ax.set_title(figname, fontsize=fs)
    ax.axhline(0, color='k', linewidth=0.5)

    # Plot iNet profiles for each US amplitude (with specific color code)
    for i, Adrive in enumerate(amps):
        lbl = '{:.2f} kPa'.format(Adrive * 1e-3)
        c = sm.to_rgba(Adrive * 1e-3)
        ax.plot(Qref * 1e5, iNet[i] * 1e-3, label=lbl, c=c)

    fig.tight_layout()

    # Plot US amplitude colorbar
    fig.subplots_adjust(bottom=0.15, top=0.9, right=0.80, hspace=0.5)
    cbarax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    fig.colorbar(sm, cax=cbarax)
    cbarax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    for item in cbarax.get_yticklabels():
        item.set_fontsize(fs)

    fig.canvas.set_window_title(
        '{}_iNet_QSS_vs_amp'.format(neuron.name))

    return fig


def getChargeStabilizationFromSims(inputdir, neuron, a, Fdrive, amps, tstim, PRF=100, DC=1.0):

    # Get filenames
    fnames = ['{}.pkl'.format(ASTIM_filecode(neuron.name, a, Fdrive, A, tstim, PRF, DC, 'sonic'))
              for A in amps]

    # Initialize output arrays
    tstab = np.empty(amps.size)
    Qstab = np.empty(amps.size)

    # For each file
    for i, fn in enumerate(fnames):

        # Extract charge temporal profile during stimulus
        fp = os.path.join(inputdir, 'STN', fn)
        logger.info('loading data from file "{}"'.format(fn))
        with open(fp, 'rb') as fh:
            frame = pickle.load(fh)
        df = frame['data']
        t = df['t'].values
        Qm = df['Qm'].values
        Qm = Qm[t < tstim]
        t = t[t < tstim]
        dt = np.diff(t)

        # If charge signal is stable during last 100 ms of stimulus
        if np.ptp(Qm[-int(100e-3 // dt[0]):]) < 5e-5:

            # Compute instant of stabilization by iNet thresholding
            iNet_abs = np.abs(np.diff(Qm)) / dt
            Qstab[i] = Qm[-1]
            tstab[i] = t[np.where(iNet_abs > 1e-3)[0][-1] + 2]
            logger.info('Qm stabilization around %.2f nC/cm2 from t = %.0f ms onward',
                        Qstab[i] * 1e5, tstab[i] * 1e3)

        # Otherwise, populate arrays with NaN
        else:
            Qstab[i] = np.nan
            tstab[i] = np.nan
            logger.info('No Qm stabilization')

    return Qstab, tstab


def getEqChargesFromQSS(neuron, a, Fdrive, amps, Qthr=None):

    # Compute net current profile for each amplitude, from QSS states and Vmeff profiles
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    _, Qref, Vmeff, QS_states = nbls.getQSSvars(Fdrive, amps=amps)
    iNet = neuron.iNet(Vmeff, QS_states)

    # Restrict iNet root-finding to a certain charge interval if provided
    if Qthr is not None:
        iNet = iNet[:, Qref >= Qthr]
        Qref = Qref[Qref >= Qthr]

    # Interpolate charge density vector at iNet = 0 for each amplitude
    Qeq_QSS = np.array([np.interp(0, iNet[i, :], Qref, left=0., right=np.nan)
                        for i in range(amps.size)])

    return Qeq_QSS


def compareEqChargesQSSvsSim(inputdir, neuron, a, Fdrive, amps, tstim, fs=12):

    # Get charge value that cancels out net current in QSS approx. and sim
    Qeq_QSS = getEqChargesFromQSS(neuron, a, Fdrive, amps, Qthr=-20e-5)
    Qeq_sim, _ = getChargeStabilizationFromSims(inputdir, neuron, a, Fdrive, amps, tstim)

    # Plot Qm balancing net current as function of amplitude
    fig, ax = plt.subplots(figsize=(6, 4))
    figname = '{} neuron - equilibrium charge vs. amplitude'.format(neuron.name)
    ax.set_title(figname)
    ax.set_xlabel('Amplitude (kPa)', fontsize=fs)
    ax.set_ylabel('$\\rm Q_{thr}\ (nC/cm^2)$', fontsize=fs)
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.plot(amps * 1e-3, Qeq_QSS * 1e5, label='QSS approximation')
    ax.plot(amps * 1e-3, Qeq_sim * 1e5, label='end of {:.2f} s stimulus (simulation)'.format(tstim))
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()

    fig.canvas.set_window_title(
        '{}_Qthr_vs_amp'.format(neuron.name))


def main():
    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-i', '--inputdir', type=str, default=None, help='Input directory')
    ap.add_argument('-f', '--figset', type=str, nargs='+', help='Figure set', default='all')
    ap.add_argument('-c', '--cmap', type=str, default='viridis', help='Colormap name')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')

    # Parse arguments
    args = ap.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    figset = args.figset
    if figset == 'all':
        figset = ['a', 'b', 'c']

    neuron = getNeuronsDict()['STN']()
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    intensities = getLowIntensitiesSTN()  # W/m2
    amps = Intensity2Pressure(intensities)  # Pa
    tstim = 1.0  # s

    figs = []
    if 'a' in figset:
        for Adrive in [amps[0], amps[-1]]:
            figs.append(plotQSSvars_vs_Qm(neuron, a, Fdrive, Adrive))
    if 'b' in figset:
        figs.append(plotInetQSS_vs_Qm(neuron, a, Fdrive, amps))
    if 'c' in figset:
        inputdir = args.inputdir if args.inputdir is not None else selectDirDialog()
        figs.append(compareEqChargesQSSvsSim(inputdir, neuron, a, Fdrive, amps, tstim))

    plt.show()


if __name__ == '__main__':
    main()
