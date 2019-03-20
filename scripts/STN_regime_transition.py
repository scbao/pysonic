# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-20 17:57:06

''' Script to study STN transitions between different behavioral regimesl. '''

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from argparse import ArgumentParser
import logging

from PySONIC.constants import Z_Ca
from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import *
from PySONIC.postpro import getStableFixedPoints
from PySONIC.neurons import getNeuronsDict

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Set logging level
logger.setLevel(logging.INFO)


def getStableQmQSS(neuron, a, Fdrive, amps):

    # Compute net current profile for each amplitude, from QSS states and Vmeff profiles
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    _, Qref, Vmeff, QS_states = nbls.getQSSvars(Fdrive, amps=amps)
    iNet = neuron.iNet(Vmeff, QS_states)

    # Find stable fixed points in iNet(Qref) profile
    Qstab = []
    for i, Adrive in enumerate(amps):
        Qstab.append(getStableFixedPoints(Qref, -iNet[i, :]))

    return Qstab


def getChargeStabilizationFromSims(inputdir, neuron, a, Fdrive, amps, tstim, PRF=100, DC=1.0):

    # Get filenames
    fnames = ['{}.pkl'.format(ASTIM_filecode(neuron.name, a, Fdrive, A, tstim, PRF, DC, 'sonic'))
              for A in amps]

    # Initialize output arrays
    t_stab = np.empty(amps.size)
    Q_stab = np.empty(amps.size)
    Ca_stab = np.empty(amps.size)

    # For each file
    for i, fn in enumerate(fnames):

        # Extract charge temporal profile during stimulus
        fp = os.path.join(inputdir, fn)
        # logger.info('loading data from file "{}"'.format(fn))
        with open(fp, 'rb') as fh:
            frame = pickle.load(fh)
        df = frame['data']
        t = df['t'].values
        Qm = df['Qm'].values
        Ca = df['Cai'].values
        Qm = Qm[t < tstim]
        Ca = Ca[t < tstim]
        t = t[t < tstim]
        dt = np.diff(t)

        # If charge signal is stable during last 100 ms of stimulus
        if np.ptp(Qm[-int(100e-3 // dt[0]):]) < 5e-5:

            # Compute instant of stabilization by iNet thresholding
            iNet_abs = np.abs(np.diff(Qm)) / dt
            t_stab[i] = t[np.where(iNet_abs > 1e-3)[0][-1] + 2]

            # Get steady-state charge and Calcium concentration values
            Q_stab[i] = Qm[-1]
            Ca_stab[i] = Ca[-1]

            logger.debug('A = %.2f kPa: Qm stabilization around %.2f nC/cm2 from t = %.0f ms onward',
                         amps[i] * 1e-3, Q_stab[i] * 1e5, t_stab[i] * 1e3)

        # Otherwise, populate arrays with NaN
        else:
            t_stab[i] = np.nan
            Q_stab[i] = np.nan
            Ca_stab[i] = np.nan
            logger.debug('A = %.2f kPa: no Qm stabilization', amps[i] * 1e-3)

    return t_stab, Q_stab, Ca_stab


def plotCaiDynamics(neuron, a, Cai, Fdrive, Adrive, charges, fs=12):

    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)

    # Compute charge and amplitude dependent variables
    _, _, Vmeff, QS_states = nbls.getQSSvars(Fdrive, amps=Adrive, charges=charges)
    c_ss, d1_ss, p_ss, q_ss = [QS_states[i] for i in [2, 3, 8, 9]]

    # Compute Cai-dependent variables
    ECa_ss = neuron.nernst(Z_Ca, Cai, neuron.Cao, neuron.T)  # mV
    d2_ss = neuron.d2inf(Cai)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(7, 6))
    axes[0].set_title(
        '{} neuron - [Cai] dynamics @ {:.2f} kPa'.format(neuron.name, Adrive * 1e-3),
        fontsize=fs)
    for ax in axes:
        ax.set_xscale('log')
        for key in ['top', 'right']:
            ax.spines[key].set_visible(False)
    axes[-1].set_xlabel('$\\rm [Ca_i]\ (uM)$', fontsize=fs)
    axes[0].set_ylabel('$\\rm E_{Ca, \infty}$ (mV)', fontsize=fs)
    axes[1].set_ylabel('$\\rm d2_{\infty}$ (-)', fontsize=fs)
    axes[2].set_ylabel('$\\rm {d[Ca_i]/dt}_{\infty}\ (\mu M/s)$', fontsize=fs)
    axes[2].set_ylim(-40, 40)
    axes[2].axhline(0, c='k', linewidth=0.5)
    axes[2].axvline(neuron.Cai0 * 1e6, c='k', label='$\\rm [Ca_i]_0$')

    # Plot Ca-dependent variables
    axes[0].plot(Cai * 1e6, ECa_ss, c='k')
    axes[1].plot(Cai * 1e6, d2_ss, c='k')

    # For each amplitude-charge combination
    icolor = 0
    for j, Qm in enumerate(charges):
        lbl = 'Q = {:.0f} nC/cm2'.format(Qm * 1e5)

        # Compute Cai-derivative as a function of Cai
        dCaidt_ss = neuron.derCai(
            p_ss[j], q_ss[j], c_ss[j], d1_ss[j], d2_ss, Cai, Vmeff[j])  # M/s

        # Find Cai value that cancels derivative
        Cai_eq = neuron.findCaiSteadyState(
            *[x[j] for x in [p_ss, q_ss, c_ss, d1_ss, Vmeff]])
        logger.debug('steady-state Calcium concentration @ %s: %.2e uM', lbl, Cai_eq * 1e6)

        # Plot Cai-derivative and its root
        c = 'C{}'.format(icolor)
        axes[2].plot(Cai * 1e6, dCaidt_ss * 1e6, c=c, label=lbl)
        axes[2].axvline(Cai_eq * 1e6, linestyle='--', c=c)
        icolor += 1

    axes[2].legend(frameon=False, fontsize=fs - 3)
    for ax in axes:
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)

    fig.tight_layout()
    fig.canvas.set_window_title('{}_Ca_dynamics_{:.2f}kPa'.format(neuron.name, Adrive * 1e-3))

    return fig


def plotQSSvars_vs_Qm(neuron, a, Fdrive, Adrive, fs=12):

    # Get quasi-steady states and effective membrane potential profiles at this amplitude
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    _, Qref, Vmeff, QS_states = nbls.getQSSvars(Fdrive, amps=Adrive)

    # Compute charge and amplitude dependent variables
    _, charges, Vmeff, QS_states = nbls.getQSSvars(Fdrive, amps=Adrive)
    c_ss, d1_ss, p_ss, q_ss = [QS_states[i] for i in [2, 3, 8, 9]]
    Cai_ss = np.array([
        neuron.findCaiSteadyState(*[x[j] for x in [p_ss, q_ss, c_ss, d1_ss, Vmeff]])
        for j in range(len(Qref))])
    QS_states[4] = neuron.d2inf(Cai_ss)
    QS_states[10] = neuron.rinf(Cai_ss)

    # Compute QSS currents
    currents = neuron.currents(Vmeff, QS_states)
    iNet = sum(currents.values())

    # Qi = -22.76e-5  # C/m2
    # print('interpolated QSS system at Qm = {:.5f} nC/cm2:'.format(Qi * 1e5))
    # print('Vmeff = {:.5f} mV'.format(np.interp(Qi, Qref, Vmeff)))
    # for name, vec in currents.items():
    #     print('{} = {:.5f} A/m2'.format(name, np.interp(Qi, Qref, vec) * 1e-3))
    # print('iNet = {:.5f} A/m2'.format(np.interp(Qi, Qref, iNet) * 1e-3))
    # for name, vec in zip(neuron.states, QS_states[:-1, :]):
    #     print('{} = {:.5f}'.format(name, np.interp(Qi, Qref, vec)))
    # print('Cai = {:.5f} uM'.format(np.interp(Qi, Qref, QS_states[-1, :]) * 1e6))

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
    figname = '{} neuron - QSS dynamics @ {:.2f} kPa'.format(neuron.name, Adrive * 1e-3)
    fig.suptitle(figname, fontsize=fs)

    # Subplot 1: Vmeff
    ax = axes[0]
    ax.set_ylabel('$V_m^*$ (mV)', fontsize=fs)
    ax.plot(Qref * 1e5, Vmeff, color='k')
    ax.axhline(neuron.Vm0, linewidth=0.5, color='k')

    # Subplot 2: quasi-steady states
    colors = plt.get_cmap('tab10').colors + plt.get_cmap('Dark2').colors
    ax = axes[1]
    ax.set_ylabel('$X_\infty$', fontsize=fs)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([-0.05, 1.05])
    for i, label in enumerate(neuron.states):
        if label != 'Cai':
            ax.plot(Qref * 1e5, QS_states[i], label=label, c=colors[i])

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

    fig.canvas.set_window_title(
        '{}_QSS_states_vs_Qm_{:.2f}kPa'.format(neuron.name, Adrive * 1e-3))

    return fig


def plotCaiSS_vs_Qm(neuron, a, Fdrive, amps, fs=12, cmap='viridis', zscale='lin'):

    #  Define color code
    mymap = plt.get_cmap(cmap)
    zref = amps * 1e-3
    if zscale == 'lin':
        norm = matplotlib.colors.Normalize(zref.min(), zref.max())
    elif zscale == 'log':
        norm = matplotlib.colors.LogNorm(zref.min(), zref.max())
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=mymap)
    sm._A = []

    # Compute charge and amplitude dependent variables
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    _, charges, Vmeff, QS_states = nbls.getQSSvars(Fdrive, amps=amps)
    c_ss, d1_ss, p_ss, q_ss = [QS_states[i] for i in [2, 3, 8, 9]]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title('{} neuron - [Cai] steady-state vs. amplitude'.format(neuron.name), fontsize=fs)
    ax.set_xlabel('Charge Density (nC/cm2)', fontsize=fs)
    ax.set_ylabel('$\\rm [Ca_i]_{\infty}\ (\mu M)$', fontsize=fs)
    ax.set_yscale('log')
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    ax.axvline(neuron.Vm0 * neuron.Cm0 * 1e2, label='$\\rm Q_{m0}$', c='silver', linewidth=0.5)
    ax.axhline(neuron.Cai0 * 1e6, label='$\\rm [Ca_i]_0$', c='k', linewidth=0.5)

    # Find Cai_eq as a function of charge for each amplitude value
    for i, Adrive in enumerate(amps):
        Cai_eq = np.array([
            neuron.findCaiSteadyState(*[x[i, j] for x in [p_ss, q_ss, c_ss, d1_ss, Vmeff]])
            for j in range(len(charges))])
        ax.plot(charges * 1e5, Cai_eq * 1e6, c=sm.to_rgba(Adrive * 1e-3))

    ax.legend(frameon=False, fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    fig.tight_layout()

    # Plot US amplitude colorbar
    fig.subplots_adjust(bottom=0.15, top=0.9, right=0.80, hspace=0.5)
    cbarax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    fig.colorbar(sm, cax=cbarax)
    cbarax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    for item in cbarax.get_yticklabels():
        item.set_fontsize(fs)

    fig.canvas.set_window_title('{}_Cai_QSS_vs_amp'.format(neuron.name))

    return fig



def plotInetQSS_vs_Qm(neuron, a, Fdrive, amps, fs=12, cmap='viridis', zscale='lin'):

    # Compute net current profile for each amplitude, from QSS states and Vmeff profiles
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    _, Qref, Vmeff, QS_states = nbls.getQSSvars(Fdrive, amps=amps)

    # Compute charge and amplitude dependent variables
    _, charges, Vmeff, QS_states = nbls.getQSSvars(Fdrive, amps=amps)
    c_ss, d1_ss, p_ss, q_ss = [QS_states[i] for i in [2, 3, 8, 9]]
    for i, Adrive in enumerate(amps):
        Cai_ss = np.array([
            neuron.findCaiSteadyState(*[x[i, j] for x in [p_ss, q_ss, c_ss, d1_ss, Vmeff]])
            for j in range(len(Qref))])
    QS_states[4, i] = neuron.d2inf(Cai_ss)
    QS_states[10, i] = neuron.rinf(Cai_ss)

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
    ax.set_xlabel('Charge density (nC/cm2)', fontsize=fs)
    ax.set_ylabel('$\\rm I_{net, QSS}\ (A/m^2)$', fontsize=fs)
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    figname = '{} neuron - QSS current imbalance vs. amplitude'.format(neuron.name)
    ax.set_title(figname, fontsize=fs)

    # Plot iNet profiles for each US amplitude (with specific color code)
    for i, Adrive in enumerate(amps):
        lbl = '{:.2f} kPa'.format(Adrive * 1e-3)
        c = sm.to_rgba(Adrive * 1e-3)
        ax.plot(Qref * 1e5, iNet[i] * 1e-3, label=lbl, c=c)
    for i, Adrive in enumerate(amps):
        Qstab = getStableFixedPoints(Qref, -iNet[i, :])
        if Qstab is not None:
            ax.plot(Qstab * 1e5, np.zeros(Qstab.size), '.', c='k')
    ax.axhline(0, color='k', linewidth=0.5)

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


def compareEqChargesQSSvsSim(inputdir, neuron, a, Fdrive, amps, tstim, fs=12):

    # Get charge value that cancels out net current in QSS approx. and sim
    Qeq_QSS = getStableQmQSS(neuron, a, Fdrive, amps)
    Qeq_QSS2 = np.array([Q[0] if Q is not None and len(Q) == 1 else np.nan for Q in Qeq_QSS])

    _, Qeq_sim, _ = getChargeStabilizationFromSims(inputdir, neuron, a, Fdrive, amps, tstim)

    Q_rmse = np.sqrt(np.nanmean((Qeq_sim - Qeq_QSS2)**2))
    logger.info('RMSE Q = %.3f nC/cm2', Q_rmse * 1e5)

    # Plot Qm balancing net current as function of amplitude
    fig, ax = plt.subplots(figsize=(6, 4))
    figname = '{} neuron - equilibrium charge vs. amplitude'.format(neuron.name)
    ax.set_title(figname)
    ax.set_xlabel('Amplitude (kPa)', fontsize=fs)
    ax.set_ylabel('$\\rm Q_{eq}\ (nC/cm^2)$', fontsize=fs)
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)

    lgd = True
    for Adrive, Qstab in zip(amps, Qeq_QSS):
        if Qstab is not None:
            if lgd:
                lbl = 'QSS approximation'
                lgd = False
            else:
                lbl = None
            ax.plot(np.ones(Qstab.size) * Adrive * 1e-3, Qstab * 1e5, '.', c='C0', label=lbl)
    ax.plot(amps * 1e-3, Qeq_sim * 1e5, '.', c='C1',
            label='end of {:.2f} s stimulus (simulation)'.format(tstim))
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()

    fig.canvas.set_window_title(
        '{}_Qeq_vs_amp'.format(neuron.name))

    return fig


def main():
    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default='STN', help='Neuron type')
    ap.add_argument('-i', '--inputdir', type=str, default=None, help='Input directory')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')
    ap.add_argument('-f', '--figset', type=str, nargs='+', help='Figure set', default='all')
    ap.add_argument('-c', '--cmap', type=str, default='viridis', help='Colormap name')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save output figures as png')

    # Parse arguments
    args = ap.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    figset = args.figset
    if figset == 'all':
        figset = ['a', 'b', 'c']

    neuron = getNeuronsDict()[args.neuron]()
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    intensities = getLowIntensitiesSTN()  # W/m2
    amps = Intensity2Pressure(intensities)  # Pa
    tstim = 1.0  # s
    Cai = np.logspace(np.log10(neuron.Cai0 * 1e-4), np.log10(neuron.Cai0 * 1e3), 100)
    charges = np.array([neuron.Qbounds()[1], neuron.Vm0 * neuron.Cm0 * 1e-3])

    figs = []
    if 'a' in figset:
        # for Adrive in [amps[0], amps[amps.size // 2], amps[-1]]:
        for Adrive in [amps[-1]]:
            figs += [
                plotQSSvars_vs_Qm(neuron, a, Fdrive, Adrive),
                plotCaiDynamics(neuron, a, Cai, Fdrive, Adrive, charges)
            ]
    if 'b' in figset:
        figs += [
            plotInetQSS_vs_Qm(neuron, a, Fdrive, amps),
            plotCaiSS_vs_Qm(neuron, a, Fdrive, amps)
        ]
    if 'c' in figset:
        inputdir = args.inputdir if args.inputdir is not None else selectDirDialog()
        if inputdir == '':
            logger.error('no input directory')
        else:
            figs.append(compareEqChargesQSSvsSim(inputdir, neuron, a, Fdrive, amps, tstim))

    if args.save:
        outputdir = args.outputdir if args.outputdir is not None else selectDirDialog()
        if outputdir == '':
            logger.error('no output directory')
        else:
            for fig in figs:
                s = fig.canvas.get_window_title()
                s = s.replace('(', '- ').replace('/', '_').replace(')', '')
                figname = '{}.png'.format(s)
                fig.savefig(os.path.join(outputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
