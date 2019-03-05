# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-05 11:58:23

''' Subpanels of the QSS approximation figure. '''

import os
import logging
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from argparse import ArgumentParser

from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.utils import logger, getLookups2D, selectDirDialog
from PySONIC.neurons import getNeuronsDict


# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Figure basename
figbase = os.path.splitext(__file__)[0]


def plotQuasiSteadySystem(neuron, a, Fdrive, PRF, DC, fs=8, markers=['-', '--', '.-'], title=None):

    neuron = getNeuronsDict()[neuron]()

    # Determine spiking threshold
    Vthr = neuron.VT  # mV
    Qthr = neuron.Cm0 * Vthr * 1e-3  # C/m2

    # Get lookups
    amps, Qref, lookups2D, _ = getLookups2D(neuron.name, a=a, Fdrive=Fdrive)
    amps *= 1e-3
    lookups1D = {key: interp1d(Qref, y2D, axis=1)(Qthr) for key, y2D in lookups2D.items()}

    # Remove unnecessary items ot get ON rates and effective potential at threshold charge
    rates_on = lookups1D
    rates_on.pop('ng')
    Vm_on = rates_on.pop('V')
    Vm_off = Qthr / neuron.Cm0 * 1e3

    # Compute neuron OFF rates at current charge value
    rates_off = neuron.getRates(Vm_off)

    # Compute pulse-average quasi-steady states
    qsstates_pulse = np.empty((len(neuron.states_names), amps.size))
    for j, x in enumerate(neuron.states_names):
        # If channel state, compute pulse-average steady-state values
        if x in neuron.getGates():
            x = x.lower()
            alpha_str, beta_str = ['{}{}'.format(s, x) for s in ['alpha', 'beta']]
            alphax_pulse = rates_on[alpha_str] * DC + rates_off[alpha_str] * (1 - DC)
            betax_pulse = rates_on[beta_str] * DC + rates_off[beta_str] * (1 - DC)
            qsstates_pulse[j, :] = alphax_pulse / (alphax_pulse + betax_pulse)
        # Otherwise assume the state has reached a steady-state value for Vthr
        else:
            qsstates_pulse[j, :] = np.ones(amps.size) * neuron.steadyStates(Vthr)[j]

    # Compute quasi-steady ON and OFF currents
    iLeak_on = neuron.iLeak(Vm_on)
    iLeak_off = np.ones(amps.size) * neuron.iLeak(Vm_off)
    m = qsstates_pulse[0, :]
    h = qsstates_pulse[1, :]
    iNa_on = neuron.iNa(m, h, Vm_on)
    iNa_off = neuron.iNa(m, h, Vm_off)
    n = qsstates_pulse[2, :]
    iK_on = neuron.iK(n, Vm_on)
    iK_off = neuron.iK(n, Vm_off)
    p = qsstates_pulse[3, :]
    iM_on = neuron.iM(p, Vm_on)
    iM_off = neuron.iM(p, Vm_off)
    if neuron.name == 'LTS':
        s = qsstates_pulse[4, :]
        u = qsstates_pulse[5, :]
        iCa_on = neuron.iCa(s, u, Vm_on)
        iCa_off = neuron.iCa(s, u, Vm_off)
    iNet_on = neuron.iNet(Vm_on, qsstates_pulse)
    iNet_off = neuron.iNet(Vm_off, qsstates_pulse)

    # Compute quasi-steady ON, OFF and net charge variations, and threshold amplitude
    dQ_on = -iNet_on * DC / PRF
    dQ_off = -iNet_off * (1 - DC) / PRF
    dQ_net = dQ_on + dQ_off
    Athr = np.interp(0, dQ_net, amps, left=0., right=np.nan)

    # Create figure
    fig, axes = plt.subplots(4, 1, figsize=(4, 6))
    axes[-1].set_xlabel('Amplitude (kPa)', fontsize=fs)
    for ax in axes:
        for skey in ['top', 'right']:
            ax.spines[skey].set_visible(False)
            ax.set_xscale('log')
        ax.set_xlim(1e1, 1e2)
        ax.set_xticks([1e1, 1e2])
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
        for item in ax.get_xticklabels(minor=True):
            item.set_visible(False)
    figname = '{} neuron thr dynamics {:.1f}nC_cm2 {:.0f}% DC'.format(
        neuron.name, Qthr * 1e5, DC * 1e2)
    fig.suptitle(figname, fontsize=fs)

    # Subplot 1: Vmeff
    ax = axes[0]
    ax.set_ylabel('Effective potential (mV)', fontsize=fs)
    Vbounds = (-120, -40)
    ax.set_ylim(Vbounds)
    ax.set_yticks([Vbounds[0], neuron.Vm0, Vbounds[1]])
    ax.set_yticklabels(['{:.0f}'.format(Vbounds[0]), '$V_{m0}$', '{:.0f}'.format(Vbounds[1])])
    ax.plot(amps, Vm_on, color='C0', label='ON')
    ax.plot(amps, Vm_off * np.ones(amps.size), '--', color='C0', label='OFF')
    ax.axhline(neuron.Vm0, linewidth=0.5, color='k')

    # Subplot 2: quasi-steady states
    ax = axes[1]
    ax.set_ylabel('Quasi-steady states', fontsize=fs)
    ax.set_yticks([0, 0.5, 0.6])
    ax.set_yticklabels(['0', '0.5', '1'])
    ax.set_ylim([-0.05, 0.65])
    d = .01
    f = 1.03
    xcut = ax.get_xlim()[0]
    for ycut in [0.54, 0.56]:
        ax.plot([xcut / f, xcut * f], [ycut - d, ycut + d], color='k', clip_on=False)
    for label, qsstate in zip(neuron.states_names, qsstates_pulse):
        if label == 'h':
            qsstate -= 0.4
        ax.plot(amps, qsstate, label=label)

    # Subplot 3: currents
    ax = axes[2]
    ax.set_ylabel('QS Currents (mA/m2)', fontsize=fs)
    Ibounds = (-10, 10)
    ax.set_ylim(Ibounds)
    ax.set_yticks([Ibounds[0], 0.0, Ibounds[1]])
    ax.plot(amps, iLeak_on, color='C0', label='$I_{Leak}$')
    ax.plot(amps, iLeak_off, '--', color='C0')
    ax.plot(amps, iNa_on, '-', color='C1', label='$I_{Na}$')
    ax.plot(amps, iNa_off, '--', color='C1')
    ax.plot(amps, iK_on, '-', color='C2', label='$I_{K}$')
    ax.plot(amps, iK_off, '--', color='C2')
    ax.plot(amps, iM_on, '-', color='C3', label='$I_{M}$')
    ax.plot(amps, iM_off, '--', color='C3')
    if neuron.name == 'LTS':
        ax.plot(amps, iCa_on, color='C5', label='$I_{Ca}$')
        ax.plot(amps, iCa_off, '--', color='C5')
    ax.plot(amps, iNet_on, '-', color='k', label='$I_{Net}$')
    ax.plot(amps, iNet_off, '--', color='k')

    # Subplot 4: charge variations and activation threshold
    ax = axes[3]
    ax.set_ylabel('$\\rm \Delta Q_{QS}\ (nC/cm^2)$', fontsize=fs)
    dQbounds = (-0.06, 0.1)
    ax.set_ylim(dQbounds)
    ax.set_yticks([dQbounds[0], 0.0, dQbounds[1]])
    ax.plot(amps, dQ_on, color='C0', label='ON')
    ax.plot(amps, dQ_off, '--', color='C0', label='OFF')
    ax.plot(amps, dQ_net, '-.', color='C0', label='Net')
    ax.plot([Athr] * 2, [ax.get_ylim()[0], 0], linestyle='--', color='k')
    ax.plot([Athr], [0], 'o', c='k')
    ax.axhline(0, color='k', linewidth=0.5)

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    for ax in axes:
        ax.legend(loc='center right', fontsize=fs, frameon=False, bbox_to_anchor=(1.3, 0.5))

    if title is not None:
        fig.canvas.set_window_title(title)
    return fig


def plotdQvsDC(neuron, a, Fdrive, PRF, DCs, fs=8, title=None):

    neuron = getNeuronsDict()[neuron]()

    # Determine spiking threshold
    Vthr = neuron.VT  # mV
    Qthr = neuron.Cm0 * Vthr * 1e-3  # C/m2

    # Get lookups
    amps, Qref, lookups2D, _ = getLookups2D(neuron.name, a=a, Fdrive=Fdrive)
    amps *= 1e-3
    lookups1D = {key: interp1d(Qref, y2D, axis=1)(Qthr) for key, y2D in lookups2D.items()}

    # Remove unnecessary items ot get ON rates and effective potential at threshold charge
    rates_on = lookups1D
    rates_on.pop('ng')
    Vm_on = rates_on.pop('V')

    rates_off = neuron.getRates(Vthr)

    # For each duty cycle, compute net charge variation at Qthr along the amplitude range,
    # and identify rheobase amplitude
    Athr = np.empty_like(DCs)
    dQnet = np.empty((DCs.size, amps.size))
    for i, DC in enumerate(DCs):
        # Compute pulse-average quasi-steady states
        qsstates_pulse = np.empty((len(neuron.states_names), amps.size))
        for j, x in enumerate(neuron.states_names):
            # If channel state, compute pulse-average steady-state values
            if x in neuron.getGates():
                x = x.lower()
                alpha_str, beta_str = ['{}{}'.format(s, x) for s in ['alpha', 'beta']]
                alphax_pulse = rates_on[alpha_str] * DC + rates_off[alpha_str] * (1 - DC)
                betax_pulse = rates_on[beta_str] * DC + rates_off[beta_str] * (1 - DC)
                qsstates_pulse[j, :] = alphax_pulse / (alphax_pulse + betax_pulse)
            # Otherwise assume the state has reached a steady-state value for Vthr
            else:
                qsstates_pulse[j, :] = np.ones(amps.size) * neuron.steadyStates(Vthr)[j]

        # Compute the pulse average net current along the amplitude space
        iNet_on = neuron.iNet(Vm_on, qsstates_pulse)
        iNet_off = neuron.iNet(Vthr, qsstates_pulse)
        iNet_avg = iNet_on * DC + iNet_off * (1 - DC)
        dQnet[i, :] = -iNet_avg / PRF

        # Compute threshold amplitude
        Athr[i] = np.interp(0, dQnet[i, :], amps, left=0., right=np.nan)

    # Create figure
    fig, ax = plt.subplots(figsize=(4, 2))
    figname = '{} neuron thr vs DC'.format(neuron.name, Qthr * 1e5)
    fig.suptitle(figname, fontsize=fs)
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    ax.set_xscale('log')
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    for item in ax.get_xticklabels(minor=True):
            item.set_visible(False)
    ax.set_xlabel('Amplitude (kPa)', fontsize=fs)
    ax.set_ylabel('$\\rm \Delta Q_{QS}\ (nC/cm^2)$', fontsize=fs)
    ax.set_xlim(1e1, 1e2)
    ax.axhline(0., linewidth=0.5, color='k')
    ax.set_ylim(-0.06, 0.12)
    ax.set_yticks([-0.05, 0.0, 0.10])
    ax.set_yticklabels(['-0.05', '0', '0.10'])

    norm = matplotlib.colors.LogNorm(DCs.min(), DCs.max())
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    sm._A = []
    for i, DC in enumerate(DCs):
        ax.plot(amps, dQnet[i, :], c=sm.to_rgba(DC), label='{:.0f}% DC'.format(DC * 1e2))
        ax.plot([Athr[i]] * 2, [ax.get_ylim()[0], 0], linestyle='--', c=sm.to_rgba(DC))
        ax.plot([Athr[i]], [0], 'o', c=sm.to_rgba(DC))

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    ax.legend(loc='center right', fontsize=fs, frameon=False, bbox_to_anchor=(1.3, 0.5))

    if title is not None:
        fig.canvas.set_window_title(title)

    return fig


def plotRheobaseAmps(neurons, a, Fdrive, DCs_dense, DCs_sparse, fs=8, title=None):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_title('Rheobase amplitudes', fontsize=fs)
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('$\\rm A_T\ (kPa)$', fontsize=fs)
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.set_xticks([25, 50, 75, 100])
    ax.set_yscale('log')
    ax.set_ylim([10, 600])
    norm = matplotlib.colors.LogNorm(DCs_sparse.min(), DCs_sparse.max())
    sm = cm.ScalarMappable(norm=norm, cmap='viridis')
    sm._A = []
    for i, neuron in enumerate(neurons):
        neuron = getNeuronsDict()[neuron]()
        nbls = NeuronalBilayerSonophore(a, neuron)
        Athrs_dense = nbls.findRheobaseAmps(DCs_dense, Fdrive, neuron.VT)[0] * 1e-3  # kPa
        Athrs_sparse = nbls.findRheobaseAmps(DCs_sparse, Fdrive, neuron.VT)[0] * 1e-3  # kPa
        ax.plot(DCs_dense * 1e2, Athrs_dense, label='{} neuron'.format(neuron.name))
        for DC, Athr in zip(DCs_sparse, Athrs_sparse):
            ax.plot(DC * 1e2, Athr, 'o',
                    label='{:.0f}% DC'.format(DC * 1e2) if i == len(neurons) - 1 else None,
                    c=sm.to_rgba(DC))
    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    if title is not None:
        fig.canvas.set_window_title(title)
    return fig


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-o', '--outdir', type=str, help='Output directory')
    ap.add_argument('-f', '--figset', type=str, nargs='+', help='Figure set', default='all')
    ap.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save output figures as pdf')

    args = ap.parse_args()
    loglevel = logging.DEBUG if args.verbose is True else logging.INFO
    logger.setLevel(loglevel)
    figset = args.figset
    if figset == 'all':
        figset = ['a', 'b', 'c', 'e']

    logger.info('Generating panels {} of {}'.format(figset, figbase))

    # Parameters
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    PRF = 100.0  # Hz
    DC = 0.5
    DCs_sparse = np.array([5, 15, 50, 75, 95]) / 1e2
    DCs_dense = np.arange(1, 101) / 1e2

    # Figures
    figs = []
    if 'a' in figset:
        figs += [
            plotQuasiSteadySystem('RS', a, Fdrive, PRF, DC, title=figbase + 'a RS'),
            plotQuasiSteadySystem('LTS', a, Fdrive, PRF, DC, title=figbase + 'a LTS')
        ]
    if 'b' in figset:
        figs += [
            plotdQvsDC('RS', a, Fdrive, PRF, DCs_sparse, title=figbase + 'b RS'),
            plotdQvsDC('LTS', a, Fdrive, PRF, DCs_sparse, title=figbase + 'b LTS')
        ]
    if 'c' in figset:
        figs.append(plotRheobaseAmps(['RS', 'LTS'], a, Fdrive, DCs_dense, DCs_sparse,
                                     title=figbase + 'c'))

    if args.save:
        outdir = selectDirDialog() if args.outdir is None else args.outdir
        if outdir == '':
            logger.error('No input directory chosen')
            return
        for fig in figs:
            figname = '{}.pdf'.format(fig.canvas.get_window_title())
            fig.savefig(os.path.join(outdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
