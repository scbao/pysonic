# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-06-06 18:38:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-28 18:30:55


import os
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Rectangle
from argparse import ArgumentParser

from PySONIC.utils import logger, rescale, cm2inch, getStimPulses, si_format, selectDirDialog
from PySONIC.constants import NPC_FULL
from PySONIC.neurons import CorticalRS
from PySONIC.core import BilayerSonophore, NeuronalBilayerSonophore

# Set logging level
logger.setLevel(logging.INFO)

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'


def plotPmavg(bls, Z, fs=12, lw=2):
    fig, ax = plt.subplots(figsize=cm2inch(7, 7))
    for key in ['right', 'top']:
        ax.spines[key].set_visible(False)
    for key in ['bottom', 'left']:
        ax.spines[key].set_linewidth(2)
    ax.spines['bottom'].set_position('zero')
    ax.set_xlabel('Z (nm)', fontsize=fs)
    ax.set_ylabel('Pressure (kPa)', fontsize=fs, labelpad=-10)
    ax.set_xticks([0, bls.a * 1e9])
    ax.set_xticklabels(['0', 'a'])
    ax.tick_params(axis='x', which='major', length=25, pad=5)
    ax.set_yticks([0])
    ax.set_ylim([-10, 50])
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.plot(Z * 1e9, bls.v_PMavg(Z, bls.v_curvrad(Z), bls.surface(Z)) * 1e-3, label='$P_m$')
    ax.plot(Z * 1e9, bls.PMavgpred(Z) * 1e-3, '--', label='$P_{m,approx}$')
    ax.axhline(y=0, color='k')
    ax.legend(fontsize=fs, frameon=False)
    fig.tight_layout()
    return fig


def plotZoomVQ(nbls, Fdrive, Adrive, fs=12, lw=2, ps=15):

    # Run effective simulation
    t, y, states = nbls.simulate(Fdrive, Adrive, 5 / Fdrive, 0., method='full')
    t *= 1e6  # us
    Qm = y[2] * 1e5  # nC/cm2
    Vm = y[3]  # mV
    Qrange = (Qm.min(), Qm.max())
    dQ = Qrange[1] - Qrange[0]

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=cm2inch(17, 5))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    # Plot Q-trace and V-trace
    ax = axes[0]
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    for key in ['bottom', 'left']:
        ax.spines[key].set_position(('axes', -0.03))
        ax.spines[key].set_linewidth(2)
    ax.plot(t, Vm, label='Vm', linewidth=lw)
    ax.plot(t, Qm, label='Qm', linewidth=lw)
    ax.add_patch(Rectangle(
        (t[0], Qrange[0] - 5), t[-1], dQ + 10,
        fill=False, edgecolor='k', linestyle='--', linewidth=2
    ))
    ax.yaxis.set_tick_params(width=2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # ax.set_xlim((t.min(), t.max()))
    ax.set_xticks([])
    ax.set_xlabel('{}s'.format(si_format((t.max()) * 1e-3, space=' ')), fontsize=fs)
    ax.set_ylabel('$\\rm nC/cm^2$ - mV', fontsize=fs, labelpad=-15)
    ax.set_yticks(ax.get_ylim())
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)

    # Plot inset on Q-trace
    ax = axes[1]
    for key in ['top', 'right', 'bottom', 'left']:
        ax.spines[key].set_linewidth(2)
        ax.spines[key].set_linestyle('--')
    ax.plot(t, Vm, label='Vm', linewidth=lw)
    ax.plot(t, Qm, label='Qm', linewidth=lw)
    ax.set_xlim((t.min(), t.max()))
    ax.set_xticks([])
    ax.set_yticks([])
    delta = 0.05
    ax.set_ylim(Qrange[0] - delta * dQ, Qrange[1] + delta * dQ)

    return fig


def plotMechSim(bls, Fdrive, Adrive, Qm, fs=12, lw=2, ps=15):

    # Run mechanical simulation
    t, (Z, ng), _ = bls.simulate(Fdrive, Adrive, Qm)

    # Create figure
    fig, ax = plt.subplots(figsize=cm2inch(6, 6))
    fig.suptitle('Mechanical simulation', fontsize=12)
    for skey in ['bottom', 'left', 'right', 'top']:
        ax.spines[skey].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot variables and labels
    t_plot = np.insert(t, 0, -1e-6) * 1e6
    Pac = Adrive * np.sin(2 * np.pi * Fdrive * t + np.pi)  # Pa
    yvars = {'P_A': Pac * 1e-3, 'Z': Z * 1e9, 'n_g': ng * 1e22}
    dy = 1.2
    for i, ykey in enumerate(yvars.keys()):
        y = yvars[ykey]
        y_plot = rescale(np.insert(y, 0, y[0])) - dy * i
        ax.plot(t_plot, y_plot, color='k', linewidth=lw)
        ax.text(t_plot[0] - 0.1, y_plot[0], '$\mathregular{{{}}}$'.format(ykey), fontsize=fs,
                horizontalalignment='right', verticalalignment='center')

    # Acoustic pressure annotations
    ax.annotate(s='', xy=(1.5, 1.1), xytext=(3.5, 1.1),
                arrowprops=dict(arrowstyle='<|-|>', color='dimgrey'))
    ax.text(2.5, 1.12, '1/f', fontsize=fs, color='dimgrey',
            horizontalalignment='center', verticalalignment='bottom')
    ax.annotate(s='', xy=(1.5, -0.1), xytext=(1.5, 1),
                arrowprops=dict(arrowstyle='<|-|>', color='dimgrey'))
    ax.text(1.55, 0.4, '2A', fontsize=fs, color='dimgrey',
            horizontalalignment='left', verticalalignment='center')

    # Periodic stabilization patch
    ax.add_patch(Rectangle((2, -2 * dy - 0.1), 2, 2 * dy, color='dimgrey', alpha=0.3))
    ax.text(3, -2 * dy - 0.2, 'periodic\n stabilization', fontsize=fs, color='dimgrey',
            horizontalalignment='center', verticalalignment='top')
    # Z_last patch
    ax.add_patch(Rectangle((2, -dy - 0.1), 2, dy, edgecolor='k', facecolor='none', linestyle='--'))

    # ngeff annotations
    ax.text(t_plot[-1] + 0.1, y_plot[-1], '$\mathregular{n_{g,eff}}$', fontsize=fs, color='orange',
            horizontalalignment='left', verticalalignment='center')
    ax.scatter([t_plot[-1]], [y_plot[-1]], color='orange', s=ps)

    return fig


def plotCycleAveraging(bls, neuron, Fdrive, Adrive, Qm, fs=12, lw=2, ps=15):

    # Run mechanical simulation
    t, (Z, ng), _ = bls.simulate(Fdrive, Adrive, Qm)

    # Compute variables evolution over last acoustic cycle
    t_last = t[-NPC_FULL:] * 1e6  # us
    Z_last = Z[-NPC_FULL:]  # m
    Cm = bls.v_Capct(Z_last) * 1e2  # uF/m2
    Vm = Qm / Cm * 1e5  # mV
    yvars = {
        'C_m': Cm,  # uF/cm2
        'V_m': Vm,  # mV
        '\\alpha_m': neuron.alpham(Vm) * 1e3,  # ms-1
        '\\beta_m': neuron.betam(Vm) * 1e3,  # ms-1
        'p_\\infty / \\tau_p': neuron.pinf(Vm) / neuron.taup(Vm) * 1e3,  # ms-1
        '(1-p_\\infty) / \\tau_p': (1 - neuron.pinf(Vm)) / neuron.taup(Vm) * 1e3  # ms-1
    }

    # Create figure and axes
    fig, axes = plt.subplots(6, 1, figsize=cm2inch(4, 15))
    fig.suptitle('Cycle-averaging', fontsize=fs)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for skey in ['bottom', 'left', 'right', 'top']:
            ax.spines[skey].set_visible(False)

    # Plot variables
    for ax, ykey in zip(axes, yvars.keys()):
        ax.set_xticks([])
        ax.set_yticks([])
        for skey in ['bottom', 'left', 'right', 'top']:
            ax.spines[skey].set_visible(False)
        y = yvars[ykey]
        ax.plot(t_last, y, color='k', linewidth=lw)
        ax.plot([t_last[0], t_last[-1]], [np.mean(y)] * 2, '--', color='dimgrey')
        ax.scatter([t_last[-1]], [np.mean(y)], s=ps, color='dimgrey')
        ax.text(t_last[0] - 0.1, y[0], '$\mathregular{{{}}}$'.format(ykey), fontsize=fs,
                horizontalalignment='right', verticalalignment='center')

    return fig


def plotQtrace(nbls, Fdrive, Adrive, tstim, toffset, PRF, DC, fs=12, lw=2, ps=15):

    # Run effective simulation
    t, y, states = nbls.simulate(Fdrive, Adrive, tstim, toffset, PRF, DC, method='sonic')
    t *= 1e3  # ms
    Qm = y[2] * 1e5  # nC/cm2
    _, tpulse_on, tpulse_off = getStimPulses(t, states)

    # Add small onset
    t = np.insert(t, 0, -5.0)
    Qm = np.insert(Qm, 0, Qm[0])

    # Create figure and axes
    fig, ax = plt.subplots(figsize=cm2inch(12, 6))
    ax.set_xticks([])
    ax.set_yticks([])
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    for key in ['bottom', 'left']:
        ax.spines[key].set_position(('axes', -0.03))
        ax.spines[key].set_linewidth(2)

    # Plot Q-trace and stimulation pulses
    ax.plot(t, Qm, label='Qm', linewidth=lw)
    for ton, toff in zip(tpulse_on, tpulse_off):
        ax.axvspan(ton, toff, edgecolor='none', facecolor='#8A8A8A', alpha=0.2)
    ax.yaxis.set_tick_params(width=2)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.set_xlim((t.min(), t.max()))
    ax.set_xticks([])
    ax.set_xlabel('{}s'.format(si_format((t.max()) * 1e-3, space=' ')), fontsize=fs)
    ax.set_ylabel('$\\rm nC/cm^2$', fontsize=fs, labelpad=-15)
    ax.set_yticks(ax.get_ylim())
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.legend(fontsize=fs, frameon=False)
    return fig


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save output figures as pdf')

    args = ap.parse_args()
    if args.save:
        # Select output directory
        outdir = selectDirDialog()

    # Parameters
    neuron = CorticalRS()
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    Adrive = 100e3  # Pa
    PRF = 100.  # Hz
    DC = 0.5
    tstim = 150e-3  # s
    toffset = 100e-3  # s
    Qm = -71.9e-5  # C/cm2
    bls = BilayerSonophore(a, neuron.Cm0, neuron.Cm0 * neuron.Vm0 * 1e-3)
    nbls = NeuronalBilayerSonophore(a, neuron)

    # Figures
    fig2a = plotPmavg(bls, np.linspace(-0.4 * bls.Delta_, bls.a, 1000))
    fig2b = plotZoomVQ(nbls, Fdrive, Adrive)
    fig2c1 = plotMechSim(bls, Fdrive, Adrive, Qm)
    fig2c2 = plotCycleAveraging(bls, neuron, Fdrive, Adrive, Qm)
    fig2e = plotQtrace(nbls, Fdrive, Adrive, tstim, toffset, PRF, DC)

    if args.save:
        fig2a.savefig(os.path.join(outdir, 'fig2a.pdf'), transparent=True)
        fig2b.savefig(os.path.join(outdir, 'fig2b.pdf'), transparent=True)
        fig2c1.savefig(os.path.join(outdir, 'fig2c1.pdf'), transparent=True)
        fig2c2.savefig(os.path.join(outdir, 'fig2c2.pdf'), transparent=True)
        fig2e.savefig(os.path.join(outdir, 'fig2e.pdf'), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
