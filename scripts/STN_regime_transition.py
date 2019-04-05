# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-04-05 17:39:08

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
from PySONIC.postpro import getFixedPoints
from PySONIC.neurons import getNeuronsDict
from PySONIC.plt import plotVarsQSS, plotQSSVarVsAmp, plotVarDynamics

# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Set logging level
logger.setLevel(logging.INFO)


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



def compareEqChargesQSSvsSim(inputdir, neuron, a, Fdrive, amps, tstim, fs=12):

    # Get charge value that cancels out net current in QSS approx.
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    _, Qref, Vmeff, QS_states = nbls.quasiSteadyStates(Fdrive, amps=amps)
    iNet = neuron.iNet(Vmeff, QS_states)

    # For each amplitude, take the max of detected stable fixed points
    # (since stabilization occurs during repolarization phase)
    Qeq_QSS = np.empty(amps.size)
    for i, Adrive in enumerate(amps):
        SFPs = getFixedPoints(Qref, -iNet[i, :])
        Qeq_QSS[i] = SFPs.max() if SFPs is not None else np.nan

    # Get sabilization charge value in simulations
    _, Qeq_sim, _ = getChargeStabilizationFromSims(inputdir, neuron, a, Fdrive, amps, tstim)

    Q_rmse = np.sqrt(np.nanmean((Qeq_sim - Qeq_QSS)**2))
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

    ax.plot(amps * 1e-3, Qeq_QSS * 1e5, '.', c='C0', label='QSS approximation')
    ax.plot(amps * 1e-3, Qeq_sim * 1e5, '.', c='C1',
            label='end of {:.2f} s stimulus (simulation)'.format(tstim))
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()

    fig.canvas.set_window_title(
        '{}_Qeq_vs_amp_{:.0f}s'.format(neuron.name, tstim))

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
    tstim = 3.0  # s
    Cai_range = np.logspace(np.log10(neuron.Cai0 * 1e-4), np.log10(neuron.Cai0 * 1e3), 100)
    charges = np.array([neuron.Qbounds()[1], neuron.Vm0 * neuron.Cm0 * 1e-3])

    figs = []
    if 'a' in figset:
        for Adrive in [21.35e3]: #[amps[0], amps[amps.size // 2], amps[-1]]:
            figs += [
                plotVarsQSS(neuron, a, Fdrive, Adrive),
                plotVarDynamics(neuron, a, Fdrive, Adrive, charges, 'Cai', Cai_range)
            ]
    if 'b' in figset:
        figs += [
            plotQSSVarVsAmp(neuron, a, Fdrive, 'Cai', amps=amps, yscale='log'),
            plotQSSVarVsAmp(neuron, a, Fdrive, 'iNet', amps=amps)
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
