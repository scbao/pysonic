# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:24:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-15 21:17:51

import inspect
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ..core import NeuronalBilayerSonophore, Batch
from .pltutils import *
from ..utils import logger, fileCache


root = '../../../QSS analysis/data'


def plotVarQSSDynamics(pneuron, a, Fdrive, Adrive, charges, varname, varrange, fs=12):
    ''' Plot the QSS-approximated derivative of a specific variable as function of
        the variable itself, as well as equilibrium values, for various membrane
        charge densities at a given acoustic amplitude.

        :param pneuron: point-neuron model
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param Adrive: US amplitude (Pa)
        :param charges: charge density vector (C/m2)
        :param varname: name of variable to plot
        :param varrange: range over which to compute the derivative
        :return: figure handle
    '''

    # Extract information about variable to plot
    pltvar = pneuron.getPltVars()[varname]

    # Get methods to compute derivative and steady-state of variable of interest
    derX_func = getattr(pneuron, 'der{}{}'.format(varname[0].upper(), varname[1:]))
    Xinf_func = getattr(pneuron, '{}inf'.format(varname))
    derX_args = inspect.getargspec(derX_func)[0][1:]
    Xinf_args = inspect.getargspec(Xinf_func)[0][1:]

    # Get dictionary of charge and amplitude dependent QSS variables
    nbls = NeuronalBilayerSonophore(a, pneuron, Fdrive)
    _, Qref, lookups, QSS = nbls.getQuasiSteadyStates(
        Fdrive, amps=Adrive, charges=charges, squeeze_output=True)
    df = QSS
    df['Vm'] = lookups['V']

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title('{} neuron - QSS {} dynamics @ {:.2f} kPa'.format(
        pneuron.name, pltvar['desc'], Adrive * 1e-3), fontsize=fs)
    ax.set_xscale('log')
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)
    ax.set_xlabel('$\\rm {}\ ({})$'.format(pltvar['label'], pltvar.get('unit', '')),
                  fontsize=fs)
    ax.set_ylabel('$\\rm QSS\ d{}/dt\ ({}/s)$'.format(pltvar['label'], pltvar.get('unit', '1')),
                  fontsize=fs)
    ax.set_ylim(-40, 40)
    ax.axhline(0, c='k', linewidth=0.5)

    y0_str = '{}0'.format(varname)
    if hasattr(pneuron, y0_str):
        ax.axvline(getattr(pneuron, y0_str) * pltvar.get('factor', 1),
                   label=y0_str, c='k', linewidth=0.5)

    # For each charge value
    icolor = 0
    for j, Qm in enumerate(charges):
        lbl = 'Q = {:.0f} nC/cm2'.format(Qm * 1e5)

        # Compute variable derivative as a function of its value, as well as equilibrium value,
        # keeping other variables at quasi steady-state
        derX_inputs = [varrange if arg == varname else df[arg][j] for arg in derX_args]
        Xinf_inputs = [df[arg][j] for arg in Xinf_args]
        dX_QSS = pneuron.derCai(*derX_inputs)
        Xeq_QSS = pneuron.Caiinf(*Xinf_inputs)

        # Plot variable derivative and its root as a function of the variable itself
        c = 'C{}'.format(icolor)
        ax.plot(varrange * pltvar.get('factor', 1), dX_QSS * pltvar.get('factor', 1),
                c=c, label=lbl)
        ax.axvline(Xeq_QSS * pltvar.get('factor', 1), linestyle='--', c=c)
        icolor += 1

    ax.legend(frameon=False, fontsize=fs - 3)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    fig.tight_layout()

    fig.canvas.set_window_title('{}_QSS_{}_dynamics_{:.2f}kPa'.format(
        pneuron.name, varname, Adrive * 1e-3))

    return fig


def plotQSSdynamics(pneuron, a, Fdrive, Adrive, DC=1., fs=12):
    ''' Plot effective membrane potential, quasi-steady states and resulting membrane currents
        as a function of membrane charge density, for a given acoustic amplitude.

        :param pneuron: point-neuron model
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param Adrive: US amplitude (Pa)
        :return: figure handle
    '''

    # Get neuron-specific pltvars
    pltvars = pneuron.getPltVars()

    # Compute neuron-specific charge and amplitude dependent QS states at this amplitude
    nbls = NeuronalBilayerSonophore(a, pneuron, Fdrive)
    lookups, QSS = nbls.getQuasiSteadyStates(Fdrive, amps=Adrive, DCs=DC, squeeze_output=True)
    Qref = lookups.refs['Q']
    Vmeff = lookups['V']

    # Compute QSS currents and 1D charge variation array
    states = {k: QSS[k] for k in pneuron.states}
    currents = {name: cfunc(Vmeff, states) for name, cfunc in pneuron.currents().items()}
    iNet = sum(currents.values())
    dQdt = -iNet

    # Compute stable and unstable fixed points
    Q_SFPs, Q_UFPs = nbls.fixedPointsQSS(Fdrive, Adrive, DC, lookups, dQdt)

    # Extract dimensionless states
    norm_QSS = {}
    for x in pneuron.states:
        if 'unit' not in pltvars[x]:
            norm_QSS[x] = QSS[x]

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(7, 9))
    axes[-1].set_xlabel('$\\rm Q_m\ (nC/cm^2)$', fontsize=fs)
    for ax in axes:
        for skey in ['top', 'right']:
            ax.spines[skey].set_visible(False)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
        for item in ax.get_xticklabels(minor=True):
            item.set_visible(False)
    fig.suptitle('{} neuron - QSS dynamics @ {:.2f} kPa, {:.0f}%DC'.format(
        pneuron.name, Adrive * 1e-3, DC * 1e2), fontsize=fs)

    # Subplot: Vmeff
    ax = axes[0]
    ax.set_ylabel('$V_m^*$ (mV)', fontsize=fs)
    ax.plot(Qref * 1e5, Vmeff, color='k')
    ax.axhline(pneuron.Vm0, linewidth=0.5, color='k')

    # Subplot: dimensionless quasi-steady states
    cset = plt.get_cmap('Dark2').colors + plt.get_cmap('tab10').colors
    ax = axes[1]
    ax.set_ylabel('QSS gating variables (-)', fontsize=fs)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([-0.05, 1.05])
    for i, (label, QS_state) in enumerate(norm_QSS.items()):
        ax.plot(Qref * 1e5, QS_state, label=label, c=cset[i])

    # Subplot: currents
    ax = axes[2]
    cset = plt.get_cmap('tab10').colors
    ax.set_ylabel('QSS currents ($\\rm A/m^2$)', fontsize=fs)
    for i, (k, I) in enumerate(currents.items()):
        ax.plot(Qref * 1e5, -I * 1e-3, '--', c=cset[i],
                label='$\\rm -{}$'.format(pneuron.getPltVars()[k]['label']))
    ax.plot(Qref * 1e5, -iNet * 1e-3, color='k', label='$\\rm -I_{Net}$')
    ax.axhline(0, color='k', linewidth=0.5)

    if len(Q_SFPs) > 0:
        ax.scatter(np.array(Q_SFPs) * 1e5, np.zeros(len(Q_SFPs)),
                   marker='.', s=200, facecolors='g', edgecolors='none',
                   label='QSS stable FPs', zorder=3)
    if len(Q_UFPs) > 0:
        ax.scatter(np.array(Q_UFPs) * 1e5, np.zeros(len(Q_UFPs)),
                   marker='.', s=200, facecolors='r', edgecolors='none',
                   label='QSS unstable FPs', zorder=3)

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    for ax in axes[1:]:
        ax.legend(loc='center right', fontsize=fs, frameon=False, bbox_to_anchor=(1.3, 0.5))
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    fig.canvas.set_window_title(
        '{}_QSS_dynamics_vs_Qm_{:.2f}kPa_DC{:.0f}%'.format(pneuron.name, Adrive * 1e-3, DC * 1e2))

    return fig


def plotQSSVarVsQm(pneuron, a, Fdrive, varname, amps=None, DC=1.,
                   fs=12, cmap='viridis', yscale='lin', zscale='lin',
                   mpi=False, loglevel=logging.INFO):
    ''' Plot a specific QSS variable (state or current) as a function of
        membrane charge density, for various acoustic amplitudes.

        :param pneuron: point-neuron model
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param amps: US amplitudes (Pa)
        :param DC: duty cycle (-)
        :param varname: extraction key for variable to plot
        :return: figure handle
    '''

    # Extract information about variable to plot
    pltvar = pneuron.getPltVars()[varname]
    Qvar = pneuron.getPltVars()['Qm']
    Afactor = 1e-3

    logger.info('plotting %s neuron QSS %s vs. Qm for various amplitudes @ %.0f%% DC',
                pneuron.name, pltvar['desc'], DC * 1e2)

    nbls = NeuronalBilayerSonophore(a, pneuron, Fdrive)

    # Get reference dictionaries for zero amplitude
    lookups0, QSS0 = nbls.getQuasiSteadyStates(Fdrive, amps=0., squeeze_output=True)
    Vmeff0 = lookups0['V']
    Qref = lookups0.refs['Q']
    df0 = QSS0.tables
    df0['Vm'] = Vmeff0

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    title = '{} neuron - QSS {} vs. Qm - {:.0f}% DC'.format(pneuron.name, varname, DC * 1e2)
    ax.set_title(title, fontsize=fs)
    ax.set_xlabel('$\\rm {}\ ({})$'.format(Qvar['label'], Qvar['unit']), fontsize=fs)
    ax.set_ylabel('$\\rm QSS\ {}\ ({})$'.format(pltvar['label'], pltvar.get('unit', '')),
                  fontsize=fs)
    if yscale == 'log':
        ax.set_yscale('log')
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)

    # Plot y-variable reference line, if any
    y0 = None
    y0_str = '{}0'.format(varname)
    if hasattr(pneuron, y0_str):
        y0 = getattr(pneuron, y0_str) * pltvar.get('factor', 1)
    elif varname in pneuron.getCurrentsNames() + ['iNet', 'dQdt']:
        y0 = 0.
        y0_str = ''
    if y0 is not None:
        ax.axhline(y0, label=y0_str, c='k', linewidth=0.5)

    # Plot reference QSS profile of variable as a function of charge density
    var0 = extractPltVar(
        pneuron, pltvar, pd.DataFrame({k: df0[k] for k in df0.keys()}), name=varname)
    ax.plot(Qref * Qvar['factor'], var0, '--', c='k', zorder=1, label='A = 0')

    if varname == 'dQdt':
        # Plot charge SFPs and UFPs for each acoustic amplitude
        SFPs, UFPs = getQSSFixedPointsvsAdrive(
            nbls, Fdrive, amps, DC, mpi=mpi, loglevel=loglevel)
        if len(SFPs) > 0:
            _, Q_SFPs = np.array(SFPs).T
            ax.scatter(np.array(Q_SFPs) * 1e5, np.zeros(len(Q_SFPs)),
                       marker='.', s=100, facecolors='g', edgecolors='none',
                       label='QSS stable fixed points')
        if len(UFPs) > 0:
            _, Q_UFPs = np.array(UFPs).T
            ax.scatter(np.array(Q_UFPs) * 1e5, np.zeros(len(Q_UFPs)),
                       marker='.', s=100, facecolors='r', edgecolors='none',
                       label='QSS unstable fixed points')

    # Define color code
    mymap = plt.get_cmap(cmap)
    zref = amps * Afactor
    norm, sm = setNormalizer(mymap, (zref.min(), zref.max()), zscale)

    # Get amplitude-dependent QSS dictionary
    lookups, QSS = nbls.getQuasiSteadyStates(
        Fdrive, amps=amps, DCs=DC, squeeze_output=True)
    df = QSS.tables
    df['Vm'] = lookups['V']

    # Plot QSS profiles for various amplitudes
    for i, A in enumerate(amps):
        var = extractPltVar(
            pneuron, pltvar, pd.DataFrame({k: df[k][i] for k in df.keys()}), name=varname)
        ax.plot(Qref * Qvar['factor'], var, c=sm.to_rgba(A * Afactor), zorder=0)

    # Add legend and adjust layout
    ax.legend(frameon=False, fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15, top=0.9, right=0.80, hspace=0.5)

    # Plot amplitude colorbar
    if amps is not None:
        cbarax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
        fig.colorbar(sm, cax=cbarax)
        cbarax.set_ylabel('Amplitude (kPa)', fontsize=fs)
        for item in cbarax.get_yticklabels():
            item.set_fontsize(fs)

    fig.canvas.set_window_title('{}_QSS_{}_vs_Qm_{}A_{:.2f}-{:.2f}kPa_DC{:.0f}%'.format(
        pneuron.name, varname, zscale, amps.min() * 1e-3, amps.max() * 1e-3, DC * 1e2))

    return fig


@fileCache(
    root,
    lambda nbls, Fdrive, amps, DC:
        '{}_QSS_FPs_{:.0f}kHz_{:.2f}-{:.2f}kPa_DC{:.0f}%'.format(
            nbls.pneuron.name, Fdrive * 1e-3, amps.min() * 1e-3, amps.max() * 1e-3, DC * 1e2)
)
def getQSSFixedPointsvsAdrive(nbls, Fdrive, amps, DC, mpi=False, loglevel=logging.INFO):

    # Compute 2D QSS charge variation array
    lkp2d, QSS = nbls.getQuasiSteadyStates(
        Fdrive, amps=amps, DCs=DC, squeeze_output=True)
    dQdt = -nbls.pneuron.iNet(lkp2d['V'], QSS.tables)  # mA/m2

    # Generate batch queue
    queue = []
    for iA, Adrive in enumerate(amps):
        lkp1d = lkp2d.project('A', Adrive)
        queue.append([Fdrive, Adrive, DC, lkp1d, dQdt[iA, :]])

    # Run batch to find stable and unstable fixed points at each amplitude
    batch = Batch(nbls.fixedPointsQSS, queue)
    output = batch(mpi=mpi, loglevel=loglevel)

    # Sort points by amplitude
    SFPs, UFPs = [], []
    for i, Adrive in enumerate(amps):
        SFPs += [(Adrive, Qm) for Qm in output[i][0]]
        UFPs += [(Adrive, Qm) for Qm in output[i][1]]
    return SFPs, UFPs


def runAndGetStab(nbls, *args):
    args = list(args[:-1]) + [1., args[-1]]  # hacking coverage fraction into args
    return nbls.pneuron.getStabilizationValue(nbls.getOutput(*args)[0])


@fileCache(
    root,
    lambda nbls, Fdrive, amps, tstim, toffset, PRF, DC:
        '{}_sim_FPs_{:.0f}kHz_{:.0f}ms_offset{:.0f}ms_PRF{:.0f}Hz_{:.2f}-{:.2f}kPa_DC{:.0f}%'.format(
            nbls.pneuron.name, Fdrive * 1e-3, tstim * 1e3, toffset * 1e3, PRF,
            amps.min() * 1e-3, amps.max() * 1e-3, DC * 1e2)
)
def getSimFixedPointsvsAdrive(nbls, Fdrive, amps, tstim, toffset, PRF, DC,
                              outputdir=None, mpi=False, loglevel=logging.INFO):
    # Run batch to find stabilization point from simulations (if any) at each amplitude
    queue = [[nbls, outputdir, Fdrive, Adrive, tstim, toffset, PRF, DC, 'sonic'] for Adrive in amps]
    batch = Batch(runAndGetStab, queue)
    output = batch(mpi=mpi, loglevel=loglevel)
    return list(zip(amps, output))


def plotEqChargeVsAmp(pneuron, a, Fdrive, amps=None, tstim=None, toffset=None, PRF=None,
                      DC=1., fs=12, xscale='lin', compdir=None, mpi=False,
                      loglevel=logging.INFO):
    ''' Plot the equilibrium membrane charge density as a function of acoustic amplitude,
        given an initial value of membrane charge density.

        :param pneuron: point-neuron model
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param amps: US amplitudes (Pa)
        :return: figure handle
    '''

    logger.info('plotting equilibrium charges for various amplitudes')

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    figname = '{} neuron - charge stability vs. amplitude @ {:.0f}%DC'.format(
        pneuron.name, DC * 1e2)
    ax.set_title(figname)
    ax.set_xlabel('Amplitude (kPa)', fontsize=fs)
    ax.set_ylabel('$\\rm Q_m\ (nC/cm^2)$', fontsize=fs)
    if xscale == 'log':
        ax.set_xscale('log')
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)

    nbls = NeuronalBilayerSonophore(a, pneuron, Fdrive)
    Afactor = 1e-3

    # Plot charge SFPs and UFPs for each acoustic amplitude
    SFPs, UFPs = getQSSFixedPointsvsAdrive(
        nbls, Fdrive, amps, DC, mpi=mpi, loglevel=loglevel)
    if len(SFPs) > 0:
        A_SFPs, Q_SFPs = np.array(SFPs).T
        ax.scatter(np.array(A_SFPs) * Afactor, np.array(Q_SFPs) * 1e5,
                   marker='.', s=20, facecolors='g', edgecolors='none',
                   label='QSS stable fixed points')
    if len(UFPs) > 0:
        A_UFPs, Q_UFPs = np.array(UFPs).T
        ax.scatter(np.array(A_UFPs) * Afactor, np.array(Q_UFPs) * 1e5,
                   marker='.', s=20, facecolors='r', edgecolors='none',
                   label='QSS unstable fixed points')

    # Plot charge asymptotic stabilization points from simulations for each acoustic amplitude
    if compdir is not None:
        stab_points = getSimFixedPointsvsAdrive(
            nbls, Fdrive, amps, tstim, toffset, PRF, DC,
            outputdir=compdir, mpi=mpi, loglevel=loglevel)
        if len(stab_points) > 0:
            A_stab, Q_stab = np.array(stab_points).T
            ax.scatter(np.array(A_stab) * Afactor, np.array(Q_stab) * 1e5,
                       marker='o', s=20, facecolors='none', edgecolors='k',
                       label='stabilization points from simulations')

    # Post-process figure
    ax.set_ylim(np.array([pneuron.Qm0() - 10e-5, 0]) * 1e5)
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()

    fig.canvas.set_window_title('{}_QSS_Qstab_vs_{}A_{:.0f}%DC{}'.format(
        pneuron.name,
        xscale,
        DC * 1e2,
        '_with_comp' if compdir is not None else ''
    ))

    return fig


@fileCache(
    root,
    lambda nbls, Fdrive, DCs:
        '{}_QSS_threshold_curve_{:.0f}kHz_DC{:.2f}-{:.2f}%'.format(
            nbls.pneuron.name, Fdrive * 1e-3, DCs.min() * 1e2, DCs.max() * 1e2),
    ext='csv'
)
def getQSSThresholdAmps(nbls, Fdrive, DCs, mpi=False, loglevel=logging.INFO):
    queue = [[Fdrive, DC] for DC in DCs]
    batch = Batch(nbls.titrateQSS, queue)
    return batch(mpi=mpi, loglevel=loglevel)


@fileCache(
    root,
    lambda nbls, Fdrive, tstim, toffset, PRF, DCs:
        '{}_sim_threshold_curve_{:.0f}kHz_{:.0f}ms_offset{:.0f}ms_PRF{:.0f}Hz_DC{:.2f}-{:.2f}%'.format(
            nbls.pneuron.name, Fdrive * 1e-3, tstim * 1e3, toffset * 1e3, PRF,
            DCs.min() * 1e2, DCs.max() * 1e2),
    ext='csv'
)
def getSimThresholdAmps(nbls, Fdrive, tstim, toffset, PRF, DCs, mpi=False, loglevel=logging.INFO):
    # Run batch to find threshold amplitude from titrations at each DC
    queue = [[Fdrive, tstim, toffset, PRF, DC, 'sonic'] for DC in DCs]
    batch = Batch(nbls.titrate, queue)
    return batch(mpi=mpi, loglevel=loglevel)


def plotQSSThresholdCurve(pneuron, a, Fdrive, tstim=None, toffset=None, PRF=None, DCs=None,
                          fs=12, Ascale='lin', comp=False, mpi=False, loglevel=logging.INFO):

    logger.info('plotting %s neuron threshold curve', pneuron.name)

    if pneuron.name == 'STN':
        raise ValueError('cannot compute threshold curve for STN neuron')

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    figname = '{} neuron - threshold amplitude vs. duty cycle'.format(pneuron.name)
    ax.set_title(figname)
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    if Ascale == 'log':
        ax.set_yscale('log')
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)

    nbls = NeuronalBilayerSonophore(a, pneuron, Fdrive)
    Athrs_QSS = np.array(getQSSThresholdAmps(nbls, Fdrive, DCs, mpi=mpi, loglevel=loglevel))
    ax.plot(DCs * 1e2, Athrs_QSS * 1e-3, '-', c='k', label='QSS curve')
    if comp:
        Athrs_sim = np.array(getSimThresholdAmps(
            nbls, Fdrive, tstim, toffset, PRF, DCs, mpi=mpi, loglevel=loglevel))
        ax.plot(DCs * 1e2, Athrs_sim * 1e-3, '--', c='k', label='sim curve')

    # Post-process figure
    ax.set_xlim([0, 100])
    ax.set_ylim([10, 600])
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()

    fig.canvas.set_window_title('{}_QSS_threhold_curve_{:.0f}-{:.0f}%DC_{}A{}'.format(
        pneuron.name,
        DCs.min() * 1e2,
        DCs.max() * 1e2,
        Ascale,
        '_with_comp' if comp else ''
    ))

    return fig
