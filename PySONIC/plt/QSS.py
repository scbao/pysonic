
import inspect
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from ..postpro import getFixedPoints
from ..core import NeuronalBilayerSonophore, Batch
from .pltutils import *
from ..utils import logger, fileCache


root = '../../../QSS analysis'


def plotVarQSSDynamics(neuron, a, Fdrive, Adrive, charges, varname, varrange, fs=12):
    ''' Plot the QSS-approximated derivative of a specific variable as function of
        the variable itself, as well as equilibrium values, for various membrane
        charge densities at a given acoustic amplitude.

        :param neuron: neuron object
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param Adrive: US amplitude (Pa)
        :param charges: charge density vector (C/m2)
        :param varname: name of variable to plot
        :param varrange: range over which to compute the derivative
        :return: figure handle
    '''

    # Extract information about variable to plot
    pltvar = neuron.getPltVars()[varname]

    # Get methods to compute derivative and steady-state of variable of interest
    derX_func = getattr(neuron, 'der{}{}'.format(varname[0].upper(), varname[1:]))
    Xinf_func = getattr(neuron, '{}inf'.format(varname))
    derX_args = inspect.getargspec(derX_func)[0][1:]
    Xinf_args = inspect.getargspec(Xinf_func)[0][1:]

    # Get dictionary of charge and amplitude dependent QSS variables
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    _, Qref, lookups, QSS = nbls.quasiSteadyStates(
        Fdrive, amps=Adrive, charges=charges, squeeze_output=True)
    df = QSS
    df['Vm'] = lookups['V']

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title('{} neuron - QSS {} dynamics @ {:.2f} kPa'.format(
        neuron.name, pltvar['desc'], Adrive * 1e-3), fontsize=fs)
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
    if hasattr(neuron, y0_str):
        ax.axvline(getattr(neuron, y0_str) * pltvar.get('factor', 1),
                   label=y0_str, c='k', linewidth=0.5)

    # For each charge value
    icolor = 0
    for j, Qm in enumerate(charges):
        lbl = 'Q = {:.0f} nC/cm2'.format(Qm * 1e5)

        # Compute variable derivative as a function of its value, as well as equilibrium value,
        # keeping other variables at quasi steady-state
        derX_inputs = [varrange if arg == varname else df[arg][j] for arg in derX_args]
        Xinf_inputs = [df[arg][j] for arg in Xinf_args]
        dX_QSS = neuron.derCai(*derX_inputs)
        Xeq_QSS = neuron.Caiinf(*Xinf_inputs)

        # Plot variable derivative and its root as a function of the variable itself
        c = 'C{}'.format(icolor)
        ax.plot(varrange * pltvar.get('factor', 1), dX_QSS * pltvar.get('factor', 1), c=c, label=lbl)
        ax.axvline(Xeq_QSS * pltvar.get('factor', 1), linestyle='--', c=c)
        icolor += 1

    ax.legend(frameon=False, fontsize=fs - 3)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    fig.tight_layout()

    fig.canvas.set_window_title('{}_QSS_{}_dynamics_{:.2f}kPa'.format(
        neuron.name, varname, Adrive * 1e-3))

    return fig


def plotQSSvars(neuron, a, Fdrive, Adrive, fs=12):
    ''' Plot effective membrane potential, quasi-steady states and resulting membrane currents
        as a function of membrane charge density, for a given acoustic amplitude.

        :param neuron: neuron object
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param Adrive: US amplitude (Pa)
        :return: figure handle
    '''

    # Get neuron-specific pltvars
    pltvars = neuron.getPltVars()

    # Compute neuron-specific charge and amplitude dependent QS states at this amplitude
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    _, Qref, lookups, QSS = nbls.quasiSteadyStates(Fdrive, amps=Adrive, squeeze_output=True)
    Vmeff = lookups['V']

    # Compute QSS currents
    currents = neuron.currents(Vmeff, np.array([QSS[k] for k in neuron.states]))
    iNet = sum(currents.values())

    # Compute fixed points in dQdt profile
    dQdt = -iNet
    Q_SFPs = getFixedPoints(Qref, dQdt, filter='stable')
    Q_UFPs = getFixedPoints(Qref, dQdt, filter='unstable')

    # Extract dimensionless states
    norm_QSS = {}
    for x in neuron.states:
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
    figname = '{} neuron - QSS dynamics @ {:.2f} kPa'.format(neuron.name, Adrive * 1e-3)
    fig.suptitle(figname, fontsize=fs)

    # Subplot: Vmeff
    ax = axes[0]
    ax.set_ylabel('$V_m^*$ (mV)', fontsize=fs)
    ax.plot(Qref * 1e5, Vmeff, color='k')
    ax.axhline(neuron.Vm0, linewidth=0.5, color='k')

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
                label='$\\rm -{}$'.format(neuron.getPltVars()[k]['label']))
    ax.plot(Qref * 1e5, -iNet * 1e-3, color='k', label='$\\rm -I_{Net}$')
    ax.axhline(0, color='k', linewidth=0.5)
    if Q_SFPs.size > 0:
        ax.plot(Q_SFPs * 1e5, np.zeros(Q_SFPs.size), 'o', c='k', markersize=5, zorder=2)
    if Q_SFPs.size > 0:
        ax.plot(Q_UFPs * 1e5, np.zeros(Q_UFPs.size), 'o', c='k', markersize=5, mfc='none', zorder=2)

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    for ax in axes[1:]:
        ax.legend(loc='center right', fontsize=fs, frameon=False, bbox_to_anchor=(1.3, 0.5))
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    fig.canvas.set_window_title(
        '{}_QSS_states_vs_Qm_{:.2f}kPa'.format(neuron.name, Adrive * 1e-3))

    return fig


def plotQSSVarVsAmp(neuron, a, Fdrive, varname, amps=None, DC=1.,
                    fs=12, cmap='viridis', yscale='lin', zscale='lin'):
    ''' Plot a specific QSS variable (state or current) as a function of
        membrane charge density, for various acoustic amplitudes.

        :param neuron: neuron object
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param amps: US amplitudes (Pa)
        :param DC: duty cycle (-)
        :param varname: extraction key for variable to plot
        :return: figure handle
    '''

    # Determine stimulation modality
    if a is None and Fdrive is None:
        stim_type = 'elec'
        a = 32e-9
        Fdrive = 500e3
    else:
        stim_type = 'US'

    # Extract information about variable to plot
    pltvar = neuron.getPltVars()[varname]
    Qvar = neuron.getPltVars()['Qm']
    Afactor = {'US': 1e-3, 'elec': 1.}[stim_type]

    log = 'plotting {} neuron QSS {} vs. amp for {} stimulation @ {:.0f}% DC'.format(
        neuron.name, varname, stim_type, DC * 1e2)
    logger.info(log)

    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)

    # Get reference dictionaries for zero amplitude
    _, Qref, lookups0, QSS0 = nbls.quasiSteadyStates(Fdrive, amps=0., squeeze_output=True)
    Vmeff0 = lookups0['V']
    if stim_type == 'elec':  # if E-STIM case, compute steady states with constant capacitance
        Vmeff0 = Qref / neuron.Cm0 * 1e3
        QSS0 = neuron.steadyStates(Vmeff0)
    df0 = QSS0
    df0['Vm'] = Vmeff0

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    title = '{} neuron - {}steady-state {}'.format(
        neuron.name, 'quasi-' if amps is not None else '', pltvar['desc'])
    if amps is not None:
        title += '\nvs. {} amplitude @ {:.0f}% DC'.format(stim_type, DC * 1e2)
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
    if hasattr(neuron, y0_str):
        y0 = getattr(neuron, y0_str) * pltvar.get('factor', 1)
    elif varname in neuron.getCurrentsNames() + ['iNet', 'dQdt']:
        y0 = 0.
        y0_str = ''
    if y0 is not None:
        ax.axhline(y0, label=y0_str, c='k', linewidth=0.5)

    # Plot reference QSS profile of variable as a function of charge density
    var0 = extractPltVar(
        neuron, pltvar, pd.DataFrame({k: df0[k] for k in df0.keys()}), name=varname)
    ax.plot(Qref * Qvar['factor'], var0, '--', c='k', zorder=1,
            label='$\\rm A_{{{}}}=0$'.format(stim_type))

    # Define color code
    mymap = plt.get_cmap(cmap)
    zref = amps * Afactor
    if zscale == 'lin':
        norm = colors.Normalize(zref.min(), zref.max())
    elif zscale == 'log':
        norm = colors.LogNorm(zref.min(), zref.max())
    sm = cm.ScalarMappable(norm=norm, cmap=mymap)
    sm._A = []

    # Get amplitude-dependent QSS dictionary
    if stim_type == 'US':
        # Get dictionary of charge and amplitude dependent QSS variables
        _, Qref, lookups, QSS = nbls.quasiSteadyStates(
            Fdrive, amps=amps, DCs=DC, squeeze_output=True)
        df = QSS
        df['Vm'] = lookups['V']
    else:
        # Repeat zero-amplitude QSS dictionary for all amplitudes
        df = {k: np.tile(df0[k], (amps.size, 1)) for k in df0}

    # Plot QSS profiles for various amplitudes
    for i, A in enumerate(amps):
        var = extractPltVar(
            neuron, pltvar, pd.DataFrame({k: df[k][i] for k in df.keys()}), name=varname)
        if varname == 'dQdt' and stim_type == 'elec':
            var += A * DC * pltvar['factor']
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
        cbarax.set_ylabel(
            'Amplitude ({})'.format({'US': 'kPa', 'elec': 'mA/m2'}[stim_type]), fontsize=fs)
        for item in cbarax.get_yticklabels():
            item.set_fontsize(fs)

    title = '{}_{}SS_{}'.format(neuron.name, 'Q' if amps is not None else '', varname)
    if amps is not None:
        title += '_vs_{}A_{}_{:.0f}%DC'.format(zscale, stim_type, DC * 1e2)
    fig.canvas.set_window_title(title)

    return fig


@fileCache(
    root,
    lambda nbls, Fdrive, amps, DC: 'FPs_vs_Adrive_{}_{:.0f}kHz_{:.2f}-{:.2f}kPa_{:.0f}%DC'.format(
        nbls.neuron.name, Fdrive * 1e-3, amps.min() * 1e-3, amps.max() * 1e-3, DC * 1e2)
)
def getQSSFixedPointsvsAdrive(nbls, Fdrive, amps, DC, mpi=False, loglevel=logging.INFO):

    # Compute 2D QSS charge variation array
    _, Qref, lookups, QSS = nbls.quasiSteadyStates(
        Fdrive, amps=amps, DCs=DC, squeeze_output=True)
    dQdt = -nbls.neuron.iNet(lookups['V'], np.array([QSS[k] for k in nbls.neuron.states]))  # mA/m2

    # Generate batch queue
    queue = []
    for iA, Adrive in enumerate(amps):
        lookups1D = {k: v[iA, :] for k, v in lookups.items()}
        lookups1D['Q'] = Qref
        queue.append([Fdrive, Adrive, DC, lookups1D, dQdt[iA, :]])

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
    return nbls.neuron.getStabilizationValue(nbls.load(*args)[0])


@fileCache(
    root,
    lambda nbls, Fdrive, amps, tstim, toffset, PRF, DC:
        'stab_vs_Adrive_{}_{:.0f}kHz_{:.0f}ms_{:.0f}ms_offset_{:.0f}Hz_PRF_{:.2f}-{:.2f}kPa_{:.0f}%DC'.format(
            nbls.neuron.name, Fdrive * 1e-3, tstim * 1e3, toffset * 1e3, PRF,
            amps.min() * 1e-3, amps.max() * 1e-3, DC * 1e2)
)
def getSimFixedPointsvsAdrive(nbls, Fdrive, amps, tstim, toffset, PRF, DC,
                              outputdir=None, mpi=False, loglevel=logging.INFO):
    # Run batch to find stabilization point from simulations (if any) at each amplitude
    queue = [[nbls, outputdir, Fdrive, Adrive, tstim, toffset, PRF, DC, 'sonic'] for Adrive in amps]
    batch = Batch(runAndGetStab, queue)
    output = batch(mpi=mpi, loglevel=loglevel)
    return list(zip(amps, output))


def plotEqChargeVsAmp(neuron, a, Fdrive, amps=None, tstim=None, toffset=None, PRF=None,
                      DC=1., fs=12, xscale='lin', compdir=None, mpi=False,
                      loglevel=logging.INFO):
    ''' Plot the equilibrium membrane charge density as a function of acoustic amplitude,
        given an initial value of membrane charge density.

        :param neuron: neuron object
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param amps: US amplitudes (Pa)
        :return: figure handle
    '''

    # Determine stimulation modality
    stim_type = 'US'
    logger.info('plotting equilibrium charges for %s stimulation', stim_type)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    figname = '{} neuron - charge stability vs. amplitude @ {:.0f}%DC'.format(neuron.name, DC * 1e2)
    ax.set_title(figname)
    ax.set_xlabel('Amplitude ({})'.format({'US': 'kPa', 'elec': 'mA/m2'}[stim_type]),
                  fontsize=fs)
    ax.set_ylabel('$\\rm Q_m\ (nC/cm^2)$', fontsize=fs)
    if xscale == 'log':
        ax.set_xscale('log')
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)

    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
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
    ax.set_ylim(np.array([neuron.Qm0 - 10e-5, 0]) * 1e5)
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()

    fig.canvas.set_window_title('{}_QSS_Qstab_vs_{}A_{}_{:.0f}%DC{}'.format(
        neuron.name,
        xscale,
        stim_type,
        DC * 1e2,
        '_with_comp' if compdir is not None else ''
    ))

    return fig


@fileCache(
    root,
    lambda nbls, Fdrive, tstim, toffset, PRF, DCs:
        'threshold_curve_{}_{:.0f}kHz_{:.0f}ms_{:.0f}ms_offset_{:.0f}Hz_PRF_{:.2f}-{:.2f}%DC'.format(
            nbls.neuron.name, Fdrive * 1e-3, tstim * 1e3, toffset * 1e3, PRF,
            DCs.min() * 1e2, DCs.max() * 1e2)
)
def getSimThresholdAmps(nbls, Fdrive, tstim, toffset, PRF, DCs, mpi=False, loglevel=logging.INFO):
    # Run batch to find threshold amplitude from titrations at each DC
    queue = [[Fdrive, tstim, toffset, PRF, DC, 'sonic'] for DC in DCs]
    batch = Batch(nbls.titrate, queue)
    output = batch(mpi=mpi, loglevel=loglevel)
    return list(zip(DCs, output))


def plotQSSThresholdCurve(neuron, a, Fdrive, tstim=None, toffset=None, PRF=None, DCs=None,
                          fs=12, Ascale='lin', comp=False, mpi=False, loglevel=logging.INFO):

    print(DCs)
    logger.info('plotting %s neuron threshold curve', neuron.name)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    figname = '{} neuron - threshold amplitude vs. duty cycle'.format(neuron.name)
    ax.set_title(figname)
    ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    ax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    if Ascale == 'log':
        ax.set_yscale('log')
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)

    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    # Athrs_QSS = getQSSThresholdAmps(nbls, Fdrive, amps, DCs, mpi=mpi, loglevel=loglevel)
    # ax.plot(DCs * 1e2, Athrs_QSS * 1e-3, '-', c='k', label='QSS curve')
    if comp:
        Athrs_sim = getSimThresholdAmps(
            nbls, Fdrive, tstim, toffset, PRF, DCs, mpi=mpi, loglevel=loglevel)
        ax.plot(DCs * 1e2, Athrs_sim * 1e-3, '--', c='k', label='sim curve')

    # Post-process figure
    ax.set_ylim(np.array([neuron.Qm0 - 10e-5, 0]) * 1e5)
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()

    fig.canvas.set_window_title('{}_QSS_threhold_curve_{:.0f}-{:.0f}%DC_{}A_{}'.format(
        neuron.name,
        DCs.min() * 1e2,
        DCs.max() * 1e2,
        Ascale,
        '_with_comp' if comp else ''
    ))

    return fig
