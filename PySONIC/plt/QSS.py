
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from ..postpro import getFixedPoints, getEqPoint1D
from ..core import NeuronalBilayerSonophore
from .pltutils import *
from ..constants import TITRATION_T_OFFSET
from ..utils import logger


def plotVarDynamics(neuron, a, Fdrive, Adrive, charges, varname, varrange, fs=12):
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
    _, Qref, Vmeff, QS_states = nbls.quasiSteadyStates(Fdrive, amps=Adrive, charges=charges)
    df = {k: QS_states[i] for i, k in enumerate(neuron.states)}
    df['Vm'] = Vmeff

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


def plotVarsQSS(neuron, a, Fdrive, Adrive, fs=12):
    ''' Plot effective membrane potential, quasi-steady states and resulting membrane currents
        as a function of membrane charge density, for a given acoustic amplitudes.

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
    _, Qref, Vmeff, QS_states = nbls.quasiSteadyStates(Fdrive, amps=Adrive)

    # Compute QSS currents
    currents = neuron.currents(Vmeff, QS_states)
    iNet = sum(currents.values())

    # Extract dimensionless states
    norm_QS_states = {}
    for i, label in enumerate(neuron.states):
        if 'unit' not in pltvars[label]:
            norm_QS_states[label] = QS_states[i]

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

    # Subplot: Vmeff
    ax = axes[0]
    ax.set_ylabel('$V_m^*$ (mV)', fontsize=fs)
    ax.plot(Qref * 1e5, Vmeff, color='k')
    ax.axhline(neuron.Vm0, linewidth=0.5, color='k')

    # Subplot: dimensionless quasi-steady states
    cset = plt.get_cmap('tab10').colors + plt.get_cmap('Dark2').colors
    ax = axes[1]
    ax.set_ylabel('$X_\infty$', fontsize=fs)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([-0.05, 1.05])
    for i, (label, QS_state) in enumerate(norm_QS_states.items()):
        ax.plot(Qref * 1e5, QS_state, label=label, c=cset[i])

    # Subplot: currents
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


def plotQSSVarVsAmp(neuron, a, Fdrive, varname, amps=None, DC=1., Qi=None,
                    plotQi=True, fs=12, cmap='viridis', yscale='lin', zscale='lin'):
    ''' Plot a specific QSS variable (state or current) as a function of
        membrane charge density, for various acoustic amplitudes.

        :param neuron: neuron object
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param amps: US amplitudes (Pa)
        :param DC: duty cycle (-)
        :param Qi: initial membrane charge density for phase-plane analysis (C/m2)
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
    Qzeros = []

    logger.info('plotting %s %s QSS profiles for %s stimulation @ %.0f%% DC',
                neuron.name, varname, stim_type, DC * 1e2)

    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)

    # Get reference QSS dictionary for zero amplitude
    _, Qref, Vmeff0, QS_states0 = nbls.quasiSteadyStates(Fdrive, amps=0.)
    if stim_type == 'elec':  # if E-STIM case, compute steady states with constant capacitance
        Vmeff0 = Qref / neuron.Cm0 * 1e3
        QS_states0 = np.array(list(map(neuron.steadyStates, Vmeff0))).T
    df0 = {k: QS_states0[i] for i, k in enumerate(neuron.states)}
    df0['Vm'] = Vmeff0

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title('{} neuron - {:.0f} % DC {} stim\nquasi steady-state {} vs. amplitude'.format(
        neuron.name, DC * 1e2, stim_type, pltvar['desc']), fontsize=fs)
    ax.set_xlabel('$\\rm {}\ ({})$'.format(Qvar['label'], Qvar['unit']), fontsize=fs)
    ax.set_ylabel('$\\rm QSS\ {}\ ({})$'.format(pltvar['label'], pltvar.get('unit', '')),
                  fontsize=fs)
    if yscale == 'log':
        ax.set_yscale('log')
    for key in ['top', 'right']:
        ax.spines[key].set_visible(False)

    # Plot charge starting point, if any
    if plotQi and Qi is not None:
        ax.axvline(Qi * Qvar['factor'], label='$\\rm Q_{m,i}$', c='silver')

    # Plot y-variable reference line, if any
    y0_str = '{}0'.format(varname)
    if hasattr(neuron, y0_str):
        y0 = getattr(neuron, y0_str) * pltvar.get('factor', 1)
    elif varname == 'iNet':
        y0 = 0.
        y0_str = ''
    ax.axhline(y0, label=y0_str, c='k', linewidth=0.5)

    # Plot reference QSS profile of variable as a function of charge density
    var0 = extractPltVar(
        neuron, pltvar, pd.DataFrame({k: df0[k] for k in df0.keys()}), name=varname)
    if varname == 'iNet':
        var0 = -var0

    if amps is None:
        # Plot arrows on ref profile
        nQ_sparse = 25
        Qsparse = np.linspace(Qref.min(), Qref.max(), nQ_sparse) * Qvar['factor']
        var0_sparse = np.interp(Qsparse, Qref * Qvar['factor'], var0)
        var0_signs = np.sign(var0_sparse)
        arrow_colors = [{-1: 'r', 1: 'g'}[s] for s in var0_signs]
        ax.quiver(Qsparse, var0_sparse, var0_signs, np.zeros(Qsparse.size),
                  color=arrow_colors, units='width', scale=20., width=0.005)
    ax.plot(Qref * Qvar['factor'], var0, c='k', zorder=1,
            label='$\\rm resting\ {}\ (A=0)$'.format(pltvar['label']))

    if varname == 'iNet':
        if Qi is None:
            Qzeros += getFixedPoints(Qref, var0).tolist()
        else:
            Qzeros += [getEqPoint1D(Qref, var0, Qi)]

    # Plot QSS profile of variable as a function of charge density for various amplitudes
    if amps is not None:

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
            _, Qref, Vmeff, QS_states = nbls.quasiSteadyStates(Fdrive, amps=amps, DCs=DC)
            df = {k: QS_states[i] for i, k in enumerate(neuron.states)}
            df['Vm'] = Vmeff
        else:
            # Repeat zero-amplitude QSS dictionary for all amplitudes
            df = {k: np.tile(df0[k], (amps.size, 1)) for k in df0}

        # Plot QSS profiles for various amplitudes
        for i, A in enumerate(amps):
            var = extractPltVar(
                neuron, pltvar, pd.DataFrame({k: df[k][i] for k in df.keys()}), name=varname)
            if varname == 'iNet':
                var = -var
                if stim_type == 'elec':
                    var += A * DC * pltvar['factor']
            ax.plot(Qref * Qvar['factor'], var, c=sm.to_rgba(A * Afactor), zorder=0)
            if varname == 'iNet':
                # mark eq. point if starting point provided, otherwise mark all SFPs
                if Qi is None:
                    SFPs = getFixedPoints(Qref, var)
                    if SFPs is not None:
                        Qzeros += SFPs.tolist()
                else:
                    Qzeros += [getEqPoint1D(Qref, var, Qi)]

    # Plot fixed-points, if any
    if len(Qzeros) > 0:
        if amps is not None:
            ax.plot(np.array(Qzeros) * Qvar['factor'], np.zeros(len(Qzeros)), '.', c='k', zorder=2,
                    label='$\\rm Q_{m,eq}$' if Qi is not None else '$\\rm Q_m\ SPFs$')
        else:
            for Qz in Qzeros:
                ax.axvline(Qz * Qvar['factor'], linestyle='--', c='k', zorder=2)

    # Add legend and adjust layout
    if amps is not None:
        ax.legend(frameon=False, fontsize=fs)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    fig.tight_layout()

    # Plot amplitude colorbar
    if amps is not None:
        fig.subplots_adjust(bottom=0.15, top=0.9, right=0.80, hspace=0.5)
        cbarax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
        fig.colorbar(sm, cax=cbarax)
        cbarax.set_ylabel(
            'Amplitude ({})'.format({'US': 'kPa', 'elec': 'mA/m2'}[stim_type]), fontsize=fs)
        for item in cbarax.get_yticklabels():
            item.set_fontsize(fs)

    fig.canvas.set_window_title('{}_QSS_{}_vs_{}A_{}_{:.0f}%DC'.format(
        neuron.name, varname, zscale, stim_type, DC * 1e2))

    return fig


def plotEqChargeVsAmp(neurons, a, Fdrive, amps=None, tstim=250e-3, PRF=100.0,
                      DCs=[1.], Qi=None, fs=12, xscale='lin', titrate=False):
    ''' Plot the equilibrium membrane charge density as a function of acoustic amplitude,
        given an initial value of membrane charge density.

        :param neurons: neuron objects
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param amps: US amplitudes (Pa)
        :param Qi: initial membrane charge density for phase-plane analysis (C/m2)
        :return: figure handle
    '''

    # Determine stimulation modality
    if a is None and Fdrive is None:
        stim_type = 'elec'
        a = 32e-9
        Fdrive = 500e3
    else:
        stim_type = 'US'

    logger.info('plotting equilibrium charges for %s stimulation', stim_type)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))
    figname = 'equilibrium charge vs. amplitude'
    ax.set_title(figname)
    ax.set_xlabel('Amplitude ({})'.format({'US': 'kPa', 'elec': 'mA/m2'}[stim_type]),
                  fontsize=fs)
    if Qi[0] is not None:
        ax.set_ylabel('$\\rm Q_{m, eq}\ (nC/cm^2)$', fontsize=fs)
    else:
        ax.set_ylabel('$\\rm Q_m\ SPFs\ (nC/cm^2)$', fontsize=fs)
    if xscale == 'log':
        ax.set_xscale('log')
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)

    Qrange = (np.inf, -np.inf)

    icolor = 0
    for i, neuron in enumerate(neurons):

        nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)

        # Compute reference charge variation array for zero amplitude
        _, Qref, Vmeff0, QS_states0 = nbls.quasiSteadyStates(Fdrive, amps=0.)
        Qrange = (min(Qrange[0], Qref.min()), max(Qrange[1], Qref.max()))
        if stim_type == 'elec':  # if E-STIM case, compute steady states with constant capacitance
            Vmeff0 = Qref / neuron.Cm0 * 1e3
            QS_states0 = np.array(list(map(neuron.steadyStates, Vmeff0))).T
        dQdt0 = -neuron.iNet(Vmeff0, QS_states0)  # mA/m2

        # Compute 3D QSS charge variation array
        if stim_type == 'US':
            Aref, _, Vmeff, QS_states = nbls.quasiSteadyStates(Fdrive, amps=amps, DCs=DCs)
            if DCs.size == 1:
                QS_states = QS_states.reshape((*QS_states.shape, 1))
                Vmeff = Vmeff.reshape((*Vmeff.shape, 1))
            dQdt = -neuron.iNet(Vmeff, QS_states)  # mA/m2
            Afactor = 1e-3
        else:
            Aref = amps
            Afactor = 1.
            dQdt = np.empty((Aref.size, Qref.size, DCs.size))
            for iA, A in enumerate(Aref):
                for iDC, DC in enumerate(DCs):
                    dQdt[iA, :, iDC] = dQdt0 + A * DC

        # For each duty cycle
        for j, DC in enumerate(DCs):
            color = 'C{}'.format(icolor)

            # Plot either all charge SFPs or only equilibrium charge (if Qi provided)
            # for each  acoustic amplitude
            Aplot, Qplot = [], []
            for k, Adrive in enumerate(Aref):
                dQ_profile = dQdt[k, :, j]
                if Qi[i] is None:
                    Qzeros = getFixedPoints(Qref, dQ_profile)
                    if Qzeros is not None:
                        Qzeros = Qzeros.tolist()
                    else:
                        Qzeros = [np.nan]
                else:
                    Qzeros = [getEqPoint1D(Qref, dQ_profile, Qi[i])]
                Qplot += Qzeros
                Aplot += [Adrive] * len(Qzeros)
            ax.plot(np.array(Aplot) * Afactor, np.array(Qplot) * 1e5, '.', c=color,
                    label='{} neuron - {:.0f} % DC'.format(neuron.name, DC * 1e2))
            ax.axhline(neuron.Qm0() * 1e5, c=color, linestyle='--')

            # If specified, compute and plot the threshold excitation amplitude
            if titrate:
                if stim_type == 'US':
                    Athr = nbls.titrate(Fdrive, tstim, TITRATION_T_OFFSET, PRF=PRF, DC=DC,
                                        Arange=(Aref.min(), Aref.max()))  # Pa
                    ax.axvline(Athr * Afactor, c=color, linestyle='--')
                else:
                    Athr_pos = neuron.titrate(tstim, TITRATION_T_OFFSET, PRF=PRF, DC=DC,
                                              Arange=(0., Aref.max()))  # mA/m2
                    ax.axvline(Athr_pos * Afactor, c=color, linestyle='--')
                    Athr_neg = neuron.titrate(tstim, TITRATION_T_OFFSET, PRF=PRF, DC=DC,
                                              Arange=(Aref.min(), 0.))  # mA/m2
                    ax.axvline(Athr_neg * Afactor, c=color, linestyle='-.')

            icolor += 1

    # Post-process figure
    ax.set_ylim(np.array([Qrange[0], 0]) * 1e5)
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()

    fig.canvas.set_window_title('QSS_{}_vs_{}A_{}_{}_{}%DC{}'.format(
        'Qeq' if Qi[0] is not None else 'SFPs',
        xscale,
        '_'.join([n.name for n in neurons]),
        stim_type,
        '_'.join(['{:.0f}'.format(DC * 1e2) for DC in DCs]),
        '_with_thresholds' if titrate else ''
    ))

    return fig
