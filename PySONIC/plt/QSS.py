
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.colors import ListedColormap
from scipy.interpolate import interp1d, interp2d

from ..postpro import getFixedPoints
from ..core import NeuronalBilayerSonophore
from .pltutils import *
from ..constants import TITRATION_T_OFFSET
from ..utils import logger


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
    _, Qref, lookups, QSS = nbls.quasiSteadyStates(Fdrive, amps=Adrive, charges=charges)
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
    _, Qref, lookups, QSS = nbls.quasiSteadyStates(Fdrive, amps=Adrive)
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
    Q_SFPs = []
    Q_UFPs = []

    log = 'plotting {} neuron QSS {} vs. amp for {} stimulation @ {:.0f}% DC'.format(
        neuron.name, varname, stim_type, DC * 1e2)
    logger.info(log)

    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)

    # Get reference dictionaries for zero amplitude
    _, Qref, lookups0, QSS0 = nbls.quasiSteadyStates(Fdrive, amps=0.)
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
    if varname == 'dQdt':
        Q_SFPs += getFixedPoints(Qref, var0, filter='stable').tolist()
        Q_UFPs += getFixedPoints(Qref, var0, filter='unstable').tolist()

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
        _, Qref, lookups, QSS = nbls.quasiSteadyStates(Fdrive, amps=amps, DCs=DC)
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
        if varname == 'dQdt':
            # mark eq. point if starting point provided, otherwise mark all FPs
            Q_SFPs += getFixedPoints(Qref, var, filter='stable').tolist()
            Q_UFPs += getFixedPoints(Qref, var, filter='unstable').tolist()

    # Plot fixed-points, if any
    if len(Q_SFPs) > 0:
        ax.plot(np.array(Q_SFPs) * Qvar['factor'], np.zeros(len(Q_SFPs)), 'o', c='k',
                markersize=5, zorder=2)
    if len(Q_UFPs) > 0:
        ax.plot(np.array(Q_UFPs) * Qvar['factor'], np.zeros(len(Q_UFPs)), 'x', c='k',
                markersize=5, zorder=2)

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


def plotEqChargeVsAmp(neurons, a, Fdrive, amps=None, tstim=250e-3, PRF=100.0,
                      DCs=[1.], fs=12, xscale='lin', titrate=False):
    ''' Plot the equilibrium membrane charge density as a function of acoustic amplitude,
        given an initial value of membrane charge density.

        :param neurons: neuron objects
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param amps: US amplitudes (Pa)
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
    figname = 'charge stability vs. amplitude'
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

    Qrange = (np.inf, -np.inf)

    icolor = 0
    for i, neuron in enumerate(neurons):

        nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)

        # Compute reference charge variation array for zero amplitude
        _, Qref, lookups0, QSS0 = nbls.quasiSteadyStates(Fdrive, amps=0.)
        Qrange = (min(Qrange[0], Qref.min()), max(Qrange[1], Qref.max()))
        Vmeff0 = lookups0['V']
        if stim_type == 'elec':  # if E-STIM case, compute steady states with constant capacitance
            Vmeff0 = Qref / neuron.Cm0 * 1e3
            QSS0 = neuron.steadyStates(Vmeff0)
        dQdt0 = -neuron.iNet(Vmeff0, np.array([QSS0[k] for k in neuron.states]))  # mA/m2

        # Compute 3D QSS charge variation array
        if stim_type == 'US':
            _, _, lookups, QSS = nbls.quasiSteadyStates(Fdrive, amps=amps, DCs=DCs)
            if DCs.size == 1:
                lookups = {k: v.reshape(*v.shape, 1) for k, v in lookups.items()}
                QSS = {k: v.reshape(*v.shape, 1) for k, v in QSS.items()}
            dQdt = -neuron.iNet(lookups['V'], np.array([QSS[k] for k in neuron.states]))  # mA/m2
            Afactor = 1e-3
        else:
            Afactor = 1.
            dQdt = np.empty((amps.size, Qref.size, DCs.size))
            for iA, A in enumerate(amps):
                for iDC, DC in enumerate(DCs):
                    dQdt[iA, :, iDC] = dQdt0 + A * DC

        # For each duty cycle
        for iDC, DC in enumerate(DCs):
            color = 'k' if len(neurons) * len(DCs) == 1 else 'C{}'.format(icolor)

            # Plot charge SFPs and UFPs for each  acoustic amplitude
            A_SFPs, A_UFPs, Q_SFPs, Q_UFPs = [], [], [], []
            for iA, Adrive in enumerate(amps):

                # print('- A = {:.2f} kPa'.format(Adrive * 1e-3))

                dQ_profile = dQdt[iA, :, iDC]
                sfp = getFixedPoints(Qref, dQ_profile, filter='stable').tolist()
                ufp = getFixedPoints(Qref, dQ_profile, filter='unstable').tolist()
                Q_SFPs += sfp
                A_SFPs += [Adrive] * len(sfp)
                Q_UFPs += ufp
                A_UFPs += [Adrive] * len(ufp)

                QSS1D = {k: v[iA, :, iDC] for k, v in QSS.items()}
                lookups1D = {k: v[iA, :, iDC] for k, v in lookups.items()}
                lookups1D['Q'] = Qref

                # Analyze real stability of selected SFPs
                if len(sfp) > 0:
                    for ipoint, Qpoint in enumerate(sfp):
                        # Interpolate QSS at the Q-SFPs and compute effective derivatives
                        print('-- Q-SFP at {:.2f} nC/cm2'.format(Qpoint * 1e5))
                        QSS_sfp = {k: np.interp(Qpoint, Qref, v, left=np.nan, right=np.nan)
                                   for k, v in QSS1D.items()}
                        dQSS_sfp = neuron.derEffStates(Qpoint, QSS_sfp.values(), lookups1D)
                        abs_ratios = {x: np.abs(dQSS_sfp[x] / QSS_sfp[x]) for x in QSS_sfp.keys()}
                        for x in neuron.states:
                            print('--- {} = {:.5f}, d{}/dt = {:.5f}, |ratio| = {:.2e}'.format(
                                x, QSS_sfp[x], x, dQSS_sfp[x], abs_ratios[x]))

            ax.plot(np.array(A_SFPs) * Afactor, np.array(Q_SFPs) * 1e5, 'o', c=color, markersize=3,
                    label='{} neuron - SFPs @ {:.0f} % DC'.format(neuron.name, DC * 1e2))
            ax.plot(np.array(A_UFPs) * Afactor, np.array(Q_UFPs) * 1e5, 'x', c=color, markersize=3,
                    label='{} neuron - UFPs @ {:.0f} % DC'.format(neuron.name, DC * 1e2))

            # If specified, compute and plot the threshold excitation amplitude
            if titrate:
                if stim_type == 'US':
                    Athr = nbls.titrate(Fdrive, tstim, TITRATION_T_OFFSET, PRF=PRF, DC=DC,
                                        Arange=(amps.min(), amps.max()))  # Pa
                    ax.axvline(Athr * Afactor, c=color, linestyle='--')
                else:
                    Athr_pos = neuron.titrate(tstim, TITRATION_T_OFFSET, PRF=PRF, DC=DC,
                                              Arange=(0., amps.max()))  # mA/m2
                    ax.axvline(Athr_pos * Afactor, c=color, linestyle='--')
                    Athr_neg = neuron.titrate(tstim, TITRATION_T_OFFSET, PRF=PRF, DC=DC,
                                              Arange=(amps.min(), 0.))  # mA/m2
                    ax.axvline(Athr_neg * Afactor, c=color, linestyle='-.')

            icolor += 1

        if len(neurons) * len(DCs) == 1:
            dQdt_sign = np.sign(np.squeeze(dQdt))
            cmap = ListedColormap(plt.get_cmap('Pastel2').colors[:2])
            # x = computeMeshEdges(amps, scale=xscale) * Afactor
            # y = computeMeshEdges(Qref) * 1e5
            # xx, yy = np.meshgrid(x, y)
            # print(xx.shape, yy.shape)
            # ax.pcolormesh(xx.T, yy.T, dQdt_sign, cmap=cmap)
            ax.contourf(amps * Afactor, Qref * 1e5, dQdt_sign.T, cmap=cmap)

    # Post-process figure
    ax.set_ylim(np.array([Qrange[0], 0]) * 1e5)
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()

    fig.canvas.set_window_title('QSS_Qstab_vs_{}A_{}_{}_{}%DC{}'.format(
        xscale,
        '_'.join([n.name for n in neurons]),
        stim_type,
        '_'.join(['{:.0f}'.format(DC * 1e2) for DC in DCs]),
        '_with_thresholds' if titrate else ''
    ))

    return fig
