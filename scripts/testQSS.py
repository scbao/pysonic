# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-11 13:50:10


import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib

from PySONIC.utils import getLookups2D, Intensity2Pressure, getLowIntensitiesSTN
from PySONIC.neurons import getNeuronsDict
from PySONIC.core import NeuronalBilayerSonophore


# Plot parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'


def getQSSvars(neuron, a, Fdrive, Adrive):

    # Get lookups for specific (a, f, A) combination
    Aref, Qref, lookups2D, _ = getLookups2D(neuron.name, a=a, Fdrive=Fdrive)
    lookups1D = {key: interp1d(Aref, y2D, axis=0)(Adrive) for key, y2D in lookups2D.items()}

    # Remove unnecessary items ot get ON rates and effective potential
    rates = lookups1D
    rates.pop('ng')
    Vm = rates.pop('V')

    # Compute quasi-steady states for each charge value
    qsstates = np.empty((len(neuron.states_names), Qref.size))
    for j, x in enumerate(neuron.states_names):
        # If channel state, compute steady-state values from rate constants
        if x in neuron.getGates():
            x = x.lower()
            alpha_str, beta_str = ['{}{}'.format(s, x) for s in ['alpha', 'beta']]
            alphax = rates[alpha_str]
            betax = rates[beta_str]
            qsstates[j, :] = alphax / (alphax + betax)
        # Otherwise assume the state has reached a steady-state value at the specific charge value
        else:
            qsstates[j, :] = np.array([neuron.steadyStates(Q / neuron.Cm0 * 1e3)[j] for Q in Qref])

    return Qref, Vm, qsstates


def plotQSSdetails(neuron, a, Fdrive, Adrive, fs=12):

    # Get quasi-steady states and effective membrane potential profiles
    Qref, Vm, qsstates = getQSSvars(neuron, a, Fdrive, Adrive)

    # Compute QSS currents
    currents = neuron.currents(Vm, qsstates)
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
    ax.plot(Qref * 1e5, Vm, color='C0')
    ax.axhline(neuron.Vm0, linewidth=0.5, color='k')

    # Subplot 2: quasi-steady states
    ax = axes[1]
    ax.set_ylabel('$X_\infty$', fontsize=fs)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([-0.05, 1.05])
    for label, qsstate in zip(neuron.states_names[:-1], qsstates[:-1]):
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


def plotInetQSSvsAmp(neuron, a, Fdrive, amps, fs=12, cmap='viridis', zscale='lin'):

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
        Qref, Vm, qsstates = getQSSvars(neuron, a, Fdrive, Adrive)
        iNet = neuron.iNet(Vm, qsstates)
        ax.plot(Qref * 1e5, iNet * 1e-3, label=lbl, c=c)

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


def getLastQm(neuron, a, Fdrive, amps, tstim=150e-3):
    nbls = NeuronalBilayerSonophore(a, neuron, Fdrive)
    Qlast = np.empty(amps.size)
    for i, Adrive in enumerate(amps):
        Qm = nbls.runSONIC(Fdrive, Adrive, tstim, 0, 1e2, 1.)[2]
        Qlast[i] = Qm[-1]
    return Qlast


def getEqChargesQSS(neuron, a, Fdrive, amps):
    Qthr_QSS = np.empty(amps.size)
    for i, Adrive in enumerate(amps):
        Qref, Vm, qsstates = getQSSvars(neuron, a, Fdrive, Adrive)
        iNet = neuron.iNet(Vm, qsstates)
        Qthr_QSS[i] = np.interp(0, iNet, Qref, left=0., right=np.nan)
    return Qthr_QSS


def compareQSSvsSim(neuron, a, Fdrive, amps, fs=12):

    # Plot Qm balancing net current as function of amplitude
    fig, ax = plt.subplots(figsize=(6, 4))
    figname = '{} neuron - balance charge vs. amplitude'.format(neuron.name)
    ax.set_title(figname)
    ax.set_xlabel('Amplitude (kPa)', fontsize=fs)
    ax.set_ylabel('$\\rm Q_{thr}\ (nC/cm^2)$', fontsize=fs)
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    ax.plot(amps * 1e-3, getEqChargesQSS(neuron, a, Fdrive, amps) * 1e5, label='QSS')
    # ax.plot(amps * 1e-3, getLastQm(neuron, a, Fdrive, amps) * 1e5, label='sim')
    ax.legend(frameon=False, fontsize=fs)
    fig.tight_layout()

    fig.canvas.set_window_title(
        '{}_Qthr_QSS_vs_sim'.format(neuron.name))


neuron = getNeuronsDict()['STN']()
a = 32e-9  # m
Fdrive = 500e3  # Hz
Amin = 10e3  # Pa
Amax = 60e3  # Pa

# for Adrive in np.linspace(Amin, Amax, 5):
#     plotQSSdetails(neuron, a, Fdrive, Adrive)

plotInetQSSvsAmp(neuron, a, Fdrive, np.linspace(Amin, Amax, 20))

compareQSSvsSim(neuron, a, Fdrive, np.linspace(Amin, Amax, 20))

plt.show()
