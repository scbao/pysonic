# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-08 14:33:42


import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib

from PySONIC.utils import getLookups2D
from PySONIC.neurons import getNeuronsDict


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
    iNet = neuron.iNet(Vm, qsstates)

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(6, 7))
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
    for label, qsstate in zip(neuron.states_names, qsstates):
        ax.plot(Qref * 1e5, qsstate, label=label)

    # Subplot 3: currents
    ax = axes[2]
    ax.set_ylabel('QSS currents (A/m2)', fontsize=fs)
    ax.plot(Qref * 1e5, iNet * 1e-3, '-', color='k', label='$I_{Net}$')
    ax.axhline(0, color='k', linewidth=0.5)

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    for ax in axes[1:]:
        ax.legend(loc='center right', fontsize=fs, frameon=False, bbox_to_anchor=(1.3, 0.5))

    return fig


def plotIQSSvsAmp(neuron, a, Fdrive, amps, fs=12, cmap='viridis', zscale='lin'):

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
    ax.set_xlabel('$Q_m$ (nC/cm2)', fontsize=fs)
    ax.set_ylabel('$I_{net, QSS}$ (A/m2)', fontsize=fs)
    for skey in ['top', 'right']:
        ax.spines[skey].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    for item in ax.get_xticklabels(minor=True):
        item.set_visible(False)
    figname = '{} neuron - QSS current imbalance vs. amplitude'.format(neuron.name)
    ax.set_title(figname, fontsize=fs)
    ax.axhline(0, color='k', linewidth=0.5)

    for Adrive in amps:
        lbl = '{:.2f} kPa'.format(Adrive * 1e-3)
        c = sm.to_rgba(Adrive * 1e-3)
        Qref, Vm, qsstates = getQSSvars(neuron, a, Fdrive, Adrive)
        ax.plot(Qref * 1e5, neuron.iNet(Vm, qsstates) * 1e-3, label=lbl, c=c)

    # ax.legend(loc='center right', fontsize=fs, frameon=False, bbox_to_anchor=(1.3, 0.5))
    fig.tight_layout()

    # Plot colorbar
    fig.subplots_adjust(bottom=0.1, top=0.9, right=0.80, hspace=0.5)
    cbarax = fig.add_axes([0.85, 0.1, 0.03, 0.80])
    fig.colorbar(sm, cax=cbarax)
    cbarax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    for item in cbarax.get_yticklabels():
        item.set_fontsize(fs)

    return fig


neuron = getNeuronsDict()['STN']()
a = 32e-9  # m
Fdrive = 500e3  # Hz
amps = np.linspace(10, 60, 20) * 1e3  # Pa

# for Adrive in amps:
#     plotQSSdetails(neuron, a, Fdrive, Adrive)

plotIQSSvsAmp(neuron, a, Fdrive, amps)

plt.show()
