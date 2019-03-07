# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-07 14:47:20


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


def plotQSS(neuron, a, Fdrive, Adrive, fs=12):

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

    # Compute quasi-steady currents
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
    ax.set_ylabel('Effective potential (mV)', fontsize=fs)
    ax.plot(Qref * 1e5, Vm, color='C0')
    ax.axhline(neuron.Vm0, linewidth=0.5, color='k')

    # Subplot 2: quasi-steady states
    ax = axes[1]
    ax.set_ylabel('Quasi-steady states', fontsize=fs)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylim([-0.05, 1.05])
    for label, qsstate in zip(neuron.states_names, qsstates):
        ax.plot(Qref * 1e5, qsstate, label=label)

    # Subplot 3: currents
    ax = axes[2]
    ax.set_ylabel('QS Currents (mA/m2)', fontsize=fs)
    ax.plot(Qref * 1e5, iNet, '-', color='C0', label='$I_{Net}$')
    ax.axhline(0, color='k', linewidth=0.5)

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    for ax in axes:
        ax.legend(loc='center right', fontsize=fs, frameon=False, bbox_to_anchor=(1.3, 0.5))

    return fig


neuron = getNeuronsDict()['STN']()
a = 32e-9  # m
Fdrive = 500e3  # Hz
amps = np.array([10, 20, 60]) * 1e3  # Pa

for Adrive in amps:
    plotQSS(neuron, a, Fdrive, Adrive)

plt.show()
