#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-15 15:59:37
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-21 21:11:54

''' Plot the profiles of the 9 charge-dependent "effective" HH coefficients,
    as a function of charge density or membrane potential. '''

import os
import re
import ntpath
import inspect
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import PointNICE
from PointNICE.utils import OpenFilesDialog, rescale
from PointNICE.pltvars import pltvars


# Dictionary of neurons
neurons = {}
for classname, obj in inspect.getmembers(PointNICE.channels):
    if inspect.isclass(obj) and isinstance(obj.name, str):
        neurons[obj.name] = obj

# Select data files (PKL)
lookup_root = '../lookups/'
lookup_absroot = os.path.abspath(lookup_root)
lookup_filepaths, _ = OpenFilesDialog('pkl', lookup_absroot)

# Quit if no file selected
if not lookup_filepaths:
    print('error: no lookup table selected')
    quit()

print('importing lookup tables')

nfiles = len(lookup_filepaths)
rgxp = re.compile('([A-Za-z]*)_lookups_a(\d*.\d*)nm_f(\d*.\d*)kHz.pkl')
xvar = 'V'  # 'Q' (abscissa variable)

nvars = 9
fs = 15

for i in range(nfiles):

    # Load lookup table
    lookup_filename = ntpath.basename(lookup_filepaths[i])
    mo = rgxp.fullmatch(lookup_filename)
    if not mo:
        print('Error: lookup file does not match regular expression pattern')
    else:
        # Retrieve stimulus parameters
        neuron_name = mo.group(1)
        neuron = neurons[neuron_name]()
        varlist = neuron.coeff_names
        print(varlist)
        Fdrive = float(mo.group(3)) * 1e3

        # Retrieve coefficients data
        with open(lookup_filepaths[i], 'rb') as fh:
            lookup = pickle.load(fh)
            Qm = lookup['Q']
            amps = lookup['A']
            Veff = lookup['V']
            Amin = np.amin(amps)
            Amax = np.amax(amps)
            Qmin = np.amin(Qm)
            Qmax = np.amax(Qm)
            namps = amps.size

        # Plotting
        print('plotting')

        mymap = cm.get_cmap('jet')
        sm_amp = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Amin * 1e-3, Amax * 1e-3))
        sm_amp._A = []

        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 8))

        ax = axes[0, 0]
        ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=fs)
        ax.set_ylabel('$V_m\ (mV)$', fontsize=fs)
        for i in range(namps):
            ax.plot(Qm * 1e5, Veff[i, :] * 1e0, c=mymap(rescale(amps[i], Amin, Amax)))

        for j in range(nvars - 1):
            pltvar = pltvars[varlist[j]]
            ax = axes[int((j + 1) / 3), (j + 1) % 3]
            ax.set_ylabel('${}\ ({})$'.format(pltvar['label'], pltvar['unit']), fontsize=fs)
            if xvar == 'Q':
                ax.set_xlabel('$Q_m \ (nC/cm^2)$', fontsize=fs)
                for i in range(namps):
                    ax.plot(Qm * 1e5, lookup[varlist[j]][i, :] * pltvar['factor'],
                            c=mymap(rescale(amps[i], Amin, Amax)))
            elif xvar == 'V':
                ax.set_xlabel('$V_m \ (mV)$', fontsize=fs)
                for i in range(namps):
                    ax.plot(Veff[i, :] * 1e0, lookup[varlist[j]][i, :] * pltvar['factor'],
                            c=mymap(rescale(amps[i], Amin, Amax)))
        plt.tight_layout()

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.1, 0.02, 0.8])
        fig.add_axes()
        fig.colorbar(sm_amp, cax=cbar_ax)
        cbar_ax.set_ylabel('$A_{drive} \ (kPa)$', fontsize=fs)

plt.show()
