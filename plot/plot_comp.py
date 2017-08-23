#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-02-13 12:41:26
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-23 14:18:57

""" Compare profiles of several specific output variables of NICE simulations. """

import pickle
import ntpath
import re
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import PointNICE
from PointNICE.utils import OpenFilesDialog, InteractiveLegend, getPatchesLoc
from PointNICE.pltvars import pltvars


# List of variables to plot
# varlist = ['Qm']
yvars = ['Pac', 'Pmavg', 'Telastic', 'Vm', 'iL']

# Plotting options
fs = 12
show_patches = True

t_plt = pltvars['t']
y_pltvars = {key: pltvars[key] for key in yvars}

# Dictionary of neurons
neurons = {}
for classname, obj in inspect.getmembers(PointNICE.channels):
    if inspect.isclass(obj) and isinstance(obj.name, str):
        neurons[obj.name] = obj


# Regular expression for input files
rgxp = re.compile('sim_([A-Za-z]*)_(.*).pkl')

# Select data files
pkl_filepaths, _ = OpenFilesDialog('pkl')
if not pkl_filepaths:
    print('error: no input file')
    quit()

# Initialize figure and axes
# nvars = len(varlist)
nvars = len(yvars)
if nvars == 1:
    _, ax = plt.subplots(figsize=(11, 4))
    axes = [ax]
else:
    _, axes = plt.subplots(nvars, 1, figsize=(11, min(3 * nvars, 9)))
labels = [ntpath.basename(fp)[4:-4].replace('_', ' ') for fp in pkl_filepaths]
for i in range(nvars):
    ax = axes[i]
    # pltvar = pltvars[varlist[i]]
    pltvar = y_pltvars[i]
    if 'min' in pltvar and 'max' in pltvar:
        ax.set_ylim(pltvar['min'], pltvar['max'])
    if pltvar['unit']:
        ax.set_ylabel('${}\ ({})$'.format(pltvar['label'], pltvar['unit']), fontsize=fs)
    else:
        ax.set_ylabel('${}$'.format(pltvar['label']), fontsize=fs)
    if i < nvars - 1:
        ax.get_xaxis().set_ticklabels([])
    else:
        ax.set_xlabel('${}\ ({})$'.format(t_plt['label'], t_plt['unit']), fontsize=fs)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fs)
    ax.locator_params(axis='y', nbins=2)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)



# Loop through data files
tstim_ref = 0.0
nstim = 0
j = 0
aliases = {}
for pkl_filepath in pkl_filepaths:

    pkl_filename = ntpath.basename(pkl_filepath)

    # Retrieve neuron name
    mo = rgxp.fullmatch(pkl_filename)
    if not mo:
        print('Error: PKL file does not match regular expression pattern')
        quit()
    neuron_name = mo.group(1)

    # Load data
    print('Loading data from "' + pkl_filename + '"')
    with open(pkl_filepath, 'rb') as pkl_file:
        data = pickle.load(pkl_file)

    # Extract useful variables
    t = data['t']
    states = data['states']
    tstim = data['tstim']
    Fdrive = data['Fdrive']
    Adrive = data['Adrive']
    params = data['params']
    a = data['a']
    d = data['d']
    geom = {"a": a, "d": d}

    # Initialize BLS and channels mechanism
    neuron = neurons[neuron_name]()
    Qm0 = neuron.Cm0 * neuron.Vm0 * 1e-3
    bls = PointNICE.BilayerSonophore(geom, params, Fdrive, neuron.Cm0, Qm0)

    # Get data of variables to plot
    vrs = []
    for i in range(nvars):
        pltvar = y_pltvars[i]
        # pltvar = pltvars[varlist[i]]
        if 'alias' in pltvar:
            var = eval(pltvar['alias'])
        elif 'key' in pltvar:
            var = data[pltvar['key']]
        else:
            var = data[varlist[i]]
        vrs.append(var)

    # Determine patches location
    npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)

    # Adding onset to all signals
    if t_plt['onset'] > 0.0:
        t = np.insert(t + t_plt['onset'], 0, 0.0)
        for i in range(nvars):
            vrs[i] = np.insert(vrs[i], 0, vrs[i][0])
        tpatch_on += t_plt['onset']
        tpatch_off += t_plt['onset']

    # Plotting
    handles = [axes[i].plot(t * t_plt['factor'], vrs[i] * pltvars[varlist[i]]['factor'],
                            linewidth=2, label=labels[j]) for i in range(nvars)]
    plt.tight_layout()

    if show_patches:
        k = 0
        # stimulation patches
        for ax in axes:
            handle = handles[k]
            (ybottom, ytop) = ax.get_ylim()
            la = []
            for i in range(npatches):
                la.append(ax.add_patch(patches.Rectangle((tpatch_on[i] * t_plt['factor'], ybottom),
                                                         (tpatch_off[i] - tpatch_on[i]) * t_plt['factor'],
                                                         ytop - ybottom,
                                                         color=handle[0].get_color(), alpha=0.2)))

        aliases[handle[0]] = la
        k += 1

    if tstim != tstim_ref:
        if nstim == 0:
            nstim += 1
            tstim_ref = tstim
        else:
            print('Warning: comparing different stimulation durations')

    j += 1


iLegends = []
for k in range(nvars):
    axes[k].legend(loc='upper left', fontsize=fs)
    iLegends.append(InteractiveLegend(axes[k].legend_, aliases))

plt.show()
