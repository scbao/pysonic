#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-03-20 12:19:55
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-21 21:12:31

""" Batch plot profiles of several specific output variables of NICE simulations. """

import pickle
import ntpath
import re
import inspect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import PointNICE
from PointNICE.utils import SaveFigDialog, OpenFilesDialog, getPatchesLoc
from PointNICE.pltvars import pltvars


# List of variables to plot and positions
tag = 'test'
varlist = ['Vm', 'VL']
positions = [0, 0]

nvars = len(varlist)
naxes = np.unique(positions).size


# Plotting options
t_unit = 'ms'  # us
t_factor = 1e3
t_onset = 3e-3
fs = 15
show_patches = 1
plt_show = 1
plt_save = 0
fig_ext = 'png'
ask_before_save = 1


# Dictionary of neurons
neurons = {}
for classname, obj in inspect.getmembers(PointNICE.channels):
    if inspect.isclass(obj) and isinstance(obj.name, str):
        neurons[obj.name] = obj

# Regular expression for input files
rgxp = re.compile('sim_([A-Za-z]*)_(.*).pkl')

# Select data files
pkl_filepaths, pkl_dir = OpenFilesDialog('pkl')
if not pkl_filepaths:
    print('error: no input file')
    quit()

# Loop through data files
for pkl_filepath in pkl_filepaths:

    # Get code from file name
    pkl_filename = ntpath.basename(pkl_filepath)
    filecode = pkl_filename[0:-4]

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

    # Extract variables
    print('Extracting variables')
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

    # neuron = neurons[neuron_name]
    Qm0 = neuron.Cm0 * neuron.Vm0 * 1e-3
    bls = PointNICE.BilayerSonophore(geom, params, Fdrive, neuron.Cm0, Qm0)

    # Get data of variables to plot
    vrs = []
    for i in range(nvars):
        pltvar = pltvars[varlist[i]]
        if 'alias' in pltvar:
            var = eval(pltvar['alias'])
        elif 'key' in pltvar:
            var = data[pltvar['key']]
        elif 'constant' in pltvar:
            var = eval(pltvar['constant']) * np.ones(t.size)
        else:
            var = data[varlist[i]]
        vrs.append(var)

    # Determine patches location
    npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)

    # Adding onset to all signals
    if t_onset > 0.0:
        t = np.insert(t + t_onset, 0, 0.0)
        for i in range(nvars):
            vrs[i] = np.insert(vrs[i], 0, vrs[i][0])
        tpatch_on += t_onset
        tpatch_off += t_onset

    # Plotting
    if naxes == 1:
        _, ax = plt.subplots(figsize=(11, 4))
        axes = [ax]
    else:
        _, axes = plt.subplots(naxes, 1, figsize=(11, min(3 * naxes, 9)))

    # Axes
    for i in range(naxes):
        ax = axes[i]
        if positions[i] < naxes - 1:
            ax.get_xaxis().set_ticklabels([])
        else:
            ax.set_xlabel('$time \ (' + t_unit + ')$', fontsize=fs)
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fs)
        ax.locator_params(axis='y', nbins=2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fs)

    # Time series
    icolor = 0
    for i in range(nvars):
        pltvar = pltvars[varlist[i]]
        ax = axes[positions[i]]
        if 'constant' in pltvar:
            ax.plot(t * t_factor, vrs[i] * pltvar['factor'], '--', c='black', lw=4)
        else:
            ax.plot(t * t_factor, vrs[i] * pltvar['factor'], c='C{}'.format(icolor), lw=4)
            if 'min' in pltvar and 'max' in pltvar:
                ax.set_ylim(pltvar['min'], pltvar['max'])
            if pltvar['unit']:
                ax.set_ylabel('${}\ ({})$'.format(pltvar['label'], pltvar['unit']),
                              fontsize=fs)
            else:
                ax.set_ylabel('${}$'.format(pltvar['label']), fontsize=fs)
            icolor += 1

    # Patches
    if show_patches == 1:
        for ax in axes:
            (ybottom, ytop) = ax.get_ylim()
            for j in range(npatches):
                ax.add_patch(patches.Rectangle((tpatch_on[j] * t_factor, ybottom),
                                               (tpatch_off[j] - tpatch_on[j]) * t_factor,
                                               ytop - ybottom, color='#8A8A8A', alpha=0.1))

    plt.tight_layout()

    # Save figure if needed (automatic or checked)
    if plt_save == 1:
        if ask_before_save == 1:
            plt_filename = SaveFigDialog(pkl_dir, '{}_{}.{}'.format(filecode, tag, fig_ext))
        else:
            plt_filename = '{}/{}_{}.{}'.format(pkl_dir, filecode, tag, fig_ext)
        if plt_filename:
            plt.savefig(plt_filename)
            print('Saving figure as "{}"'.format(plt_filename))
            plt.close()

# Show all plots if needed
if plt_show == 1:
    plt.show()
