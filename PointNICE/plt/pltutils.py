# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-23 14:55:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-05-25 22:09:19

''' Plotting utilities '''

import sys
import os
import pickle
import ntpath
import re
import logging
import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.interpolate import interp2d
# from scipy.optimize import brentq
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

from .. import neurons
from ..utils import getNeuronsDict, getLookupDir, rescale, InputError, computeMeshEdges, si_format, itrpLookupsFreq
from ..bls import BilayerSonophore
from .pltvars import pltvars

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Get package logger
logger = logging.getLogger('PointNICE')

# Define global variables
neuron = None
bls = None
timeunits = {'ASTIM': 't_ms', 'ESTIM': 't_ms', 'MECH': 't_us'}

# Regular expression for input files
rgxp = re.compile('(ESTIM|ASTIM)_([A-Za-z]*)_(.*).pkl')
rgxp_mech = re.compile('(MECH)_(.*).pkl')


# nb = '[0-9]*[.]?[0-9]+'
# rgxp_ASTIM = re.compile('(ASTIM)_(\w+)_(PW|CW)_({0})nm_({0})kHz_({0})kPa_({0})ms(.*)_(\w+).pkl'.format(nb))
# rgxp_ESTIM = re.compile('(ESTIM)_(\w+)_(PW|CW)_({0})mA_per_m2_({0})ms(.*).pkl'.format(nb))
# rgxp_PW = re.compile('_PRF({0})kHz_DC({0})_(PW|CW)_(\d+)kHz_(\d+)kPa_(\d+)ms_(.*).pkl'.format(nb))


# Figure naming conventions
ESTIM_CW_title = '{} neuron: CW E-STIM {:.2f}mA/m2, {:.0f}ms'
ESTIM_PW_title = '{} neuron: PW E-STIM {:.2f}mA/m2, {:.0f}ms, {:.2f}Hz PRF, {:.0f}% DC'
ASTIM_CW_title = '{} neuron: CW A-STIM {:.0f}kHz, {:.0f}kPa, {:.0f}ms'
ASTIM_PW_title = '{} neuron: PW A-STIM {:.0f}kHz, {:.0f}kPa, {:.0f}ms, {:.2f}Hz PRF, {:.2f}% DC'
MECH_title = '{:.0f}nm BLS structure: MECH-STIM {:.0f}kHz, {:.0f}kPa'


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


class InteractiveLegend(object):
    """ Class defining an interactive matplotlib legend, where lines visibility can
    be toggled by simply clicking on the corresponding legend label. Other graphic
    objects can also be associated to the toggle of a specific line

    Adapted from:
    http://stackoverflow.com/questions/31410043/hiding-lines-after-showing-a-pyplot-figure
    """

    def __init__(self, legend, aliases):
        self.legend = legend
        self.fig = legend.axes.figure
        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()
        self.handles_aliases = aliases
        self.update()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10)  # 10 points tolerance

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def _build_lookups(self, legend):
        ''' Method of the InteractiveLegend class building
            the legend lookups. '''

        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))

        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist

        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()

    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
                if artist in self.handles_aliases:
                    for al in self.handles_aliases[artist]:
                        al.set_visible(True)
            else:
                handle.set_visible(False)
                if artist in self.handles_aliases:
                    for al in self.handles_aliases[artist]:
                        al.set_visible(False)
        self.fig.canvas.draw()


    def show(self):
        ''' showing the interactive legend '''

        plt.show()


def getPatchesLoc(t, states):
    ''' Determine the location of stimulus patches.

        :param t: simulation time vector (s).
        :param states: a vector of stimulation state (ON/OFF) at each instant in time.
        :return: 3-tuple with number of patches, timing of STIM-ON an STIM-OFF instants.
    '''

    # Compute states derivatives and identify bounds indexes of pulses
    dstates = np.diff(states)
    ipatch_on = np.insert(np.where(dstates > 0.0)[0] + 1, 0, 0)
    ipatch_off = np.where(dstates < 0.0)[0]
    if ipatch_off.size < ipatch_on.size:
        ioff = t.size - 1
        if ipatch_off.size == 0:
            ipatch_off = np.array([ioff])
        else:
            ipatch_off = np.insert(ipatch_off, ipatch_off.size - 1, ioff)

    # Get time instants for pulses ON and OFF
    npatches = ipatch_on.size
    tpatch_on = t[ipatch_on]
    tpatch_off = t[ipatch_off]

    # return 3-tuple with #patches, pulse ON and pulse OFF instants
    return (npatches, tpatch_on, tpatch_off)


def SaveFigDialog(dirname, filename):
    """ Open a FileSaveDialogBox to set the directory and name
        of the figure to be saved.

        The default directory and filename are given, and the
        default extension is ".pdf"

        :param dirname: default directory
        :param filename: default filename
        :return: full path to the chosen filename
    """
    root = tk.Tk()
    root.withdraw()
    filename_out = filedialog.asksaveasfilename(defaultextension=".pdf", initialdir=dirname,
                                                initialfile=filename)
    return filename_out



def plotComp(varname, filepaths, labels=None, fs=15, lw=2, colors=None, lines=None, patches='one',
             xticks=None, yticks=None, blacklegend=False, straightlegend=False, showfig=True,
             inset=None, figsize=(11, 4)):
    ''' Compare profiles of several specific output variables of NICE simulations.

        :param varname: name of variable to extract and compare
        :param filepaths: list of full paths to output data files to be compared
        :param labels: list of labels to use in the legend
        :param fs: labels fontsize
        :param patches: string indicating whether to indicate periods of stimulation with
         colored rectangular patches
    '''

    # Input check 1: variable name
    if varname not in pltvars:
        raise InputError('Unknown plot variable: "{}"'.format(varname))
    pltvar = pltvars[varname]

    # Input check 2: labels
    if labels is not None:
        if len(labels) != len(filepaths):
            raise InputError('Invalid labels ({}): not matching number of compared files ({})'
                             .format(len(labels), len(filepaths)))
        if not all(isinstance(x, str) for x in labels):
            raise InputError('Invalid labels: must be string typed')

    # Input check 3: line styles and colors
    if colors is None:
        colors = ['C{}'.format(j) for j in range(len(filepaths))]
    if lines is None:
        lines = ['-'] * len(filepaths)

    # Input check 4: STIM-ON patches
    greypatch = False
    if patches == 'none':
        patches = [False] * len(filepaths)
    elif patches == 'all':
        patches = [True] * len(filepaths)
    elif patches == 'one':
        patches = [True] + [False] * (len(filepaths) - 1)
        greypatch = True
    elif isinstance(patches, list):
        if len(patches) != len(filepaths):
            raise InputError('Invalid patches ({}): not matching number of compared files ({})'
                             .format(len(patches), len(filepaths)))
        if not all(isinstance(p, bool) for p in patches):
            raise InputError('Invalid patch sequence: all list items must be boolean typed')
    else:
        raise InputError('Invalid patches: must be either "none", all", "one", or a boolean list')

    # Initialize figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_zorder(0)
    for item in ['top', 'right']:
        ax.spines[item].set_visible(False)
    if 'min' in pltvar and 'max' in pltvar:  # optional min and max on y-axis
        ax.set_ylim(pltvar['min'], pltvar['max'])
    if pltvar['unit']:  # y-label with optional unit
        ax.set_ylabel('$\\rm {}\ ({})$'.format(pltvar['label'], pltvar['unit']), fontsize=fs)
    else:
        ax.set_ylabel('$\\rm {}$'.format(pltvar['label']), fontsize=fs)
    if xticks is not None:  # optional x-ticks
        ax.set_xticks(xticks)
    if yticks is not None:  # optional y-ticks
        ax.set_yticks(yticks)
    else:
        ax.locator_params(axis='y', nbins=2)
    if any(ax.get_yticks() < 0):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%+.0f'))
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(fs)

    # Optional inset axis
    if inset is not None:
        inset_ax = fig.add_axes(ax.get_position())
        inset_ax.set_xlim(inset['xlims'][0], inset['xlims'][1])
        inset_ax.set_ylim(inset['ylims'][0], inset['ylims'][1])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        # inset_ax.patch.set_alpha(1.0)
        inset_ax.set_zorder(1)
        inset_ax.add_patch(Rectangle((inset['xlims'][0], inset['ylims'][0]),
                                     inset['xlims'][1] - inset['xlims'][0],
                                     inset['ylims'][1] - inset['ylims'][0],
                                     color='w'))

    # Retrieve neurons dictionary
    neurons_dict = getNeuronsDict()

    # Loop through data files
    aliases = {}
    for j, filepath in enumerate(filepaths):

        # Retrieve sim type
        pkl_filename = ntpath.basename(filepath)
        mo1 = rgxp.fullmatch(pkl_filename)
        mo2 = rgxp_mech.fullmatch(pkl_filename)
        if mo1:
            mo = mo1
        elif mo2:
            mo = mo2
        else:
            logger.error('Error: "%s" file does not match regexp pattern', pkl_filename)
            sys.exit(1)
        sim_type = mo.group(1)
        if sim_type not in ('MECH', 'ASTIM', 'ESTIM'):
            raise InputError('Invalid simulation type: {}'.format(sim_type))

        if j == 0:
            sim_type_ref = sim_type
            t_plt = pltvars[timeunits[sim_type]]
        elif sim_type != sim_type_ref:
            raise InputError('Invalid comparison: different simulation types')

        # Load data
        logger.info('Loading data from "%s"', pkl_filename)
        with open(filepath, 'rb') as fh:
            frame = pickle.load(fh)
            df = frame['data']
            meta = frame['meta']

        # Extract variables
        t = df['t'].values
        states = df['states'].values
        nsamples = t.size

        # Initialize neuron object if ESTIM or ASTIM sim type
        if sim_type in ['ASTIM', 'ESTIM']:
            neuron_name = mo.group(2)
            global neuron
            neuron = neurons_dict[neuron_name]()
            Cm0 = neuron.Cm0
            Qm0 = Cm0 * neuron.Vm0 * 1e-3

            # Extract neuron states if needed
            if 'alias' in pltvar and 'neuron_states' in pltvar['alias']:
                neuron_states = [df[sn].values for sn in neuron.states_names]
        else:
            Cm0 = meta['Cm0']
            Qm0 = meta['Qm0']

        # Initialize BLS if needed
        if sim_type in ['MECH', 'ASTIM'] and 'alias' in pltvar and 'bls' in pltvar['alias']:
            global bls
            bls = BilayerSonophore(meta['a'], meta['Fdrive'], Cm0, Qm0)

        # Determine patches location
        npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)

        # Add onset to time and states vectors
        if t_plt['onset'] > 0.0:
            t = np.insert(t, 0, -t_plt['onset'])
            states = np.insert(states, 0, 0)

        # Set x-axis label
        ax.set_xlabel('$\\rm {}\ ({})$'.format(t_plt['label'], t_plt['unit']), fontsize=fs)

        # Extract variable to plot
        if 'alias' in pltvar:
            var = eval(pltvar['alias'])
        elif 'key' in pltvar:
            var = df[pltvar['key']].values
        elif 'constant' in pltvar:
            var = eval(pltvar['constant']) * np.ones(nsamples)
        else:
            var = df[varname].values
        if var.size == t.size - 1:
            var = np.insert(var, 0, var[0])

        # Determine legend label
        if labels is not None:
            label = labels[j]
        else:
            if sim_type == 'ESTIM':
                if meta['DC'] == 1.0:
                    label = ESTIM_CW_title.format(neuron_name, meta['Astim'], meta['tstim'] * 1e3)
                else:
                    label = ESTIM_PW_title.format(neuron_name, meta['Astim'], meta['tstim'] * 1e3,
                                                  meta['PRF'], meta['DC'] * 1e2)
            elif sim_type == 'ASTIM':
                if meta['DC'] == 1.0:
                    label = ASTIM_CW_title.format(neuron_name, meta['Fdrive'] * 1e-3,
                                                  meta['Adrive'] * 1e-3, meta['tstim'] * 1e3)
                else:
                    label = ASTIM_PW_title.format(neuron_name, meta['Fdrive'] * 1e-3,
                                                  meta['Adrive'] * 1e-3, meta['tstim'] * 1e3,
                                                  meta['PRF'], meta['DC'] * 1e2)
            elif sim_type == 'MECH':
                label = MECH_title.format(meta['a'] * 1e9, meta['Fdrive'] * 1e-3,
                                          meta['Adrive'] * 1e-3)

        # Plot trace
        handle = ax.plot(t * t_plt['factor'], var * pltvar['factor'],
                         linewidth=lw, linestyle=lines[j], color=colors[j], label=label)

        if inset is not None:
            inset_window = np.logical_and(t > (inset['xlims'][0] / t_plt['factor']),
                                          t < (inset['xlims'][1] / t_plt['factor']))
            inset_ax.plot(t[inset_window] * t_plt['factor'], var[inset_window] * pltvar['factor'],
                          linewidth=lw, linestyle=lines[j], color=colors[j])

        # Add optional STIM-ON patches
        if patches[j]:
            (ybottom, ytop) = ax.get_ylim()
            la = []
            color = '#8A8A8A' if greypatch else handle[0].get_color()
            for i in range(npatches):
                la.append(ax.axvspan(tpatch_on[i] * t_plt['factor'], tpatch_off[i] * t_plt['factor'],
                                     edgecolor='none', facecolor=color, alpha=0.2))
            aliases[handle[0]] = la

            if inset is not None:
                cond_on = np.logical_and(tpatch_on > (inset['xlims'][0] / t_plt['factor']),
                                         tpatch_on < (inset['xlims'][1] / t_plt['factor']))
                cond_off = np.logical_and(tpatch_off > (inset['xlims'][0] / t_plt['factor']),
                                          tpatch_off < (inset['xlims'][1] / t_plt['factor']))
                cond_glob = np.logical_and(tpatch_on < (inset['xlims'][0] / t_plt['factor']),
                                           tpatch_off > (inset['xlims'][1] / t_plt['factor']))
                cond_onoff = np.logical_or(cond_on, cond_off)
                cond = np.logical_or(cond_onoff, cond_glob)
                npatches_inset = np.sum(cond)
                for i in range(npatches_inset):
                    inset_ax.add_patch(Rectangle((tpatch_on[cond][i] * t_plt['factor'], ybottom),
                                                 (tpatch_off[cond][i] - tpatch_on[cond][i]) *
                                                 t_plt['factor'], ytop - ybottom, color=color,
                                                 alpha=0.1))

    fig.tight_layout()

    # Optional operations on inset:
    if inset is not None:

        # Re-position inset axis
        axpos = ax.get_position()
        left, right, = rescale(inset['xcoords'], ax.get_xlim()[0], ax.get_xlim()[1],
                               axpos.x0, axpos.x0 + axpos.width)
        bottom, top, = rescale(inset['ycoords'], ax.get_ylim()[0], ax.get_ylim()[1],
                               axpos.y0, axpos.y0 + axpos.height)
        inset_ax.set_position([left, bottom, right - left, top - bottom])
        for i in inset_ax.spines.values():
            i.set_linewidth(2)

        # Materialize inset target region with contour frame
        ax.plot(inset['xlims'], [inset['ylims'][0]] * 2, linestyle='-', color='k')
        ax.plot(inset['xlims'], [inset['ylims'][1]] * 2, linestyle='-', color='k')
        ax.plot([inset['xlims'][0]] * 2, inset['ylims'], linestyle='-', color='k')
        ax.plot([inset['xlims'][1]] * 2, inset['ylims'], linestyle='-', color='k')

        # Link target and inset with dashed lines if possible
        if inset['xcoords'][1] < inset['xlims'][0]:
            ax.plot([inset['xcoords'][1], inset['xlims'][0]],
                    [inset['ycoords'][0], inset['ylims'][0]],
                    linestyle='--', color='k')
            ax.plot([inset['xcoords'][1], inset['xlims'][0]],
                    [inset['ycoords'][1], inset['ylims'][1]],
                    linestyle='--', color='k')
        elif inset['xcoords'][0] > inset['xlims'][1]:
            ax.plot([inset['xcoords'][0], inset['xlims'][1]],
                    [inset['ycoords'][0], inset['ylims'][0]],
                    linestyle='--', color='k')
            ax.plot([inset['xcoords'][0], inset['xlims'][1]],
                    [inset['ycoords'][1], inset['ylims'][1]],
                    linestyle='--', color='k')
        else:
            logger.warning('Inset x-coordinates intersect with those of target region')


    # Create interactive legend
    leg = ax.legend(loc=1, fontsize=fs, frameon=False)
    if blacklegend:
        for l in leg.get_lines():
            l.set_color('k')
    if straightlegend:
        for l in leg.get_lines():
            l.set_linestyle('-')
    interactive_legend = InteractiveLegend(ax.legend_, aliases)

    if showfig:
        plt.show()
    return fig




def plotBatch(directory, filepaths, vars_dict=None, plt_show=True, plt_save=False,
              ask_before_save=True, fig_ext='png', tag='fig', fs=15, lw=2, title=True,
              show_patches=True):
    ''' Plot a figure with profiles of several specific NICE output variables, for several
        NICE simulations.

        :param positions: subplot indexes of each variable
        :param filepaths: list of full paths to output data files to be compared
        :param vars_dict: dict of lists of variables names to extract and plot together
        :param plt_show: boolean stating whether to show the created figures
        :param plt_save: boolean stating whether to save the created figures
        :param ask_before_save: boolean stating whether to show the created figures
        :param fig_ext: file extension for the saved figures
        :param tag: suffix added to the end of the figures name
        :param fs: labels font size
        :param lw: curves line width
        :param title: boolean stating whether to display a general title on the figures
        :param show_patches: boolean indicating whether to indicate periods of stimulation with
         colored rectangular patches
    '''

    # Check validity of plot variables
    if vars_dict:
        yvars = list(sum(list(vars_dict.values()), []))
        for key in yvars:
            if key not in pltvars:
                raise InputError('Unknown plot variable: "{}"'.format(key))

    # Dictionary of neurons
    neurons_dict = getNeuronsDict()

    # Loop through data files
    for filepath in filepaths:

        # Get code from file name
        pkl_filename = ntpath.basename(filepath)
        filecode = pkl_filename[0:-4]

        # Retrieve sim type
        mo1 = rgxp.fullmatch(pkl_filename)
        mo2 = rgxp_mech.fullmatch(pkl_filename)
        if mo1:
            mo = mo1
        elif mo2:
            mo = mo2
        else:
            logger.error('Error: "%s" file does not match regexp pattern', pkl_filename)
            sys.exit(1)
        sim_type = mo.group(1)
        if sim_type not in ('MECH', 'ASTIM', 'ESTIM'):
            raise InputError('Invalid simulation type: {}'.format(sim_type))

        # Load data
        logger.info('Loading data from "%s"', pkl_filename)
        with open(filepath, 'rb') as fh:
            frame = pickle.load(fh)
            df = frame['data']
            meta = frame['meta']

        # Extract variables
        logger.info('Extracting variables')
        t = df['t'].values
        states = df['states'].values
        nsamples = t.size

        # Initialize channel mechanism
        if sim_type in ['ASTIM', 'ESTIM']:
            neuron_name = mo.group(2)
            global neuron
            neuron = neurons_dict[neuron_name]()
            neuron_states = [df[sn].values for sn in neuron.states_names]
            Cm0 = neuron.Cm0
            Qm0 = Cm0 * neuron.Vm0 * 1e-3
            t_plt = pltvars['t_ms']
        else:
            Cm0 = meta['Cm0']
            Qm0 = meta['Qm0']
            t_plt = pltvars['t_us']

        # Initialize BLS
        if sim_type in ['MECH', 'ASTIM']:
            global bls
            Fdrive = meta['Fdrive']
            a = meta['a']
            bls = BilayerSonophore(a, Fdrive, Cm0, Qm0)

        # Determine patches location
        npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)

        # Adding onset to time and states vectors
        if t_plt['onset'] > 0.0:
            t = np.insert(t, 0, -t_plt['onset'])
            states = np.insert(states, 0, 0)

        # Determine variables to plot if not provided
        if not vars_dict:
            if sim_type == 'ASTIM':
                vars_dict = {'Z': ['Z'], 'Q_m': ['Qm']}
            elif sim_type == 'ESTIM':
                vars_dict = {'V_m': ['Vm']}
            elif sim_type == 'MECH':
                vars_dict = {'P_{AC}': ['Pac'], 'Z': ['Z'], 'n_g': ['ng']}
            if sim_type in ['ASTIM', 'ESTIM'] and hasattr(neuron, 'pltvars_scheme'):
                vars_dict.update(neuron.pltvars_scheme)
        labels = list(vars_dict.keys())
        naxes = len(vars_dict)

        # Plotting
        if naxes == 1:
            _, ax = plt.subplots(figsize=(11, 4))
            axes = [ax]
        else:
            _, axes = plt.subplots(naxes, 1, figsize=(11, min(3 * naxes, 9)))

        for i in range(naxes):

            ax = axes[i]
            for item in ['top', 'right']:
                ax.spines[item].set_visible(False)
            ax_pltvars = [pltvars[j] for j in vars_dict[labels[i]]]
            nvars = len(ax_pltvars)

            # X-axis
            if i < naxes - 1:
                ax.get_xaxis().set_ticklabels([])
            else:
                ax.set_xlabel('${}\ ({})$'.format(t_plt['label'], t_plt['unit']), fontsize=fs)
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(fs)

            # Y-axis
            if ax_pltvars[0]['unit']:
                ax.set_ylabel('${}\ ({})$'.format(labels[i], ax_pltvars[0]['unit']),
                              fontsize=fs)
            else:
                ax.set_ylabel('${}$'.format(labels[i]), fontsize=fs)
            if 'min' in ax_pltvars[0] and 'max' in ax_pltvars[0]:
                ax_min = min([ap['min'] for ap in ax_pltvars])
                ax_max = max([ap['max'] for ap in ax_pltvars])
                ax.set_ylim(ax_min, ax_max)
            ax.locator_params(axis='y', nbins=2)
            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(fs)

            # Time series
            icolor = 0
            for j in range(nvars):

                # Extract variable
                pltvar = ax_pltvars[j]
                if 'alias' in pltvar:
                    var = eval(pltvar['alias'])
                elif 'key' in pltvar:
                    var = df[pltvar['key']].values
                elif 'constant' in pltvar:
                    var = eval(pltvar['constant']) * np.ones(nsamples)
                else:
                    var = df[vars_dict[labels[i]][j]].values
                if var.size == t.size - 1:
                    var = np.insert(var, 0, var[0])

                # Plot variable
                if 'constant' in pltvar or pltvar['desc'] in ['net current']:
                    ax.plot(t * t_plt['factor'], var * pltvar['factor'], '--', c='black', lw=lw,
                            label='${}$'.format(pltvar['label']))
                else:
                    ax.plot(t * t_plt['factor'], var * pltvar['factor'],
                            c='C{}'.format(icolor), lw=lw, label='${}$'.format(pltvar['label']))
                    icolor += 1

            # Patches
            if show_patches == 1:
                (ybottom, ytop) = ax.get_ylim()
                for j in range(npatches):
                    ax.axvspan(tpatch_on[j] * t_plt['factor'], tpatch_off[j] * t_plt['factor'],
                               edgecolor='none', facecolor='#8A8A8A', alpha=0.2)
            # Legend
            if nvars > 1:
                ax.legend(fontsize=fs, loc=7, ncol=nvars // 4 + 1)


        # Title
        if title:
            if sim_type == 'ESTIM':
                if meta['DC'] == 1.0:
                    fig_title = ESTIM_CW_title.format(neuron.name, meta['Astim'],
                                                      meta['tstim'] * 1e3)
                else:
                    fig_title = ESTIM_PW_title.format(neuron.name, meta['Astim'],
                                                      meta['tstim'] * 1e3, meta['PRF'],
                                                      meta['DC'] * 1e2)
            elif sim_type == 'ASTIM':
                if meta['DC'] == 1.0:
                    fig_title = ASTIM_CW_title.format(neuron.name, Fdrive * 1e-3,
                                                      meta['Adrive'] * 1e-3, meta['tstim'] * 1e3)
                else:
                    fig_title = ASTIM_PW_title.format(neuron.name, Fdrive * 1e-3,
                                                      meta['Adrive'] * 1e-3, meta['tstim'] * 1e3,
                                                      meta['PRF'], meta['DC'] * 1e2)
            elif sim_type == 'MECH':
                fig_title = MECH_title.format(a * 1e9, Fdrive * 1e-3, meta['Adrive'] * 1e-3)

            axes[0].set_title(fig_title, fontsize=fs)

        plt.tight_layout()

        # Save figure if needed (automatic or checked)
        if plt_save:
            if ask_before_save:
                plt_filename = SaveFigDialog(directory, '{}_{}.{}'.format(filecode, tag, fig_ext))
            else:
                plt_filename = '{}/{}_{}.{}'.format(directory, filecode, tag, fig_ext)
            if plt_filename:
                plt.savefig(plt_filename)
                logger.info('Saving figure as "{}"'.format(plt_filename))
                plt.close()

    # Show all plots if needed
    if plt_show:
        plt.show()


def plotGatingKinetics(neuron, fs=15):
    ''' Plot the voltage-dependent steady-states and time constants of activation and
        inactivation gates of the different ionic currents involved in a specific
        neuron's membrane.

        :param neuron: specific channel mechanism object
        :param fs: labels and title font size
    '''

    # Input membrane potential vector
    Vm = np.linspace(-100, 50, 300)

    xinf_dict = {}
    taux_dict = {}

    logger.info('Computing %s neuron gating kinetics', neuron.name)
    names = neuron.states_names
    print(names)
    for xname in names:
        Vm_state = True

        # Names of functions of interest
        xinf_func_str = xname.lower() + 'inf'
        taux_func_str = 'tau' + xname.lower()
        alphax_func_str = 'alpha' + xname.lower()
        betax_func_str = 'beta' + xname.lower()
        # derx_func_str = 'der' + xname.upper()

        # 1st choice: use xinf and taux function
        if hasattr(neuron, xinf_func_str) and hasattr(neuron, taux_func_str):
            xinf_func = getattr(neuron, xinf_func_str)
            taux_func = getattr(neuron, taux_func_str)
            xinf = np.array([xinf_func(v) for v in Vm])
            if isinstance(taux_func, float):
                taux = taux_func * np.ones(len(Vm))
            else:
                taux = np.array([taux_func(v) for v in Vm])

        # 2nd choice: use alphax and betax functions
        elif hasattr(neuron, alphax_func_str) and hasattr(neuron, betax_func_str):
            alphax_func = getattr(neuron, alphax_func_str)
            betax_func = getattr(neuron, betax_func_str)
            alphax = np.array([alphax_func(v) for v in Vm])
            if isinstance(betax_func, float):
                betax = betax_func * np.ones(len(Vm))
            else:
                betax = np.array([betax_func(v) for v in Vm])
            taux = 1.0 / (alphax + betax)
            xinf = taux * alphax

        # # 3rd choice: use derX choice
        # elif hasattr(neuron, derx_func_str):
        #     derx_func = getattr(neuron, derx_func_str)
        #     xinf = brentq(lambda x: derx_func(neuron.Vm, x), 0, 1)
        else:
            Vm_state = False
        if not Vm_state:
            logger.error('no function to compute %s-state gating kinetics', xname)
        else:
            xinf_dict[xname] = xinf
            taux_dict[xname] = taux

    fig, axes = plt.subplots(2)
    fig.suptitle('{} neuron: gating dynamics'.format(neuron.name))

    ax = axes[0]
    ax.get_xaxis().set_ticklabels([])
    ax.set_ylabel('$X_{\infty}$', fontsize=fs)
    for xname in names:
        if xname in xinf_dict:
            ax.plot(Vm, xinf_dict[xname], lw=2, label='$' + xname + '_{\infty}$')
    ax.legend(fontsize=fs, loc=7)

    ax = axes[1]
    ax.set_xlabel('$V_m\ (mV)$', fontsize=fs)
    ax.set_ylabel('$\\tau_X\ (ms)$', fontsize=fs)
    for xname in names:
        if xname in taux_dict:
            ax.plot(Vm, taux_dict[xname] * 1e3, lw=2, label='$\\tau_{' + xname + '}$')
    ax.legend(fontsize=fs, loc=7)

    plt.show()


def plotRateConstants(neuron, fs=15):
    ''' Plot the voltage-dependent activation and inactivation rate constants for each gate
        of all ionic currents involved in a specific neuron's membrane.

        :param neuron: specific channel mechanism object
        :param fs: labels and title font size
    '''

    # Input membrane potential vector
    Vm = np.linspace(neuron.Vm0 - 10, 50, 100)

    alphax_dict = {}
    betax_dict = {}

    logger.info('Computing %s neuron gating kinetics', neuron.name)
    names = neuron.states_names
    for xname in names:
        Vm_state = True

        # Names of functions of interest
        xinf_func_str = xname.lower() + 'inf'
        taux_func_str = 'tau' + xname.lower()
        alphax_func_str = 'alpha' + xname.lower()
        betax_func_str = 'beta' + xname.lower()

        # 1st choice: use alphax and betax functions
        if hasattr(neuron, alphax_func_str) and hasattr(neuron, betax_func_str):
            alphax_func = getattr(neuron, alphax_func_str)
            betax_func = getattr(neuron, betax_func_str)
            alphax = np.array([alphax_func(v) for v in Vm])
            betax = np.array([betax_func(v) for v in Vm])

        # 2nd choice: use xinf and taux function
        elif hasattr(neuron, xinf_func_str) and hasattr(neuron, taux_func_str):
            xinf_func = getattr(neuron, xinf_func_str)
            taux_func = getattr(neuron, taux_func_str)
            xinf = np.array([xinf_func(v) for v in Vm])
            taux = np.array([taux_func(v) for v in Vm])
            alphax = xinf / taux
            betax = 1.0 / taux - alphax

        else:
            Vm_state = False
        if not Vm_state:
            logger.error('no function to compute %s-state gating kinetics', xname)
        else:
            alphax_dict[xname] = alphax
            betax_dict[xname] = betax

    naxes = len(alphax_dict)
    _, axes = plt.subplots(naxes, figsize=(11, min(3 * naxes, 9)))

    for i, xname in enumerate(alphax_dict.keys()):
        ax1 = axes[i]
        if i == 0:
            ax1.set_title('{} neuron: rate constants'.format(neuron.name))
        if i == naxes - 1:
            ax1.set_xlabel('$V_m\ (mV)$', fontsize=fs)
        else:
            ax1.get_xaxis().set_ticklabels([])
        ax1.set_ylabel('$\\alpha_{' + xname + '}\ (ms^{-1})$', fontsize=fs, color='C0')
        for label in ax1.get_yticklabels():
            label.set_color('C0')
        ax1.plot(Vm, alphax_dict[xname] * 1e-3, lw=2)

        ax2 = ax1.twinx()
        ax2.set_ylabel('$\\beta_{' + xname + '}\ (ms^{-1})$', fontsize=fs, color='C1')
        for label in ax2.get_yticklabels():
            label.set_color('C1')
        ax2.plot(Vm, betax_dict[xname] * 1e-3, lw=2, color='C1')

    plt.tight_layout()
    plt.show()


def setGrid(n, ncolmax=3):
    ''' Determine number of rows and columns in figure grid, based on number of
        variables to plot. '''
    if n <= ncolmax:
        return (1, n)
    else:
        return ((n - 1) // ncolmax + 1, ncolmax)



def plotEffVars(neuron, Fdrive, a=32e-9, amps=None, charges=None, keys=None, fs=12, ncolmax=2):
    ''' Plot the profiles of effective variables of a specific neuron for a given frequency.
        For each variable, one line chart per amplitude is plotted, using charge as the
        input variable on the abscissa and a linear color code for the amplitude value.

        :param neuron: channel mechanism object
        :param Fdrive: acoustic drive frequency (Hz)
        :param a: sonophore diameter (m)
        :param amps: vector of amplitudes at which variables must be plotted (Pa)
        :param charges: vector of charges at which variables must be plotted (C/m2)
        :param keys: list of variables to plot
        :param fs: figure fontsize
        :param ncolmax: max number of columns on the figure
        :return: handle to the created figure
    '''

    # Check lookup file existence
    lookup_file = '{}_lookups_a{:.1f}nm.pkl'.format(neuron.name, a * 1e9)
    lookup_path = '{}/{}'.format(getLookupDir(), lookup_file)
    if not os.path.isfile(lookup_path):
        raise InputError('Missing lookup file: "{}"'.format(lookup_file))

    # Load coefficients
    with open(lookup_path, 'rb') as fh:
        lookups3D = pickle.load(fh)

    # Retrieve 1D inputs from lookup dictionary
    freqs = lookups3D.pop('f')
    amps_ref = lookups3D.pop('A')
    charges_ref = lookups3D.pop('Q')

    #  Filter lookups keys if provided
    if keys is not None:
        lookups3D = {key: lookups3D[key] for key in keys}

    # Interpolate 3D lookups at US frequency
    lookups2D = itrpLookupsFreq(lookups3D, freqs, Fdrive)
    if 'V' in lookups2D:
        lookups2D['Vm'] = lookups2D.pop('V')
        keys[keys.index('V')] = 'Vm'

    #  Define log-amplitude color code
    if amps is None:
        amps = amps_ref
    mymap = cm.get_cmap('Oranges')
    norm = matplotlib.colors.LogNorm(amps.min(), amps.max())
    sm = cm.ScalarMappable(norm=norm, cmap=mymap)
    sm._A = []

    # Plot
    logger.info('plotting')
    nrows, ncols = setGrid(len(lookups2D), ncolmax=ncolmax)
    xvar = pltvars['Qm']
    if charges is None:
        charges = charges_ref
    Qbounds = np.array([charges.min(), charges.max()]) * xvar['factor']

    fig, _ = plt.subplots(figsize=(3 * ncols, 1 * nrows), squeeze=False)
    for j, key in enumerate(keys):
        ax = plt.subplot2grid((nrows, ncols), (j // ncols, j % ncols))
        for s in ['right', 'top']:
            ax.spines[s].set_visible(False)
        yvar = pltvars[key]
        if j // ncols == nrows - 1:
            ax.set_xlabel('$\\rm {}\ ({})$'.format(xvar['label'], xvar['unit']), fontsize=fs)
            ax.set_xticks(Qbounds)
        else:
            ax.set_xticks([])
            ax.spines['bottom'].set_visible(False)

        ax.xaxis.set_label_coords(0.5, -0.1)
        ax.yaxis.set_label_coords(-0.02, 0.5)

        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)

        ymin = np.inf
        ymax = -np.inf

        # Plot effective variable for each selected amplitude
        for Adrive in amps:
            y = np.squeeze(interp2d(amps_ref, charges_ref, lookups2D[key].T)(Adrive, charges))
            ax.plot(charges * xvar['factor'], y * yvar['factor'], c=sm.to_rgba(Adrive))
            ymin = min(ymin, y.min())
            ymax = max(ymax, y.max())

        # Plot reference variable
        y0 = np.squeeze(interp2d(amps_ref, charges_ref, lookups2D[key].T)(0, charges))
        ax.plot(charges * xvar['factor'], y0 * yvar['factor'], '--', c='k')
        ymax = max(ymax, y0.max())
        ymin = min(ymin, y0.min())

        # Set axis y-limits
        if 'alpha' in key or 'beta' in key:
            ymax = min(ymax, y0.max() * 3)
        ylim = [ymin * yvar['factor'], ymax * yvar['factor']]
        if key == 'ng':
            ylim = [np.floor(ylim[0] * 1e2) / 1e2, np.ceil(ylim[1] * 1e2) / 1e2]
        else:
            ylim = [np.floor(ylim[0]), np.ceil(ylim[1])]
        dy = ylim[1] - ylim[0]
        ax.set_yticks(ylim)
        ax.set_ylim([ylim[0] - 0.05 * dy, ylim[1] + 0.05 * dy])

        # Annotate variable and unit
        xlim = ax.get_xlim()
        if np.argmax(y0) < np.argmin(y0):
            xtext = xlim[0] + 0.6 * (xlim[1] - xlim[0])
        else:
            xtext = xlim[0] + 0.01 * (xlim[1] - xlim[0])
        if key in ['Vm', 'ng']:
            ytext = ylim[0] + 0.85 * dy
        else:
            ytext = ylim[0] + 0.15 * dy
        ax.text(xtext, ytext, '$\\rm {}\ ({})$'.format(yvar['label'], yvar['unit']), fontsize=fs)

    fig.suptitle('{} neuron: original vs. effective variables @ {:.0f} kHz'.format(
        neuron.name, Fdrive * 1e-3))

    # Plot colorbar
    fig.subplots_adjust(left=0.10, bottom=0.05, top=0.9, right=0.85)
    cbarax = fig.add_axes([0.87, 0.05, 0.04, 0.85])
    fig.colorbar(sm, cax=cbarax)
    cbarax.set_ylabel('amplitude (Pa)', fontsize=fs)
    for item in cbarax.get_yticklabels():
        item.set_fontsize(fs)

    return fig


def plotActivationMap(DCs, amps, actmap, FRlims, title=None, Ascale='log', FRscale='log', fs=8):
    ''' Plot a neuron's activation map over the amplitude x duty cycle 2D space.

        :param DCs: duty cycle vector
        :param amps: amplitude vector
        :param actmap: 2D activation matrix
        :param FRlims: lower and upper bounds of firing rate color-scale
        :param title: figure title
        :param Ascale: scale to use for the amplitude dimension ('lin' or 'log')
        :param FRscale: scale to use for the firing rate coloring ('lin' or 'log')
        :param fs: fontsize to use for the title and labels
        :return: 3-tuple with the handle to the generated figure and the mesh x and y coordinates
    '''

    # Check firing rate bounding
    minFR, maxFR = (actmap[actmap > 0].min(), actmap.max())
    logger.info('FR range: %.0f - %.0f Hz', minFR, maxFR)
    if minFR < FRlims[0]:
        logger.warning('Minimal firing rate (%.0f Hz) is below defined lower bound (%.0f Hz)',
                       minFR, FRlims[0])
    if maxFR > FRlims[1]:
        logger.warning('Maximal firing rate (%.0f Hz) is above defined upper bound (%.0f Hz)',
                       maxFR, FRlims[1])

    # Plot activation map
    if FRscale == 'lin':
        norm = matplotlib.colors.Normalize(*FRlims)
    elif FRscale == 'log':
        norm = matplotlib.colors.LogNorm(*FRlims)
    fig, ax = plt.subplots(figsize=cm2inch(8, 5.8))
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.8, top=0.92)
    if title is not None:
        ax.set_title(title, fontsize=fs)
    if Ascale == 'log':
        ax.set_yscale('log')
    ax.set_xlabel('Duty cycle (%)', fontsize=fs, labelpad=-0.5)
    ax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    ax.set_xlim(np.array([DCs.min(), DCs.max()]) * 1e2)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    xedges = computeMeshEdges(DCs)
    yedges = computeMeshEdges(amps, scale='log')
    actmap[actmap == -1] = np.nan
    actmap[actmap == 0] = 1e-3
    cmap = plt.get_cmap('viridis')
    cmap.set_bad('silver')
    cmap.set_under('k')
    ax.pcolormesh(xedges * 1e2, yedges * 1e-3, actmap, cmap=cmap, norm=norm)

    # Plot firing rate colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    pos1 = ax.get_position()  # get the map axis position
    cbarax = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.03, pos1.height])
    fig.colorbar(sm, cax=cbarax)
    cbarax.set_ylabel('Firing rate (Hz)', fontsize=fs)
    for item in cbarax.get_yticklabels():
        item.set_fontsize(fs)

    return (fig, xedges, yedges)


def plotDualMaxMap(DCs, amps, maxmap, factor, actmap, title, lbl='xy', Ascale='log', fs=8):
    ''' Plot a variable maximum map over the amplitude x duty cycle 2D space, with different
        color codes for the sub and supra-threshold regions.

        :param DCs: duty cycle vector
        :param amps: amplitude vector
        :param maxmap: 2D variable maximum matrix
        :param factor: unit factor to use for the colorbars labels
        :param actmap: 2D activation matrix
        :param title: figure title
        :param lbl: indicates whether to label the x and y axes
        :param Ascale: scale to use for the amplitude dimension ('lin' or 'log')
        :param fs: fontsize to use for the title and labels
        :return: a handle to the generated figure
    '''

    # Split variable max map into sub-threshold and supra-threshold max maps
    maxmap_sub, maxmap_supra = maxmap.copy(), maxmap.copy()
    maxmap_sub[actmap >= 0] = np.nan
    maxmap_supra[actmap < 0] = np.nan

    # Plot dual max map
    fig, ax = plt.subplots(figsize=cm2inch(8, 5.8))
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.8, top=0.92)
    ax.set_title(title, fontsize=fs)
    if Ascale == 'log':
        ax.set_yscale('log')
    if 'x' in lbl:
        ax.set_xlabel('Duty cycle (%)', fontsize=fs)
    else:
        ax.set_xticklabels([])
    if 'y' in lbl:
        ax.set_ylabel('Amplitude (kPa)', fontsize=fs)
    else:
        ax.set_yticklabels([])
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(fs)
    xedges = computeMeshEdges(DCs)
    yedges = computeMeshEdges(amps, scale=Ascale)

    # Plot the 2 corresponding colorbars
    sm_sub = ax.pcolormesh(xedges * 1e2, yedges * 1e-3, maxmap_sub * factor, cmap='Blues')
    sm_supra = ax.pcolormesh(xedges * 1e2, yedges * 1e-3, maxmap_supra * factor, cmap='Reds')
    pos1 = ax.get_position()  # get the map axis position
    height = (pos1.height - 0.05) / 2.0
    cbarax_sub = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.03, height])
    cbar_sub = fig.colorbar(sm_sub, cax=cbarax_sub, format='%.1f')
    cbar_sub.set_ticks(np.array([np.nanmin(maxmap_sub), np.nanmax(maxmap_sub)]) * factor)
    cbarax_supra = fig.add_axes([pos1.x1 + 0.02, pos1.y1 - height, 0.03, height])
    cbar_supra = fig.colorbar(sm_supra, cax=cbarax_supra, format='%.1f')
    cbar_supra.set_ticks(np.array([np.nanmin(maxmap_supra), np.nanmax(maxmap_supra)]) * factor)
    for item in cbarax_sub.get_yticklabels() + cbarax_supra.get_yticklabels():
        item.set_fontsize(fs)

    return fig


def plotRawTrace(fpath, key, ybounds):
    '''  Plot the raw signal of a given variable within specified bounds.

        :param foath: full path to the data file
        :param key: key to the target variable
        :param ybounds: y-axis bounds
        :return: handle to the generated figure
    '''

    # Check file existence
    fname = ntpath.basename(fpath)
    if not os.path.isfile(fpath):
        raise InputError('Error: "{}" file does not exist'.format(fname))

    # Load data
    logger.debug('Loading data from "%s"', fname)
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
        df = frame['data']
    t = df['t'].values
    y = df[key].values * pltvars[key]['factor']

    Δy = y.max() - y.min()
    logger.info('d%s = %.1f %s', key, Δy, pltvars[key]['unit'])

    # Plot trace
    fig, ax = plt.subplots(figsize=cm2inch(12.5, 5.8))
    fig.canvas.set_window_title(fname)
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(ybounds)
    ax.plot(t, y, color='k', linewidth=1)
    fig.tight_layout()

    return fig


def plotTraces(fpath, keys, tbounds):
    '''  Plot the raw signal of sevral variables within specified bounds.

        :param foath: full path to the data file
        :param key: key to the target variable
        :param tbounds: x-axis bounds
        :return: handle to the generated figure
    '''

    # Check file existence
    fname = ntpath.basename(fpath)
    if not os.path.isfile(fpath):
        raise InputError('Error: "{}" file does not exist'.format(fname))

    # Load data
    logger.debug('Loading data from "%s"', fname)
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
        df = frame['data']
    t = df['t'].values * 1e3

    # Plot trace
    fs = 8
    fig, ax = plt.subplots(figsize=cm2inch(7, 3))
    fig.canvas.set_window_title(fname)
    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['bottom', 'left']:
        ax.spines[s].set_position(('axes', -0.03))
        ax.spines[s].set_linewidth(2)
    ax.yaxis.set_tick_params(width=2)

    # ax.spines['bottom'].set_linewidth(2)
    ax.set_xlim(tbounds)
    ax.set_xticks([])
    ymin = np.nan
    ymax = np.nan
    dt = tbounds[1] - tbounds[0]
    ax.set_xlabel('{}s'.format(si_format(dt * 1e-3, space=' ')), fontsize=fs)
    ax.set_ylabel('mV - $\\rm nC/cm^2$', fontsize=fs, labelpad=-15)

    colors = {'Vm': 'darkgrey', 'Qm': 'k'}
    for key in keys:
        y = df[key].values * pltvars[key]['factor']
        ymin = np.nanmin([ymin, y.min()])
        ymax = np.nanmax([ymax, y.max()])
        # if key == 'Qm':
            # y0 = y[0]
            # ax.plot(t, y0 * np.ones(t.size), '--', color='k', linewidth=1)
        Δy = y.max() - y.min()
        logger.info('d%s = %.1f %s', key, Δy, pltvars[key]['unit'])
        ax.plot(t, y, color=colors[key], linewidth=1)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # ax.set_yticks([ymin, ymax])
    ax.set_ylim([-200, 100])
    ax.set_yticks([-200, 100])
    for item in ax.get_yticklabels():
        item.set_fontsize(fs)
    # fig.tight_layout()
    return fig
