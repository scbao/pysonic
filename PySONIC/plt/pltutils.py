# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-23 14:55:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-25 14:41:53

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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter

from ..utils import rescale, si_format
from ..core import BilayerSonophore
from .pltvars import pltvars
from ..neurons import getNeuronsDict

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'

# Get package logger
logger = logging.getLogger('PySONIC')

# Define global variables
neuron = None
bls = None
timeunits = {'ASTIM': 't_ms', 'ESTIM': 't_ms', 'MECH': 't_us'}

# Regular expression for input files
rgxp = re.compile('(ESTIM|ASTIM)_([A-Za-z]*)_(.*).pkl')
rgxp_mech = re.compile('(MECH)_(.*).pkl')

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


def computeMeshEdges(x, scale='lin'):
    ''' Compute the appropriate edges of a mesh that quads a linear or logarihtmic distribution.

        :param x: the input vector
        :param scale: the type of distribution ('lin' for linear, 'log' for logarihtmic)
        :return: the edges vector
    '''
    if scale is 'log':
        x = np.log10(x)
    dx = x[1] - x[0]
    if scale is 'lin':
        y = np.linspace(x[0] - dx / 2, x[-1] + dx / 2, x.size + 1)
    elif scale is 'log':
        y = np.logspace(x[0] - dx / 2, x[-1] + dx / 2, x.size + 1)
    return y


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
    ipatch_off = np.where(dstates < 0.0)[0] + 1
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
        raise KeyError('Unknown plot variable: "{}"'.format(varname))
    pltvar = pltvars[varname]

    # Input check 2: labels
    if labels is not None:
        if len(labels) != len(filepaths):
            raise AssertionError('Invalid labels ({}): not matching number of compared files ({})'
                                 .format(len(labels), len(filepaths)))
        if not all(isinstance(x, str) for x in labels):
            raise TypeError('Invalid labels: must be string typed')

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
            raise AssertionError('Invalid patches ({}): not matching number of compared files ({})'
                                 .format(len(patches), len(filepaths)))
        if not all(isinstance(p, bool) for p in patches):
            raise TypeError('Invalid patch sequence: all list items must be boolean typed')
    else:
        raise ValueError('Invalid patches: must be either "none", all", "one", or a boolean list')

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
            raise ValueError('Invalid simulation type: {}'.format(sim_type))

        if j == 0:
            sim_type_ref = sim_type
            t_plt = pltvars[timeunits[sim_type]]
        elif sim_type != sim_type_ref:
            raise ValueError('Invalid comparison: different simulation types')

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
            bls = BilayerSonophore(meta['a'], Cm0, Qm0)

        # Determine patches location
        npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)

        # Add onset to time vectors
        if t_plt['onset'] > 0.0:
            tonset = np.array([-t_plt['onset'], -t[0] - t[1]])
            t = np.hstack((tonset, t))
            states = np.hstack((states, np.zeros(2)))

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
        if var.size == t.size - 2:
            if varname is 'Vm':
                var = np.hstack((np.array([neuron.Vm0] * 2), var))
            else:
                var = np.hstack((np.array([var[0]] * 2), var))
                # var = np.insert(var, 0, var[0])

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
                raise KeyError('Unknown plot variable: "{}"'.format(key))

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
            raise ValueError('Invalid simulation type: {}'.format(sim_type))

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
            bls = BilayerSonophore(a, Cm0, Qm0)

        # Determine patches location
        npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)

        # Adding onset to time vector
        if t_plt['onset'] > 0.0:
            tonset = np.array([-t_plt['onset'], -t[0] - t[1]])
            t = np.hstack((tonset, t))
            states = np.hstack((states, np.zeros(2)))

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
                if var.size == t.size - 2:
                    if pltvar['desc'] == 'membrane potential':
                        var = np.hstack((np.array([neuron.Vm0] * 2), var))
                    else:
                        var = np.hstack((np.array([var[0]] * 2), var))
                        # var = np.insert(var, 0, var[0])

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
        raise FileNotFoundError('Error: "{}" file does not exist'.format(fname))

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
        raise FileNotFoundError('Error: "{}" file does not exist'.format(fname))

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


def plotSignals(t, signals, states=None, ax=None, onset=None, lbls=None, fs=10, cmode='qual'):
    ''' Plot several signals on one graph.

        :param t: time vector
        :param signals: list of signal vectors
        :param states (optional): stimulation state vector
        :param ax (optional): handle to figure axis
        :param onset (optional): onset to add to signals on graph
        :param lbls (optional): list of legend labels
        :param fs (optional): font size to use on graph
        :param cmode: color mode ('seq' for sequentiual or 'qual' for qualitative)
        :return: figure handle (if no axis provided as input)
    '''


    # If no axis provided, create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        argout = fig
    else:
        argout = None

    # Set axis aspect
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for item in ax.get_xticklabels() + ax.get_xticklabels():
        item.set_fontsize(fs)

    # Compute number of signals
    nsignals = len(signals)

    # Adapt labels for sequential color mode
    if cmode == 'seq' and lbls is not None:
        lbls[1:-1] = ['.'] * (nsignals - 2)

    # Add stimulation patches if states provided
    if states is not None:
        npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)
        for i in range(npatches):
            ax.axvspan(tpatch_on[i], tpatch_off[i], edgecolor='none',
                       facecolor='#8A8A8A', alpha=0.2)

    # Add onset of provided
    if onset is not None:
        t0, y0 = onset
        t = np.hstack((np.array([t0, 0.]), t))
        signals = np.hstack((np.ones((nsignals, 2)) * y0, signals))

    # Determine colorset
    nlevels = nsignals
    if cmode == 'seq':
        norm = matplotlib.colors.Normalize(0, nlevels - 1)
        sm = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('viridis'))
        sm._A = []
        colors = [sm.to_rgba(i) for i in range(nlevels)]
    elif cmode == 'qual':
        nlevels_max = 10
        if nlevels > nlevels_max:
            raise Warning('Number of signals higher than number of color levels')
        colors = ['C{}'.format(i) for i in range(nlevels)]
    else:
        raise ValueError('Unknown color mode')

    # Plot signals
    for i, var in enumerate(signals):
        ax.plot(t, var, label=lbls[i] if lbls is not None else None, c=colors[i])

    # Add legend
    if lbls is not None:
        ax.legend(fontsize=fs, frameon=False)

    return argout
