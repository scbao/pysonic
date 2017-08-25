# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-23 14:55:37
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-25 18:24:33

''' Plotting utilities '''

import pickle
import ntpath
import re
import logging
import inspect
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .. import channels
from ..bls import BilayerSonophore
from .pltvars import pltvars

# Get package logger
logger = logging.getLogger('PointNICE')

# Define global variables
neuron = None
bls = None

# Regular expression for input files
rgxp = re.compile('(ESTIM|ASTIM)_([A-Za-z]*)_(.*).pkl')
rgxp_mech = re.compile('(MECH)_(.*).pkl')



# Figure naming conventions
ESTIM_CW_title = '{} neuron: CW E-STIM {:.2f}mA/m2, {:.0f}ms'
ESTIM_PW_title = '{} neuron: PW E-STIM {:.2f}mA/m2, {:.0f}ms, {:.2f}kHz PRF, {:.0f}% DC'
ASTIM_CW_title = '{} neuron: CW A-STIM {:.0f}kHz, {:.0f}kPa, {:.0f}ms'
ASTIM_PW_title = '{} neuron: PW A-STIM {:.0f}kHz, {:.0f}kPa, {:.0f}ms, {:.2f}kHz PRF, {:.0f}% DC'
MECH_title = '{:.0f}nm BLS structure: MECH-STIM {:.0f}kHz, {:.0f}kPa'


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



def plotComp(yvars, filepaths, fs=15, show_patches=True):
    ''' Compare profiles of several specific output variables of NICE simulations.

        :param yvars: list of variables names to extract and compare
        :param filepaths: list of full paths to output data files to be compared
        :param fs: labels fontsize
        :param show_patches: boolean indicating whether to indicate periods of stimulation with
         colored rectangular patches
    '''

    nvars = len(yvars)

    # y variables plotting information
    y_pltvars = [pltvars[key] for key in yvars]

    # Dictionary of neurons
    neurons = {}
    for _, obj in inspect.getmembers(channels):
        if inspect.isclass(obj) and isinstance(obj.name, str):
            neurons[obj.name] = obj

    # Initialize figure and axes
    if nvars == 1:
        _, ax = plt.subplots(figsize=(11, 4))
        axes = [ax]
    else:
        _, axes = plt.subplots(nvars, 1, figsize=(11, min(3 * nvars, 9)))


    for i in range(nvars):
        ax = axes[i]
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
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(fs)
        ax.locator_params(axis='y', nbins=2)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fs)

    # Loop through data files
    j = 0
    aliases = {}
    for filepath in filepaths:

        pkl_filename = ntpath.basename(filepath)

        # Retrieve sim type
        mo1 = rgxp.fullmatch(pkl_filename)
        mo2 = rgxp_mech.fullmatch(pkl_filename)
        if mo1:
            mo = mo1
        elif mo2:
            mo = mo2
        else:
            print('Error: PKL file does not match regular expression pattern')
            quit()
        sim_type = mo.group(1)
        assert sim_type in ['MECH', 'ASTIM', 'ESTIM'], 'invalid stimulation type'

        if j == 0:
            sim_type_ref = sim_type
        else:
            assert sim_type == sim_type_ref, 'Error: comparing different simulation types'

        # Load data
        print('Loading data from "' + pkl_filename + '"')
        with open(filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        # Extract variables
        print('Extracting variables')
        t = data['t']
        states = data['states']
        nsamples = t.size

        # Initialize channel mechanism
        if sim_type in ['ASTIM', 'ESTIM']:
            neuron_name = mo.group(2)
            global neuron
            neuron = neurons[neuron_name]()
            neuron_states = [data[sn] for sn in neuron.states_names]
            Cm0 = neuron.Cm0
            Qm0 = Cm0 * neuron.Vm0 * 1e-3
            t_plt = pltvars['t_ms']
        else:
            Cm0 = data['Cm0']
            Qm0 = data['Qm0']
            t_plt = pltvars['t_us']

        # Initialize BLS
        if sim_type in ['MECH', 'ASTIM']:
            global bls
            params = data['params']
            Fdrive = data['Fdrive']
            a = data['a']
            d = data['d']
            geom = {"a": a, "d": d}
            bls = BilayerSonophore(geom, params, Fdrive, Cm0, Qm0)

        # Determine patches location
        npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)

        # Adding onset to time and states vectors
        if t_plt['onset'] > 0.0:
            t = np.insert(t, 0, -t_plt['onset'])
            states = np.insert(states, 0, 0)

        # Extract variables to plot
        vrs = []
        for i in range(nvars):
            pltvar = y_pltvars[i]
            if 'alias' in pltvar:
                var = eval(pltvar['alias'])
            elif 'key' in pltvar:
                var = data[pltvar['key']]
            elif 'constant' in pltvar:
                var = eval(pltvar['constant']) * np.ones(nsamples)
            else:
                var = data[yvars[i]]
            if var.size == t.size - 1:
                var = np.insert(var, 0, var[0])
            vrs.append(var)

        # # Determine patches location
        # npatches, tpatch_on, tpatch_off = getPatchesLoc(t, states)

        # # Adding onset to all signals
        # if t_plt['onset'] > 0.0:
        #     t = np.insert(t + t_plt['onset'], 0, 0.0)
        #     for i in range(nvars):
        #         vrs[i] = np.insert(vrs[i], 0, vrs[i][0])
        #     tpatch_on += t_plt['onset']
        #     tpatch_off += t_plt['onset']

        # Legend label
        if sim_type == 'ESTIM':
            if data['DF'] == 1.0:
                label = ESTIM_CW_title.format(neuron.name, data['Astim'], data['tstim'] * 1e3)
            else:
                label = ESTIM_PW_title.format(neuron.name, data['Astim'], data['tstim'] * 1e3,
                                              data['PRF'] * 1e-3, data['DF'] * 1e2)
        elif sim_type == 'ASTIM':
            if data['DF'] == 1.0:
                label = ASTIM_CW_title.format(neuron.name, Fdrive * 1e-3,
                                              data['Adrive'] * 1e-3, data['tstim'] * 1e3)
            else:
                label = ASTIM_PW_title.format(neuron.name, Fdrive * 1e-3,
                                              data['Adrive'] * 1e-3, data['tstim'] * 1e3,
                                              data['PRF'] * 1e-3, data['DF'] * 1e2)
        elif sim_type == 'MECH':
            label = MECH_title.format(a * 1e9, Fdrive * 1e-3, data['Adrive'] * 1e-3)

        # Plotting
        handles = [axes[i].plot(t * t_plt['factor'], vrs[i] * y_pltvars[i]['factor'],
                                linewidth=2, label=label) for i in range(nvars)]

        if show_patches:
            k = 0
            # stimulation patches
            for ax in axes:
                handle = handles[k]
                (ybottom, ytop) = ax.get_ylim()
                la = []
                for i in range(npatches):
                    la.append(ax.add_patch(Rectangle((tpatch_on[i] * t_plt['factor'], ybottom),
                                                     (tpatch_off[i] - tpatch_on[i]) * t_plt['factor'],
                                                     ytop - ybottom,
                                                     color=handle[0].get_color(), alpha=0.2)))

            aliases[handle[0]] = la
            k += 1

        j += 1

    # set x-axis label
    axes[-1].set_xlabel('${}\ ({})$'.format(t_plt['label'], t_plt['unit']), fontsize=fs)

    plt.tight_layout()

    iLegends = []
    for k in range(nvars):
        axes[k].legend(loc='upper left', fontsize=fs)
        iLegends.append(InteractiveLegend(axes[k].legend_, aliases))

    plt.show()




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

    # Dictionary of neurons
    neurons = {}
    for _, obj in inspect.getmembers(channels):
        if inspect.isclass(obj) and isinstance(obj.name, str):
            neurons[obj.name] = obj

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
            print('Error: PKL file does not match regular expression pattern')
            quit()
        sim_type = mo.group(1)
        assert sim_type in ['MECH', 'ASTIM', 'ESTIM'], 'invalid stimulation type'

        # Load data
        print('Loading data from "' + pkl_filename + '"')
        with open(filepath, 'rb') as pkl_file:
            data = pickle.load(pkl_file)

        # Extract variables
        print('Extracting variables')
        t = data['t']
        states = data['states']
        nsamples = t.size

        # Initialize channel mechanism
        if sim_type in ['ASTIM', 'ESTIM']:
            neuron_name = mo.group(2)
            global neuron
            neuron = neurons[neuron_name]()
            neuron_states = [data[sn] for sn in neuron.states_names]
            Cm0 = neuron.Cm0
            Qm0 = Cm0 * neuron.Vm0 * 1e-3
            t_plt = pltvars['t_ms']
        else:
            Cm0 = data['Cm0']
            Qm0 = data['Qm0']
            t_plt = pltvars['t_us']

        # Initialize BLS
        if sim_type in ['MECH', 'ASTIM']:
            global bls
            params = data['params']
            Fdrive = data['Fdrive']
            a = data['a']
            d = data['d']
            geom = {"a": a, "d": d}
            bls = BilayerSonophore(geom, params, Fdrive, Cm0, Qm0)

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
                    var = data[pltvar['key']]
                elif 'constant' in pltvar:
                    var = eval(pltvar['constant']) * np.ones(nsamples)
                else:
                    var = data[vars_dict[labels[i]][j]]
                if var.size == t.size - 1:
                    var = np.insert(var, 0, var[0])

                # Plot variable
                if 'constant' in pltvar:
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
                    ax.add_patch(Rectangle((tpatch_on[j] * t_plt['factor'], ybottom),
                                           (tpatch_off[j] - tpatch_on[j]) * t_plt['factor'],
                                           ytop - ybottom, color='#8A8A8A', alpha=0.1))

            # Legend
            if nvars > 1:
                ax.legend(fontsize=fs, loc=7, ncol=nvars // 4 + 1)


        # Title
        if title:
            if sim_type == 'ESTIM':
                if data['DF'] == 1.0:
                    fig_title = ESTIM_CW_title.format(neuron.name, data['Astim'],
                                                      data['tstim'] * 1e3)
                else:
                    fig_title = ESTIM_PW_title.format(neuron.name, data['Astim'],
                                                      data['tstim'] * 1e3, data['PRF'] * 1e-3,
                                                      data['DF'] * 1e2)
            elif sim_type == 'ASTIM':
                if data['DF'] == 1.0:
                    fig_title = ASTIM_CW_title.format(neuron.name, Fdrive * 1e-3,
                                                      data['Adrive'] * 1e-3, data['tstim'] * 1e3)
                else:
                    fig_title = ASTIM_PW_title.format(neuron.name, Fdrive * 1e-3,
                                                      data['Adrive'] * 1e-3, data['tstim'] * 1e3,
                                                      data['PRF'] * 1e-3, data['DF'] * 1e2)
            elif sim_type == 'MECH':
                fig_title = MECH_title.format(a * 1e9, Fdrive * 1e-3, data['Adrive'] * 1e-3)

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
                print('Saving figure as "{}"'.format(plt_filename))
                plt.close()

    # Show all plots if needed
    if plt_show:
        plt.show()
