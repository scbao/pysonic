# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-25 16:18:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-26 11:48:36

import ntpath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

from ..utils import *
from .pltutils import *


class InteractiveLegend(object):
    ''' Class defining an interactive matplotlib legend, where lines visibility can
    be toggled by simply clicking on the corresponding legend label. Other graphic
    objects can also be associated to the toggle of a specific line

    Adapted from:
    http://stackoverflow.com/questions/31410043/hiding-lines-after-showing-a-pyplot-figure
    '''

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
        plt.show()


def plotComp(filepaths, varname, labels=None, fs=10, lw=2, colors=None, lines=None, patches='one',
             xticks=None, yticks=None, blacklegend=False, straightlegend=False,
             inset=None, figsize=(11, 4)):
    ''' Compare profiles of several specific output variables of NICE simulations.

        :param filepaths: list of full paths to output data files to be compared
        :param varname: name of variable to extract and compare
        :param labels: list of labels to use in the legend
        :param fs: labels fontsize
        :param patches: string indicating whether to indicate periods of stimulation with
         colored rectangular patches
    '''

    # Input check: labels
    if labels is not None:
        if len(labels) != len(filepaths):
            raise AssertionError('Invalid labels ({}): not matching number of compared files ({})'
                                 .format(len(labels), len(filepaths)))
        if not all(isinstance(x, str) for x in labels):
            raise TypeError('Invalid labels: must be string typed')

    # Input check: line styles and colors
    if colors is None:
        colors = ['C{}'.format(j) for j in range(len(filepaths))]
    if lines is None:
        lines = ['-'] * len(filepaths)

    # Input check: STIM-ON patches
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

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_zorder(0)
    if inset is not None:
        inset_ax = fig.add_axes(ax.get_position())
        inset_ax.set_zorder(1)
        inset_ax.set_xlim(inset['xlims'][0], inset['xlims'][1])
        inset_ax.set_ylim(inset['ylims'][0], inset['ylims'][1])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        # inset_ax.patch.set_alpha(1.0)
        inset_ax.add_patch(Rectangle((inset['xlims'][0], inset['ylims'][0]),
                                     inset['xlims'][1] - inset['xlims'][0],
                                     inset['ylims'][1] - inset['ylims'][0],
                                     color='w'))

    # Loop through data files
    aliases = {}
    for j, filepath in enumerate(filepaths):

        # Retrieve sim type
        pkl_filename = ntpath.basename(filepath)
        sim_type = getSimType(pkl_filename)

        if j == 0:
            sim_type_ref = sim_type
        elif sim_type != sim_type_ref:
            raise ValueError('Invalid comparison: different simulation types')

        # Load data and extract variables
        df, meta = loadData(filepath)
        t = df['t'].values
        stimstate = df['stimstate'].values

        # Determine stimulus patch from stimstate
        _, tpatch_on, tpatch_off = getStimPulses(t, stimstate)

        # Initialize appropriate object
        obj = getObject(sim_type, meta)

        # Retrieve plot variables
        tvar, pltvars = getTimePltVar(obj.tscale), obj.getPltVars()

        # Retrieve appropriate plot variable
        if varname not in pltvars:
            raise KeyError('Unknown plot variable: "{}". Possible plot variables are: {}'.format(
                varname, ', '.join(['"{}"'.format(p) for p in pltvars.keys()])))
        pltvar = pltvars[varname]

        # Preset and rescale time vector
        if tvar['onset'] > 0.0:
            tonset = np.array([-tvar['onset'], -t[0] - t[1]])
            t = np.hstack((tonset, t))
        t *= tvar['factor']

        # Extract variable and plot time series
        var = extractPltVar(obj, pltvar, df, meta, t.size, varname)
        handle = ax.plot(t, var, linewidth=lw, linestyle=lines[j], color=colors[j],
                         label=labels[j] if labels is not None else figtitle(meta))

        if inset is not None:
            inset_window = np.logical_and(t > (inset['xlims'][0] / tvar['factor']),
                                          t < (inset['xlims'][1] / tvar['factor']))
            inset_ax.plot(t[inset_window] * tvar['factor'], var[inset_window] * pltvar['factor'],
                          linewidth=lw, linestyle=lines[j], color=colors[j])

        # Add optional STIM-ON patches
        if patches[j]:
            (ybottom, ytop) = ax.get_ylim()
            la = []
            color = '#8A8A8A' if greypatch else handle[0].get_color()
            for i in range(tpatch_on.size):
                la.append(ax.axvspan(tpatch_on[i] * tvar['factor'], tpatch_off[i] * tvar['factor'],
                                     edgecolor='none', facecolor=color, alpha=0.2))
            aliases[handle[0]] = la

            if inset is not None:
                cond_on = np.logical_and(tpatch_on > (inset['xlims'][0] / tvar['factor']),
                                         tpatch_on < (inset['xlims'][1] / tvar['factor']))
                cond_off = np.logical_and(tpatch_off > (inset['xlims'][0] / tvar['factor']),
                                          tpatch_off < (inset['xlims'][1] / tvar['factor']))
                cond_glob = np.logical_and(tpatch_on < (inset['xlims'][0] / tvar['factor']),
                                           tpatch_off > (inset['xlims'][1] / tvar['factor']))
                cond_onoff = np.logical_or(cond_on, cond_off)
                cond = np.logical_or(cond_onoff, cond_glob)
                npatches_inset = np.sum(cond)
                for i in range(npatches_inset):
                    inset_ax.add_patch(Rectangle((tpatch_on[cond][i] * tvar['factor'], ybottom),
                                                 (tpatch_off[cond][i] - tpatch_on[cond][i]) *
                                                 tvar['factor'], ytop - ybottom, color=color,
                                                 alpha=0.1))

    # Post-process figure
    for item in ['top', 'right']:
        ax.spines[item].set_visible(False)
    if 'bounds' in pltvar:
        ax.set_ylim(*pltvar['bounds'])
    ax.set_xlabel('$\\rm {}\ ({})$'.format(tvar['label'], tvar['unit']), fontsize=fs)
    ax.set_ylabel('$\\rm {}\ ({})$'.format(pltvar['label'], pltvars.get('unit', '')), fontsize=fs)
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

    return fig
