# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-25 16:18:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-15 19:14:17

import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle

from ..core import getModel
from ..postpro import findPeaks
from ..utils import *
from ..constants import SPIKE_MIN_DT, SPIKE_MIN_QAMP, SPIKE_MIN_QPROM
from .pltutils import *


class TimeSeriesPlot:
    ''' Generic interface to build a plot displaying temporal profiles of model simulations. '''

    def __init__(self, filepaths):
        ''' Constructor.

            :param filepaths: list of full paths to output data files to be compared
        '''
        if not isIterable(filepaths):
            filepaths = [filepaths]
        self.filepaths = filepaths

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)

    def getData(self, entry, frequency):
        if isinstance(entry, str):
            data, meta = loadData(entry, frequency)
        else:
            data, meta = entry
            data = data.iloc[::frequency]
        return data, meta

    def render(*args, **kwargs):
        return NotImplementedError

    def checkInputs(self, *args, **kwargs):
        return NotImplementedError

    def createBackBone(self, *args, **kwargs):
        return NotImplementedError

    def getSimType(self, fname):
        ''' Get sim type from filename. '''
        mo = re.search('(^[A-Z]*)_(.*).pkl', fname)
        if not mo:
            raise ValueError('Could not find sim-key in filename: "{}"'.format(fname))
        return mo.group(1)

    def getTimePltVar(self, tscale):
        ''' Return time plot variable for a given temporal scale. '''
        return {
            'desc': 'time',
            'label': 'time',
            'unit': tscale,
            'factor': {'ms': 1e3, 'us': 1e6}[tscale],
            'onset': {'ms': 1e-3, 'us': 1e-6}[tscale]
        }

    def getStimPulses(self, t, states):
        ''' Determine the onset and offset times of pulses from a stimulation vector.

            :param t: time vector (s).
            :param states: a vector of stimulation state (ON/OFF) at each instant in time.
            :return: 3-tuple with number of patches, timing of STIM-ON an STIM-OFF instants.
        '''
        # Compute states derivatives and identify bounds indexes of pulses
        dstates = np.diff(states)
        ipulse_on = np.insert(np.where(dstates > 0.0)[0] + 1, 0, 0)
        ipulse_off = np.where(dstates < 0.0)[0] + 1
        if ipulse_off.size < ipulse_on.size:
            ioff = t.size - 1
            if ipulse_off.size == 0:
                ipulse_off = np.array([ioff])
            else:
                ipulse_off = np.insert(ipulse_off, ipulse_off.size - 1, ioff)

        # Get time instants for pulses ON and OFF
        tpulse_on = t[ipulse_on]
        tpulse_off = t[ipulse_off]
        return tpulse_on, tpulse_off

    def addLegend(self, ax, handles, labels, fs, black=False, straight=False, interactive=False):
        lh = ax.legend(handles, labels, loc=1, fontsize=fs, frameon=False)
        if black:
            for l in lh.get_lines():
                l.set_color('k')
        if straight:
            for l in lh.get_lines():
                l.set_linestyle('-')

    def getStimStates(self, df):
        try:
            stimstate = df['stimstate']
        except KeyError:
            stimstate = df['states']
        return stimstate.values

    def materializeSpikes(self, ax, data, tf, yf, color, mode, add_to_legend=False):
        if 'Qm' not in data:
            raise ValueError('charge profile not avilable in dataframe')

        # Get spikes details
        t, Qm = [data[k].values for k in['t', 'Qm']]
        dt = t[1] - t[0]
        mpd = int(np.ceil(SPIKE_MIN_DT / dt))
        ipeaks, prominences, widths, halfmaxbounds, _ = findPeaks(
            data['Qm'].values, mph=SPIKE_MIN_QAMP, mpd=mpd, mpp=SPIKE_MIN_QPROM)
        if ipeaks is not None:
            ax.scatter(t[ipeaks] * tf, Qm[ipeaks] * yf + 10, color=color,
                       label='spikes' if add_to_legend else None, marker='v')
            if mode == 'details':
                widths *= dt
                indexes = np.arange(t.size)
                tleftbounds = np.interp(halfmaxbounds[:, 0], indexes, t, left=np.nan, right=np.nan)
                trightbounds = np.interp(halfmaxbounds[:, 1], indexes, t, left=np.nan, right=np.nan)
                for i in range(len(ipeaks)):
                    ax.plot(np.array([t[ipeaks[i]]] * 2) * tf,
                            np.array([Qm[ipeaks[i]], Qm[ipeaks[i]] - prominences[i]]) * yf,
                            '--', color=color,
                            label='prominences' if i == 0 and add_to_legend else '')
                    ax.plot(np.array([tleftbounds[i], trightbounds[i]]) * tf,
                            np.array([Qm[ipeaks[i]] - 0.5 * prominences[i]] * 2) * yf,
                            '-.', color=color,
                            label='widths' if i == 0 and add_to_legend else '')

        return add_to_legend

    def prepareTime(self, t, tplt):
        if tplt['onset'] > 0.0:
            tonset = np.array([-tplt['onset'], -t[0] - t[1]])
            t = np.hstack((tonset, t))
        return t * tplt['factor']

    def addPatches(self, ax, tpatch_on, tpatch_off, tfactor, color='#8A8A8A'):
        for i in range(tpatch_on.size):
            ax.axvspan(tpatch_on[i] * tfactor, tpatch_off[i] * tfactor,
                       edgecolor='none', facecolor=color, alpha=0.2)

    def postProcess(self, *args, **kwargs):
        return NotImplementedError

    def addInset(self, fig, ax, inset):
        ''' Create inset axis. '''
        inset_ax = fig.add_axes(ax.get_position())
        inset_ax.set_zorder(1)
        inset_ax.set_xlim(inset['xlims'][0], inset['xlims'][1])
        inset_ax.set_ylim(inset['ylims'][0], inset['ylims'][1])
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.add_patch(Rectangle((inset['xlims'][0], inset['ylims'][0]),
                                     inset['xlims'][1] - inset['xlims'][0],
                                     inset['ylims'][1] - inset['ylims'][0],
                                     color='w'))
        return inset_ax

    def materializeInset(self, ax, inset_ax, inset):
        ''' Materialize inset with zoom boox. '''
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

    def addInsetPatches(self, ax, inset_ax, inset, tpatch_on, tpatch_off, tfactor, color):
        ybottom, ytop = ax.get_ylim()
        cond_on = np.logical_and(tpatch_on > (inset['xlims'][0] / tfactor),
                                 tpatch_on < (inset['xlims'][1] / tfactor))
        cond_off = np.logical_and(tpatch_off > (inset['xlims'][0] / tfactor),
                                  tpatch_off < (inset['xlims'][1] / tfactor))
        cond_glob = np.logical_and(tpatch_on < (inset['xlims'][0] / tfactor),
                                   tpatch_off > (inset['xlims'][1] / tfactor))
        cond_onoff = np.logical_or(cond_on, cond_off)
        cond = np.logical_or(cond_onoff, cond_glob)
        npatches_inset = np.sum(cond)
        for i in range(npatches_inset):
            inset_ax.add_patch(Rectangle((tpatch_on[cond][i] * tfactor, ybottom),
                                         (tpatch_off[cond][i] - tpatch_on[cond][i]) *
                                         tfactor, ytop - ybottom, color=color,
                                         alpha=0.1))

    def removeSpines(self, ax):
        for item in ['top', 'right']:
            ax.spines[item].set_visible(False)

    def setTimeLabel(self, ax, tplt, fs):
        ax.set_xlabel('$\\rm {}\ ({})$'.format(tplt['label'], tplt['unit']), fontsize=fs)

    def setYLabel(self, ax, yplt, fs, grouplabel=None):
        lbl = grouplabel if grouplabel is not None else yplt['label']
        ax.set_ylabel('$\\rm {}\ ({})$'.format(lbl, yplt.get('unit', '')), fontsize=fs)

    def setYTicks(self, ax, yticks=None):
        if yticks is not None:
            ax.set_yticks(yticks)

    def setTickLabelsFontSize(self, ax, fs):
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fs)


class ComparativePlot(TimeSeriesPlot):
    ''' Interface to build a comparative plot displaying profiles of a specific output variable
        across different model simulations. '''

    def __init__(self, filepaths, varname):
        ''' Constructor.

            :param filepaths: list of full paths to output data files to be compared
            :param varname: name of variable to extract and compare
        '''
        super().__init__(filepaths)
        self.varname = varname

    def checkInputs(self, lines, labels, colors, patches):
        # Input check: labels
        if labels is not None:
            if len(labels) != len(self.filepaths):
                raise ValueError(
                    'Invalid labels ({}): not matching number of compared files ({})'.format(
                        len(labels), len(self.filepaths)))
            if not all(isinstance(x, str) for x in labels):
                raise TypeError('Invalid labels: must be string typed')

        # Input check: line styles and colors
        if colors is None:
            colors = ['C{}'.format(j) for j in range(len(self.filepaths))]
        if lines is None:
            lines = ['-'] * len(self.filepaths)

        # Input check: STIM-ON patches
        greypatch = False
        if patches == 'none':
            patches = [False] * len(self.filepaths)
        elif patches == 'all':
            patches = [True] * len(self.filepaths)
        elif patches == 'one':
            patches = [True] + [False] * (len(self.filepaths) - 1)
            greypatch = True
        elif isinstance(patches, list):
            if len(patches) != len(self.filepaths):
                raise ValueError(
                    'Invalid patches ({}): not matching number of compared files ({})'.format(
                        len(patches), len(self.filepaths)))
            if not all(isinstance(p, bool) for p in patches):
                raise TypeError('Invalid patch sequence: all list items must be boolean typed')
        else:
            raise ValueError(
                'Invalid patches: must be either "none", all", "one", or a boolean list')
        return lines, labels, colors, patches, greypatch

    def createBackBone(self, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_zorder(0)
        return fig, ax

    def postProcess(self, ax, tplt, yplt, fs, xticks, yticks):
        self.removeSpines(ax)
        if 'bounds' in yplt:
            ax.set_ylim(*yplt['bounds'])
        self.setTimeLabel(ax, tplt, fs)
        self.setYLabel(ax, yplt, fs, grouplabel=None)
        if xticks is not None:  # optional x-ticks
            ax.set_xticks(xticks)
        self.setYTicks(ax, yticks)
        self.setTickLabelsFontSize(ax, fs)

    def addCmap(self, fig, cmap, handles, zvalues, zinfo, fs, zscale='lin'):
        # Create colormap and normalizer
        mymap = plt.get_cmap(cmap)
        if zscale == 'lin':
            norm = matplotlib.colors.Normalize(zvalues.min(), zvalues.max())
        elif zscale == 'log':
            norm = matplotlib.colors.LogNorm(zvalues.min(), zvalues.max())
        sm = cm.ScalarMappable(norm=norm, cmap=mymap)
        sm._A = []

        # Adjust line colors
        for lh, z in zip(handles, zvalues):
            lh.set_color(sm.to_rgba(z))

        # Add colorbar
        fig.subplots_adjust(left=0.1, right=0.8, bottom=0.15, top=0.95, hspace=0.5)
        cbarax = fig.add_axes([0.85, 0.15, 0.03, 0.8])
        fig.colorbar(sm, cax=cbarax, orientation='vertical')
        cbarax.set_ylabel('$\\rm {}\ ({})$'.format(zinfo['label'], zinfo['unit']), fontsize=fs)
        for item in cbarax.get_yticklabels():
            item.set_fontsize(fs)

    def render(self, figsize=(11, 4), fs=10, lw=2, labels=None, colors=None, lines=None,
               patches='one', xticks=None, yticks=None, blacklegend=False, straightlegend=False,
               inset=None, frequency=1, spikes='none', cmap=None, cscale='lin'):
        ''' Render plot.

            :param figsize: figure size (x, y)
            :param fs: labels fontsize
            :param lw: linewidth
            :param labels: list of labels to use in the legend
            :param colors: list of colors to use for each curve
            :param lines: list of linestyles
            :param patches: string indicating whether/how to mark stimulation periods
                with rectangular patches
            :param xticks: list of x-ticks
            :param yticks: list of y-ticks
            :param blacklegend: boolean indicating whether to use black lines in the legend
            :param straightlegend: boolean indicating whether to use straight lines in the legend
            :param inset: string indicating whether/how to mark an inset zooming on
                a particular region of the graph
            :param frequency: frequency at which to plot samples
            :param spikes: string indicating how to show spikes ("none", "marks" or "details")
            :return: figure handle
        '''
        lines, labels, colors, patches, greypatch = self.checkInputs(
            lines, labels, colors, patches)

        fig, ax = self.createBackBone(figsize)
        if inset is not None:
            inset_ax = self.addInset(fig, ax, inset)

        handles = []
        meta_ref = None
        zref = None
        zvalues = []
        full_labels = []

        # Loop through data files
        for j, filepath in enumerate(self.filepaths):

            # Load data
            data, meta = self.getData(filepath, frequency)
            meta.pop('tcomp')
            full_labels.append(figtitle(meta))

            # Extract model
            model = getModel(meta)

            # Check consistency of sim types and check differing inputs
            if meta_ref is None:
                meta_ref = meta
            else:
                if meta['simkey'] != meta_ref['simkey']:
                    raise ValueError('Invalid comparison: different simulation types')
                differing = {k: meta[k] != meta_ref[k] for k in meta.keys()}
                if sum(differing.values()) > 1:
                    raise ValueError('More than one differing inputs')
                zkey = (list(differing.keys())[list(differing.values()).index(True)])
                if zref is None:
                    zref = zkey
                    zvalues.append(meta_ref[zkey])
                    zinfo = model.inputVars().get(zkey, None)
                else:
                    if zkey != zref:
                        raise ValueError('inconsitent differing input')
                zvalues.append(meta[zkey])

            # Extract time and stim pulses
            t = data['t'].values
            stimstate = self.getStimStates(data)
            tpatch_on, tpatch_off = self.getStimPulses(t, stimstate)
            tplt = self.getTimePltVar(model.tscale)
            t = self.prepareTime(t, tplt)

            # Extract y-variable
            pltvars = model.getPltVars()
            if self.varname not in pltvars:
                raise KeyError(
                    'Unknown plot variable: "{}". Possible plot variables are: {}'.format(
                        self.varname, ', '.join(['"{}"'.format(p) for p in pltvars.keys()])))
            yplt = pltvars[self.varname]
            y = extractPltVar(model, yplt, data, meta, t.size, self.varname)

            #  Plot time series
            handles.append(
                ax.plot(t, y, linewidth=lw, linestyle=lines[j], color=colors[j])[0])

            # Optional: add spikes
            if self.varname == 'Qm' and spikes != 'none':
                self.materializeSpikes(
                    ax, data, tplt['factor'], yplt['factor'], colors[j], spikes)

            # Plot optional inset
            if inset is not None:
                inset_window = np.logical_and(t > (inset['xlims'][0] / tplt['factor']),
                                              t < (inset['xlims'][1] / tplt['factor']))
                inset_ax.plot(t[inset_window] * tplt['factor'], y[inset_window] * yplt['factor'],
                              linewidth=lw, linestyle=lines[j], color=colors[j])

            # Add optional STIM-ON patches
            if patches[j]:
                ybottom, ytop = ax.get_ylim()
                color = '#8A8A8A' if greypatch else handles[j].get_color()
                self.addPatches(ax, tpatch_on, tpatch_off, tplt['factor'], color)
                if inset is not None:
                    self.addInsetPatches(
                        ax, inset_ax, inset, tpatch_on, tpatch_off, tplt['factor'], color)
        if zinfo is not None:
            zvalues = np.array(zvalues) * zinfo['factor']
            zlabels = ['$\\rm{} = {}\ {}$'.format(zinfo['label'], z, zinfo['unit'])
                       for z in zvalues]
        else:
            zlabels = zvalues

        if labels is None:
            if zlabels is not None:
                labels = zlabels
            else:
                labels = full_labels

        # Postprocess figure
        self.postProcess(ax, tplt, yplt, fs, xticks, yticks)
        fig.tight_layout()

        if inset is not None:
            self.materializeInset(ax, inset_ax, inset)

        # Add color legend or label legend
        if cmap is not None:
            if zinfo is None:
                raise ValueError('Colormap mode unavailable for qualitative comparisons')
            self.addCmap(fig, cmap, handles, zvalues, zinfo, fs, zscale=cscale)
        else:
            self.addLegend(ax, handles, labels, fs, black=blacklegend, straight=straightlegend)

        return fig


class SchemePlot(TimeSeriesPlot):
    ''' Interface to build a plot displaying profiles of several output variables
        arranged into specific schemes. '''

    def __init__(self, filepaths, pltscheme=None):
        ''' Constructor.

            :param filepaths: list of full paths to output data files to be compared
            :param varname: name of variable to extract and compare
        '''
        super().__init__(filepaths)
        self.pltscheme = pltscheme

    def createBackBone(self, pltscheme):
        naxes = len(pltscheme)
        if naxes == 1:
            fig, ax = plt.subplots(figsize=(11, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(naxes, 1, figsize=(11, min(3 * naxes, 9)))
        return fig, axes

    def postProcess(self, axes, tplt, yplt, fs):
        for ax in axes:
            self.removeSpines(ax)
            self.setTickLabelsFontSize(ax, fs)
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        self.setTimeLabel(axes[-1], tplt, fs)

    def render(self, fs=10, lw=2, labels=None, colors=None, lines=None, patches=True, title=True,
               save=False, directory=None, fig_ext='png', frequency=1, spikes='none'):

        figs = []
        for filepath in self.filepaths:

            # Load data and extract model
            data, meta = self.getData(filepath, frequency)
            model = getModel(meta)

            # Extract time and stim pulses
            t = data['t'].values
            stimstate = self.getStimStates(data)

            tpatch_on, tpatch_off = self.getStimPulses(t, stimstate)
            tplt = self.getTimePltVar(model.tscale)
            t = self.prepareTime(t, tplt)

            # Check plot scheme if provided, otherwise generate it
            pltvars = model.getPltVars()
            if self.pltscheme is not None:
                for key in list(sum(list(self.pltscheme.values()), [])):
                    if key not in pltvars:
                        raise KeyError('Unknown plot variable: "{}"'.format(key))
                pltscheme = self.pltscheme
            else:
                pltscheme = model.getPltScheme()

            # Create figure
            fig, axes = self.createBackBone(pltscheme)

            # Loop through each subgraph
            for ax, (grouplabel, keys) in zip(axes, pltscheme.items()):
                ax_legend_spikes = False

                # Extract variables to plot
                nvars = len(keys)
                ax_pltvars = [pltvars[k] for k in keys]
                if nvars == 1:
                    ax_pltvars[0]['color'] = 'k'
                    ax_pltvars[0]['ls'] = '-'

                # Set y-axis unit and bounds
                self.setYLabel(ax, ax_pltvars[0], fs, grouplabel=grouplabel)
                if 'bounds' in ax_pltvars[0]:
                    ax_min = min([ap['bounds'][0] for ap in ax_pltvars])
                    ax_max = max([ap['bounds'][1] for ap in ax_pltvars])
                    ax.set_ylim(ax_min, ax_max)

                # Plot time series
                icolor = 0
                for yplt, name in zip(ax_pltvars, pltscheme[grouplabel]):
                    color = yplt.get('color', 'C{}'.format(icolor))
                    y = extractPltVar(model, yplt, data, meta, t.size, name)
                    ax.plot(t, y, yplt.get('ls', '-'), c=color, lw=lw,
                            label='$\\rm {}$'.format(yplt['label']))
                    if 'color' not in yplt:
                        icolor += 1

                    # Optional: add spikes
                    if name == 'Qm' and spikes != 'none':
                        ax_legend_spikes = self.materializeSpikes(
                            ax, data, tplt['factor'], yplt['factor'],
                            color, spikes, add_to_legend=True)

                # Add legend
                if nvars > 1 or 'gate' in ax_pltvars[0]['desc'] or ax_legend_spikes:
                    ax.legend(fontsize=fs, loc=7, ncol=nvars // 4 + 1, frameon=False)

            if patches:
                for ax in axes:
                    self.addPatches(ax, tpatch_on, tpatch_off, tplt['factor'])

            # Post-process figure
            self.postProcess(axes, tplt, yplt, fs)
            if title:
                axes[0].set_title(figtitle(meta), fontsize=fs)
            fig.tight_layout()

            # Save figure if needed (automatic or checked)
            if save:
                filecode = model.filecode(meta)
                if directory is None:
                    directory = os.path.split(filepath)[0]
                plt_filename = '{}/{}.{}'.format(directory, filecode, fig_ext)
                plt.savefig(plt_filename)
                logger.info('Saving figure as "{}"'.format(plt_filename))
                plt.close()

            figs.append(fig)
        return figs


if __name__ == '__main__':
    # example of use
    filepaths = OpenFilesDialog('pkl')[0]
    comp_plot = ComparativePlot(filepaths, 'Qm')
    fig = comp_plot.render(
        lines=['-', '--'],
        labels=['60 kPa', '80 kPa'],
        patches='one',
        colors=['r', 'g'],
        blacklegend=False,
        straightlegend=False,
        xticks=[0, 100],
        yticks=[-80, +50],
        inset={'xcoords': [5, 40], 'ycoords': [-35, 45], 'xlims': [57.5, 60.5], 'ylims': [10, 35]}
    )

    scheme_plot = SchemePlot(filepaths)
    figs = scheme_plot.render()

    plt.show()
