# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2018-09-25 16:18:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-22 15:08:01

import numpy as np
import matplotlib.pyplot as plt

from ..postpro import detectSpikes, convertPeaksProperties
from ..utils import *
from .pltutils import *


class TimeSeriesPlot(GenericPlot):
    ''' Generic interface to build a plot displaying temporal profiles of model simulations. '''

    @classmethod
    def setTimeLabel(cls, ax, tplt, fs):
        return super().setXLabel(ax, tplt, fs)

    @classmethod
    def setYLabel(cls, ax, yplt, fs, grouplabel=None):
        if grouplabel is not None:
            yplt['label'] = grouplabel
        return super().setYLabel(ax, yplt, fs)

    def checkInputs(self, *args, **kwargs):
        return NotImplementedError

    @staticmethod
    def getStimStates(df):
        try:
            stimstate = df['stimstate']
        except KeyError:
            stimstate = df['states']
        return stimstate.values

    @classmethod
    def getStimPulses(cls, t, states):
        ''' Determine the onset and offset times of pulses from a stimulation vector.

            :param t: time vector (s).
            :param states: a vector of stimulation state (ON/OFF) at each instant in time.
            :return: 3-tuple with number of patches, timing of STIM-ON an STIM-OFF instants.
        '''
        # Compute states derivatives and identify bounds indexes of pulses
        dstates = np.diff(states)
        ipulse_on = np.where(dstates > 0.0)[0] + 1
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

    @staticmethod
    def addLegend(ax, handles, labels, fs, color=None, ls=None):
        lh = ax.legend(handles, labels, loc=1, fontsize=fs, frameon=False)
        if color is not None:
            for l in lh.get_lines():
                l.set_color(color)
        if ls:
            for l in lh.get_lines():
                l.set_linestyle(ls)

    @classmethod
    def materializeSpikes(cls, ax, data, tplt, yplt, color, mode, add_to_legend=False):
        t = data['t'].values
        Qm = data['Qm'].values
        ispikes, properties = detectSpikes(data)
        ileft = properties['left_bases']
        iright = properties['right_bases']
        properties = convertPeaksProperties(t, properties)
        if ispikes is not None:
            yoffset = 5
            ax.plot(t[ispikes] * tplt['factor'], Qm[ispikes] * yplt['factor'] + yoffset,
                    'v', color=color, label='spikes' if add_to_legend else None)
            if mode == 'details':
                ax.plot(t[ileft] * tplt['factor'], Qm[ileft] * yplt['factor'] - 5,
                        '<', color=color, label='left-bases' if add_to_legend else None)
                ax.plot(t[iright] * tplt['factor'], Qm[iright] * yplt['factor'] - 10,
                        '>', color=color, label='right-bases' if add_to_legend else None)
                ax.vlines(
                    x=t[ispikes] * tplt['factor'],
                    ymin=(Qm[ispikes] - properties['prominences']) * yplt['factor'],
                    ymax=Qm[ispikes] * yplt['factor'],
                    color=color, linestyles='dashed',
                    label='prominences' if add_to_legend else '')
                ax.hlines(
                    y=properties['width_heights'] * yplt['factor'],
                    xmin=properties['left_ips'] * tplt['factor'],
                    xmax=properties['right_ips'] * tplt['factor'],
                    color=color, linestyles='dotted', label='half-widths' if add_to_legend else '')
        return add_to_legend

    @staticmethod
    def prepareTime(t, tplt):
        if tplt['onset'] > 0.0:
            t = np.insert(t, 0, -tplt['onset'])
        return t * tplt['factor']

    @staticmethod
    def addPatches(ax, tpatch_on, tpatch_off, tplt, color='#8A8A8A'):
        for i in range(tpatch_on.size):
            ax.axvspan(tpatch_on[i] * tplt['factor'], tpatch_off[i] * tplt['factor'],
                       edgecolor='none', facecolor=color, alpha=0.2)

    @staticmethod
    def plotInset(inset_ax, t, y, tplt, yplt, line, color, lw):
        inset_window = np.logical_and(t > (inset['xlims'][0] / tplt['factor']),
                                      t < (inset['xlims'][1] / tplt['factor']))
        inset_ax.plot(t[inset_window] * tplt['factor'], y[inset_window] * yplt['factor'],
                      linewidth=lw, linestyle=line, color=color)
        return inset_ax

    @staticmethod
    def addInsetPatches(ax, inset_ax, inset, tpatch_on, tpatch_off, tplt, color):
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


class CompTimeSeries(ComparativePlot, TimeSeriesPlot):
    ''' Interface to build a comparative plot displaying profiles of a specific output variable
        across different model simulations. '''

    def __init__(self, filepaths, varname):
        ''' Constructor.

            :param filepaths: list of full paths to output data files to be compared
            :param varname: name of variable to extract and compare
        '''
        ComparativePlot.__init__(self, filepaths, varname)

    def checkPatches(self, patches):
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
        return patches, greypatch

    def checkInputs(self, lines, labels, colors, patches):
        self.checkLabels(labels)
        lines = self.checkLines(lines)
        colors = self.checkColors(colors)
        patches, greypatch = self.checkPatches(patches)
        return lines, labels, colors, patches, greypatch

    @staticmethod
    def createBackBone(figsize):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_zorder(0)
        return fig, ax

    @classmethod
    def postProcess(cls, ax, tplt, yplt, fs, meta, prettify):
        cls.removeSpines(ax)
        if 'bounds' in yplt:
            ax.set_ylim(*yplt['bounds'])
        cls.setTimeLabel(ax, tplt, fs)
        cls.setYLabel(ax, yplt, fs)
        if prettify:
            cls.prettify(ax, xticks=(0, meta['tstim'] * tplt['factor']))
        cls.setTickLabelsFontSize(ax, fs)

    def render(self, figsize=(11, 4), fs=10, lw=2, labels=None, colors=None, lines=None,
               patches='one', inset=None, frequency=1, spikes='none', cmap=None,
               cscale='lin', trange=None, prettify=False):
        ''' Render plot.

            :param figsize: figure size (x, y)
            :param fs: labels fontsize
            :param lw: linewidth
            :param labels: list of labels to use in the legend
            :param colors: list of colors to use for each curve
            :param lines: list of linestyles
            :param patches: string indicating whether/how to mark stimulation periods
                with rectangular patches
            :param inset: string indicating whether/how to mark an inset zooming on
                a particular region of the graph
            :param frequency: frequency at which to plot samples
            :param spikes: string indicating how to show spikes ("none", "marks" or "details")
            :param cmap: color map to use for colobar-based comparison (if not None)
            :param cscale: color scale to use for colobar-based comparison
            :param trange: optional lower and upper bounds to time axis
            :return: figure handle
        '''
        lines, labels, colors, patches, greypatch = self.checkInputs(
            lines, labels, colors, patches)

        fig, ax = self.createBackBone(figsize)
        if inset is not None:
            inset_ax = self.addInset(fig, ax, inset)

        # Loop through data files
        handles, comp_values, full_labels = [], [], []
        tmin, tmax = np.inf, -np.inf
        for j, filepath in enumerate(self.filepaths):

            # Load data
            try:
                data, meta = self.getData(filepath, frequency, trange)
            except ValueError as err:
                continue
            if 'tcomp' in meta:
                meta.pop('tcomp')
            full_labels.append(self.figtitle(meta))

            # Extract model
            model = self.getModel(meta)

            # Check consistency of sim types and check differing inputs
            comp_values = self.checkConsistency(meta, comp_values)

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
            handles.append(ax.plot(t, y, linewidth=lw, linestyle=lines[j], color=colors[j])[0])

            # Optional: add spikes
            if self.varname == 'Qm' and spikes != 'none':
                self.materializeSpikes(ax, data, tplt, yplt, colors[j], spikes)

            # Plot optional inset
            if inset is not None:
                inset_ax = self.plotInset(inset_ax, t, y, tplt, yplt, lines[j], colors[j], lw)

            # Add optional STIM-ON patches
            if patches[j]:
                ybottom, ytop = ax.get_ylim()
                color = '#8A8A8A' if greypatch else handles[j].get_color()
                self.addPatches(ax, tpatch_on, tpatch_off, tplt, color)
                if inset is not None:
                    self.addInsetPatches(ax, inset_ax, inset, tpatch_on, tpatch_off, tplt, color)

            tmin, tmax = min(tmin, t.min()), max(tmax, t.max())

        # Determine labels
        if self.comp_ref_key is not None:
            self.comp_info = model.inputs().get(self.comp_ref_key, None)
        comp_values, comp_labels = self.getCompLabels(comp_values)
        labels = self.chooseLabels(labels, comp_labels, full_labels)

        # Post-process figure
        self.postProcess(ax, tplt, yplt, fs, meta, prettify)
        ax.set_xlim(tmin, tmax)
        fig.tight_layout()

        if inset is not None:
            self.materializeInset(ax, inset_ax, inset)

        # Add labels or colorbar legend
        if cmap is not None:
            if not self.is_unique_comp:
                raise ValueError('Colormap mode unavailable for multiple differing parameters')
            if self.comp_info is None:
                raise ValueError('Colormap mode unavailable for qualitative comparisons')
            self.addCmap(
                fig, cmap, handles, comp_values, self.comp_info, fs, prettify, zscale=cscale)
        else:
            self.addLegend(ax, handles, labels, fs)

        return fig


class GroupedTimeSeries(TimeSeriesPlot):
    ''' Interface to build a plot displaying profiles of several output variables
        arranged into specific schemes. '''

    def __init__(self, filepaths, pltscheme=None):
        ''' Constructor.

            :param filepaths: list of full paths to output data files to be compared
            :param varname: name of variable to extract and compare
        '''
        super().__init__(filepaths)
        self.pltscheme = pltscheme

    @staticmethod
    def createBackBone(pltscheme):
        naxes = len(pltscheme)
        if naxes == 1:
            fig, ax = plt.subplots(figsize=(11, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(naxes, 1, figsize=(11, min(3 * naxes, 9)))
        return fig, axes

    @classmethod
    def postProcess(cls, axes, tplt, fs, meta, prettify):
        for ax in axes:
            cls.removeSpines(ax)
            # if prettify:
            #     cls.prettify(ax, xticks=(0, meta['tstim'] * tplt['factor']), yfmt=None)
            cls.setTickLabelsFontSize(ax, fs)
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        cls.setTimeLabel(axes[-1], tplt, fs)

    def render(self, fs=10, lw=2, labels=None, colors=None, lines=None, patches='one', save=False,
               outputdir=None, fig_ext='png', frequency=1, spikes='none', trange=None,
               prettify=False):
        ''' Render plot.

            :param fs: labels fontsize
            :param lw: linewidth
            :param labels: list of labels to use in the legend
            :param colors: list of colors to use for each curve
            :param lines: list of linestyles
            :param patches: boolean indicating whether to mark stimulation periods
                with rectangular patches
            :param save: boolean indicating whether or not to save the figure(s)
            :param outputdir: path to output directory in which to save figure(s)
            :param fig_ext: string indcating figure extension ("png", "pdf", ...)
            :param frequency: frequency at which to plot samples
            :param spikes: string indicating how to show spikes ("none", "marks" or "details")
            :param trange: optional lower and upper bounds to time axis
            :return: figure handle(s)
        '''

        figs = []
        for filepath in self.filepaths:

            # Load data and extract model
            try:
                data, meta = self.getData(filepath, frequency, trange)
            except ValueError as err:
                continue
            model = self.getModel(meta)

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
                self.setYLabel(ax, ax_pltvars[0].copy(), fs, grouplabel=grouplabel)
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
                            ax, data, tplt, yplt, color, spikes, add_to_legend=True)

                # Add legend
                if nvars > 1 or 'gate' in ax_pltvars[0]['desc'] or ax_legend_spikes:
                    ax.legend(fontsize=fs, loc=7, ncol=nvars // 4 + 1, frameon=False)

            # Set x-limits and add optional patches
            for ax in axes:
                ax.set_xlim(t.min(), t.max())
                if patches != 'none':
                    self.addPatches(ax, tpatch_on, tpatch_off, tplt)

            # Post-process figure
            self.postProcess(axes, tplt, fs, meta, prettify)
            axes[0].set_title(self.figtitle(meta), fontsize=fs)
            fig.tight_layout()

            fig.canvas.set_window_title(model.filecode(meta))

            # Save figure if needed (automatic or checked)
            if save:
                filecode = model.filecode(meta)
                if outputdir is None:
                    outputdir = os.path.split(filepath)[0]
                plt_filename = '{}/{}.{}'.format(outputdir, filecode, fig_ext)
                plt.savefig(plt_filename)
                logger.info('Saving figure as "{}"'.format(plt_filename))
                plt.close()

            figs.append(fig)
        return figs


if __name__ == '__main__':
    # example of use
    filepaths = OpenFilesDialog('pkl')[0]
    comp_plot = CompTimeSeries(filepaths, 'Qm')
    fig = comp_plot.render(
        lines=['-', '--'],
        labels=['60 kPa', '80 kPa'],
        patches='one',
        colors=['r', 'g'],
        xticks=[0, 100],
        yticks=[-80, +50],
        inset={'xcoords': [5, 40], 'ycoords': [-35, 45], 'xlims': [57.5, 60.5], 'ylims': [10, 35]}
    )

    scheme_plot = GroupedTimeSeries(filepaths)
    figs = scheme_plot.render()

    plt.show()
