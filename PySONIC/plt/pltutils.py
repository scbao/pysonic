# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-21 14:33:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-24 22:02:53

''' Useful functions to generate plots. '''

import re
import numpy as np
import pandas as pd
from scipy.signal import peak_widths
import matplotlib
from matplotlib.patches import Rectangle
from matplotlib import cm, colors
import matplotlib.pyplot as plt

from ..core import getModel
from ..utils import logger, isIterable, loadData, rescale, swapFirstLetterCase
from ..constants import SPIKE_MIN_DT, SPIKE_MIN_QAMP, SPIKE_MIN_QPROM

# Matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'


def figtitle(meta):
    ''' Return appropriate title based on simulation metadata. '''
    if 'Cm0' in meta:
        return '{:.0f}nm radius BLS structure: MECH-STIM {:.0f}kHz, {:.2f}kPa, {:.1f}nC/cm2'.format(
            meta['a'] * 1e9, meta['Fdrive'] * 1e-3, meta['Adrive'] * 1e-3, meta['Qm'] * 1e5)
    else:
        if 'DC' in meta:
            if meta['DC'] < 1:
                wavetype = 'PW'
                suffix = ', {:.2f}Hz PRF, {:.0f}% DC'.format(meta['PRF'], meta['DC'] * 1e2)
            else:
                wavetype = 'CW'
                suffix = ''
            if 'Astim' in meta:
                return '{} neuron: {} E-STIM {:.2f}mA/m2, {:.0f}ms{}'.format(
                    meta['neuron'], wavetype, meta['Astim'], meta['tstim'] * 1e3, suffix)
            else:
                return '{} neuron ({:.1f}nm): {} A-STIM {:.0f}kHz {:.2f}kPa, {:.0f}ms{} - {} model'.format(
                    meta['neuron'], meta['a'] * 1e9, wavetype, meta['Fdrive'] * 1e-3,
                    meta['Adrive'] * 1e-3, meta['tstim'] * 1e3, suffix, meta['method'])
        else:
            return '{} neuron: V-CLAMP {:.1f}-{:.1f}mV, {:.0f}ms'.format(
                    meta['neuron'], meta['Vhold'], meta['Vstep'], meta['tstim'] * 1e3)



def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def extractPltVar(model, pltvar, df, meta=None, nsamples=0, name=''):
    if 'func' in pltvar:
        s = 'model.{}'.format(pltvar['func'])
        try:
            var = eval(s)
        except AttributeError:
            var = eval(s.replace('model', 'model.pneuron'))
    elif 'key' in pltvar:
        var = df[pltvar['key']]
    elif 'constant' in pltvar:
        var = eval(pltvar['constant']) * np.ones(nsamples)
    else:
        var = df[name]
    if isinstance(var, pd.Series):
        var = var.values
    var = var.copy()

    if var.size == nsamples - 1:
        var = np.insert(var, 0, var[0])
    var *= pltvar.get('factor', 1)

    return var


def setGrid(n, ncolmax=3):
    ''' Determine number of rows and columns in figure grid, based on number of
        variables to plot. '''
    if n <= ncolmax:
        return (1, n)
    else:
        return ((n - 1) // ncolmax + 1, ncolmax)


def setNormalizer(cmap, bounds, scale='lin'):
    norm = {
        'lin': colors.Normalize,
        'log': colors.LogNorm
    }[scale](*bounds)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = []
    return norm, sm


class GenericPlot:
    def __init__(self, filepaths):
        ''' Constructor.

            :param filepaths: list of full paths to output data files to be compared
        '''
        if not isIterable(filepaths):
            filepaths = [filepaths]
        self.filepaths = filepaths

    def __call__(self, *args, **kwargs):
        return self.render(*args, **kwargs)

    @staticmethod
    def getData(entry, frequency=1, trange=None):
        if entry is None:
            raise ValueError('non-existing data')
        if isinstance(entry, str):
            data, meta = loadData(entry, frequency)
        else:
            data, meta = entry
        data = data.iloc[::frequency]
        if trange is not None:
            tmin, tmax = trange
            data = data.loc[(data['t'] >= tmin) & (data['t'] <= tmax)]
        return data, meta

    def render(self, *args, **kwargs):
        return NotImplementedError

    @staticmethod
    def getSimType(fname):
        ''' Get sim type from filename. '''
        mo = re.search('(^[A-Z]*)_(.*).pkl', fname)
        if not mo:
            raise ValueError('Could not find sim-key in filename: "{}"'.format(fname))
        return mo.group(1)

    @staticmethod
    def getModel(*args, **kwargs):
        return getModel(*args, **kwargs)

    @staticmethod
    def figtitle(*args, **kwargs):
        return figtitle(*args, **kwargs)

    @staticmethod
    def getTimePltVar(tscale):
        ''' Return time plot variable for a given temporal scale. '''
        return {
            'desc': 'time',
            'label': 'time',
            'unit': tscale,
            'factor': {'ms': 1e3, 'us': 1e6}[tscale],
            'onset': {'ms': 1e-3, 'us': 1e-6}[tscale]
        }

    @staticmethod
    def createBackBone(*args, **kwargs):
        return NotImplementedError

    @staticmethod
    def prettify(ax, xticks=None, yticks=None, xfmt='{:.0f}', yfmt='{:+.0f}'):
        try:
            ticks = ax.get_ticks()
            ticks = (min(ticks), max(ticks))
            ax.set_ticks(ticks)
            ax.set_ticklabels([xfmt.format(x) for x in ticks])
        except AttributeError:
            if xticks is None:
                xticks = ax.get_xticks()
                xticks = (min(xticks), max(xticks))
            if yticks is None:
                yticks = ax.get_yticks()
                yticks = (min(yticks), max(yticks))
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            if xfmt is not None:
                ax.set_xticklabels([xfmt.format(x) for x in xticks])
            if yfmt is not None:
                ax.set_yticklabels([yfmt.format(y) for y in yticks])

    @staticmethod
    def addInset(fig, ax, inset):
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

    @staticmethod
    def materializeInset(ax, inset_ax, inset):
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

    def postProcess(self, *args, **kwargs):
        return NotImplementedError

    @staticmethod
    def removeSpines(ax):
        for item in ['top', 'right']:
            ax.spines[item].set_visible(False)

    @staticmethod
    def setXTicks(ax, xticks=None):
        if xticks is not None:
            ax.set_xticks(xticks)

    @staticmethod
    def setYTicks(ax, yticks=None):
        if yticks is not None:
            ax.set_yticks(yticks)

    @staticmethod
    def setTickLabelsFontSize(ax, fs):
        for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fs)

    @staticmethod
    def setXLabel(ax, xplt, fs):
        ax.set_xlabel('$\\rm {}\ ({})$'.format(xplt['label'], xplt['unit']), fontsize=fs)

    @staticmethod
    def setYLabel(ax, yplt, fs):
        ax.set_ylabel('$\\rm {}\ ({})$'.format(yplt['label'], yplt.get('unit', '')), fontsize=fs)

    @classmethod
    def addCmap(cls, fig, cmap, handles, comp_values, comp_info, fs, prettify, zscale='lin'):
        # Create colormap and normalizer
        try:
            mymap = plt.get_cmap(cmap)
        except ValueError:
            mymap = plt.get_cmap(swapFirstLetterCase(cmap))
        norm, sm = setNormalizer(mymap, (comp_values.min(), comp_values.max()), zscale)

        # Adjust line colors
        for lh, z in zip(handles, comp_values):
            if isIterable(lh):
                for item in lh:
                    item.set_color(sm.to_rgba(z))
            else:
                lh.set_color(sm.to_rgba(z))

        # Add colorbar
        fig.subplots_adjust(left=0.1, right=0.8, bottom=0.15, top=0.95, hspace=0.5)
        cbarax = fig.add_axes([0.85, 0.15, 0.03, 0.8])
        cbar = fig.colorbar(sm, cax=cbarax, orientation='vertical')
        cbarax.set_ylabel('$\\rm {}\ ({})$'.format(
            comp_info['desc'].replace(' ', '\ '), comp_info['unit']), fontsize=fs)
        if prettify:
            cls.prettify(cbar)
        for item in cbarax.get_yticklabels():
            item.set_fontsize(fs)


class ComparativePlot(GenericPlot):

    def __init__(self, filepaths, varname):
        ''' Constructor.

            :param filepaths: list of full paths to output data files to be compared
            :param varname: name of variable to extract and compare
        '''
        super().__init__(filepaths)
        self.varname = varname
        self.comp_ref_key = None
        self.meta_ref = None
        self.comp_info = None
        self.is_unique_comp = False

    def checkColors(self, colors):
        if colors is None:
            colors = ['C{}'.format(j) for j in range(len(self.filepaths))]
        return colors

    def checkLines(self, lines):
        if lines is None:
            lines = ['-'] * len(self.filepaths)
        return lines

    def checkLabels(self, labels):
        if labels is not None:
            if len(labels) != len(self.filepaths):
                raise ValueError(
                    'Invalid labels ({}): not matching number of compared files ({})'.format(
                        len(labels), len(self.filepaths)))
            if not all(isinstance(x, str) for x in labels):
                raise TypeError('Invalid labels: must be string typed')

    def checkSimType(self, meta):
        ''' Check consistency of sim types across files. '''
        if meta['simkey'] != self.meta_ref['simkey']:
            raise ValueError('Invalid comparison: different simulation types')

    def checkCompValues(self, meta, comp_values):
        ''' Check consistency of differing values across files. '''
        differing = {k: meta[k] != self.meta_ref[k] for k in meta.keys()}
        if sum(differing.values()) > 1:
            logger.warning('More than one differing inputs')
            self.comp_ref_key = None
            return []
        zkey = (list(differing.keys())[list(differing.values()).index(True)])
        if self.comp_ref_key is None:
            self.comp_ref_key = zkey
            self.is_unique_comp = True
            comp_values.append(self.meta_ref[self.comp_ref_key])
            comp_values.append(meta[self.comp_ref_key])
        else:
            if zkey != self.comp_ref_key:
                logger.warning('inconsitent differing inputs')
                self.comp_ref_key = None
                return []
            else:
                comp_values.append(meta[self.comp_ref_key])
        return comp_values

    def checkConsistency(self, meta, comp_values):
        ''' Check consistency of sim types and check differing inputs. '''
        if self.meta_ref is None:
            self.meta_ref = meta
        else:
            self.checkSimType(meta)
            comp_values = self.checkCompValues(meta, comp_values)
            if self.comp_ref_key is None:
                self.is_unique_comp = False
        return comp_values

    def getCompLabels(self, comp_values):
        if self.comp_info is not None:
            comp_values = np.array(comp_values) * self.comp_info.get('factor', 1)
            comp_labels = [
                '$\\rm{} = {}\ {}$'.format(self.comp_info['label'], x, self.comp_info['unit'])
                for x in comp_values]
        else:
            comp_labels = comp_values
        return comp_values, comp_labels

    def chooseLabels(self, labels, comp_labels, full_labels):
        if labels is not None:
            return labels
        else:
            if self.is_unique_comp:
                return comp_labels
            else:
                return full_labels

    @staticmethod
    def getCommonLabel(lbls, sep=' ', merge_keys=None):
        if merge_keys is not None:
            for k in merge_keys:
                lbls = [lbl.replace(f'{sep}{k}', f'-{k}') for lbl in lbls]
        splt_lbls = [lbl.split(sep) for lbl in lbls]
        ncomps = len(splt_lbls[0])
        splt_lbls = np.array(splt_lbls).T
        all_indentical = [np.all(x == x[0]) for x in splt_lbls]
        if np.sum(all_indentical) < ncomps - 1:
            logger.warning('More than one differing inputs')
            return ''
        common_lbl = sep.join(splt_lbls[all_indentical, 0])
        if merge_keys is not None:
            for k in merge_keys:
                common_lbl = common_lbl.replace(f'-{k}', f'{sep}{k}')
        return common_lbl
