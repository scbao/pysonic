# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-25 16:19:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-26 21:07:19

import numpy as np
import matplotlib.pyplot as plt

from ..utils import *
from .pltutils import *


def plotBatch(filepaths, pltscheme=None, plt_save=False, directory=None,
              ask_before_save=True, fig_ext='png', tag='fig', fs=10, lw=2, title=True,
              show_patches=True, frequency=1):
    ''' Plot a figure with profiles of several specific NICE output variables, for several
        NICE simulations.

        :param filepaths: list of full paths to output data files to be compared
        :param pltscheme: dict of lists of variables names to extract and plot together
        :param plt_save: boolean stating whether to save the created figures
        :param directory: directory where to save figures
        :param ask_before_save: boolean stating whether to show the created figures
        :param fig_ext: file extension for the saved figures
        :param tag: suffix added to the end of the figures name
        :param fs: labels font size
        :param lw: curves line width
        :param title: boolean stating whether to display a general title on the figures
        :param show_patches: boolean indicating whether to indicate periods of stimulation with
         colored rectangular patches
        :param frequency: downsampling frequency for time series
        :return: list of figure handles
    '''

    figs = []

    # Loop through data files
    for filepath in filepaths:

        # Retrieve file code and sim type from file name
        pkl_filename = os.path.basename(filepath)
        filecode = pkl_filename[0:-4]
        sim_type = getSimType(pkl_filename)

        # Load data and extract variables
        df, meta = loadData(filepath, frequency)
        t = df['t'].values
        try:
            stimstate = df['stimstate'].values
        except KeyError:
            stimstate = df['states'].values

        # Determine stimulus patch from stimstate
        _, tpatch_on, tpatch_off = getStimPulses(t, stimstate)

        # Initialize appropriate object
        obj = getObject(sim_type, meta)

        # Retrieve plot variables
        tvar, pltvars = getTimePltVar(obj.tscale), obj.getPltVars()

        # Check plot scheme if provided, otherwise generate it
        if pltscheme:
            for key in list(sum(list(pltscheme.values()), [])):
                if key not in pltvars:
                    raise KeyError('Unknown plot variable: "{}"'.format(key))
        else:
            pltscheme = obj.getPltScheme()

        # Preset and rescale time vector
        if tvar['onset'] > 0.0:
            tonset = np.array([-tvar['onset'], -t[0] - t[1]])
            t = np.hstack((tonset, t))
        t *= tvar['factor']

        # Create figure
        naxes = len(pltscheme)
        if naxes == 1:
            fig, ax = plt.subplots(figsize=(11, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(naxes, 1, figsize=(11, min(3 * naxes, 9)))

        # Loop through each subgraph
        for ax, (grouplabel, keys) in zip(axes, pltscheme.items()):

            # Extract variables to plot
            nvars = len(keys)
            ax_pltvars = [pltvars[k] for k in keys]
            if nvars == 1:
                ax_pltvars[0]['color'] = 'k'
                ax_pltvars[0]['ls'] = '-'

            # Set y-axis unit and bounds
            ax.set_ylabel('$\\rm {}\ ({})$'.format(grouplabel, ax_pltvars[0].get('unit', '')),
                          fontsize=fs)
            if 'bounds' in ax_pltvars[0]:
                ax_min = min([ap['bounds'][0] for ap in ax_pltvars])
                ax_max = max([ap['bounds'][1] for ap in ax_pltvars])
                ax.set_ylim(ax_min, ax_max)

            # Plot time series
            icolor = 0
            for pltvar, name in zip(ax_pltvars, pltscheme[grouplabel]):
                var = extractPltVar(obj, pltvar, df, meta, t.size, name)
                ax.plot(t, var, pltvar.get('ls', '-'), c=pltvar.get('color', 'C{}'.format(icolor)),
                        lw=lw, label='$\\rm {}$'.format(pltvar['label']))
                if 'color' not in pltvar:
                    icolor += 1

            # Add legend
            if nvars > 1 or 'gate' in ax_pltvars[0]['desc']:
                ax.legend(fontsize=fs, loc=7, ncol=nvars // 4 + 1, frameon=False)

        # Post-process figure
        for ax in axes:
            for item in ['top', 'right']:
                ax.spines[item].set_visible(False)
            ax.locator_params(axis='y', nbins=2)
            for item in ax.get_yticklabels():
                item.set_fontsize(fs)
        for ax in axes[:-1]:
            ax.set_xticklabels([])
        for item in axes[-1].get_xticklabels():
            item.set_fontsize(fs)
        axes[-1].set_xlabel('$\\rm {}\ ({})$'.format(tvar['label'], tvar['unit']), fontsize=fs)
        if show_patches == 1:
            for ax in axes:
                plotStimPatches(ax, tpatch_on, tpatch_off, tvar['factor'])
        if title:
            axes[0].set_title(figtitle(meta), fontsize=fs)
        fig.tight_layout()

        # Save figure if needed (automatic or checked)
        if plt_save:
            if directory is None:
                directory = os.path.split(filepath)[0]
            if ask_before_save:
                plt_filename = SaveFileDialog(
                    '{}_{}.{}'.format(filecode, tag, fig_ext),
                    dirname=directory, ext=fig_ext)
            else:
                plt_filename = '{}/{}_{}.{}'.format(directory, filecode, tag, fig_ext)
            if plt_filename:
                plt.savefig(plt_filename)
                logger.info('Saving figure as "{}"'.format(plt_filename))
                plt.close()

        figs.append(fig)
    return figs
