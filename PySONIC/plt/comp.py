import sys
import pickle
import ntpath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter

from .pltutils import *
from ..utils import rescale, getStimPulses
from .pltvars import pltvars
from ..core import BilayerSonophore
from ..neurons import getNeuronsDict


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


def plotComp(filepaths, varname, labels=None, fs=15, lw=2, colors=None, lines=None, patches='one',
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
        npatches, tpatch_on, tpatch_off = getStimPulses(t, states)

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
                label = ESTIM_title(
                    neuron.name, meta['Astim'], meta['tstim'] * 1e3, meta['PRF'], meta['DC'] * 1e2)
            elif sim_type == 'ASTIM':
                label = ASTIM_title(
                    neuron.name, meta['Fdrive'] * 1e-3, meta['Adrive'] * 1e-3, meta['tstim'] * 1e3,
                    meta['PRF'], meta['DC'] * 1e2)
            elif sim_type == 'MECH':
                label = MECH_title(a * 1e9, meta['Fdrive'] * 1e-3, meta['Adrive'] * 1e-3)

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

    return fig