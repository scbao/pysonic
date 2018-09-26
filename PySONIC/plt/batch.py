import sys
import pickle
import ntpath
import numpy as np
import matplotlib.pyplot as plt

from ..utils import *
from ..core import BilayerSonophore
from .pltvars import pltvars
from ..neurons import getNeuronsDict


def plotBatch(filepaths, vars_dict=None, plt_save=False, directory=None,
              ask_before_save=True, fig_ext='png', tag='fig', fs=15, lw=2, title=True,
              show_patches=True):
    ''' Plot a figure with profiles of several specific NICE output variables, for several
        NICE simulations.

        :param filepaths: list of full paths to output data files to be compared
        :param vars_dict: dict of lists of variables names to extract and plot together
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
    figs = []
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
        npatches, tpatch_on, tpatch_off = getStimPulses(t, states)

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
            fig, ax = plt.subplots(figsize=(11, 4))
            axes = [ax]
        else:
            fig, axes = plt.subplots(naxes, 1, figsize=(11, min(3 * naxes, 9)))

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
                fig_title = ESTIM_title(
                    neuron.name, meta['Astim'], meta['tstim'] * 1e3, meta['PRF'], meta['DC'] * 1e2)
            elif sim_type == 'ASTIM':
                fig_title = ASTIM_title(
                    neuron.name, Fdrive * 1e-3, meta['Adrive'] * 1e-3, meta['tstim'] * 1e3,
                    meta['PRF'], meta['DC'] * 1e2)
            elif sim_type == 'MECH':
                fig_title = MECH_title(a * 1e9, Fdrive * 1e-3, meta['Adrive'] * 1e-3)

            axes[0].set_title(fig_title, fontsize=fs)

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
