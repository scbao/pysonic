# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:24:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-29 20:20:21

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from ..core import NeuronalBilayerSonophore
from ..utils import logger, si_format, loadData
from ..postpro import findPeaks
from ..constants import *
from .pltutils import cm2inch, setNormalizer


class Map:

    @staticmethod
    def computeMeshEdges(x, scale):
        ''' Compute the appropriate edges of a mesh that quads a linear or logarihtmic distribution.

            :param x: the input vector
            :param scale: the type of distribution ('lin' for linear, 'log' for logarihtmic)
            :return: the edges vector
        '''
        if scale == 'log':
            x = np.log10(x)
        dx = x[1] - x[0]
        n = x.size + 1
        return {'lin': np.linspace, 'log': np.logspace}[scale](x[0] - dx / 2, x[-1] + dx / 2, n)


class ActivationMap(Map):

    def __init__(self, root, pneuron, a, Fdrive, tstim, PRF, amps, DCs):
        self.root = root
        self.pneuron = pneuron
        self.a = a
        self.nbls = NeuronalBilayerSonophore(self.a, self.pneuron)
        self.Fdrive = Fdrive
        self.tstim = tstim
        self.PRF = PRF
        self.amps = amps
        self.DCs = DCs

        self.Ascale = self.getAmpScale()
        self.title = '{} neuron @ {}Hz, {}Hz PRF ({}m sonophore)'.format(
            self.pneuron.name, *si_format([self.Fdrive, self.PRF, self.a]))
        out_fname = 'actmap {} {}Hz PRF{}Hz {}s.csv'.format(
            self.pneuron.name, *si_format([self.Fdrive, self.PRF, self.tstim], space=''))
        out_fpath = os.path.join(self.root, out_fname)

        if os.path.isfile(out_fpath):
            self.data = np.loadtxt(out_fpath, delimiter=',')
        else:
            self.data = self.compute()
            np.savetxt(out_fpath, self.data, delimiter=',')

    def getAmpScale(self):
        Amin, Amax, nA = self.amps.min(), self.amps.max(), self.amps.size
        if np.all(np.isclose(self.amps, np.logspace(np.log10(Amin), np.log10(Amax), nA))):
            return 'log'
        elif np.all(np.isclose(self.amps, np.linspace(Amin, Amax, nA))):
            return 'lin'
        else:
            raise ValueError('Unknown distribution type')

    def classify(self, df):
        ''' Classify based on charge temporal profile. '''

        t = df['t'].values
        Qm = df['Qm'].values

        # Detect spikes on charge profile during stimulus
        dt = t[1] - t[0]
        mpd = int(np.ceil(SPIKE_MIN_DT / dt))
        ispikes, *_ = findPeaks(
            Qm[t <= self.tstim],
            mph=SPIKE_MIN_QAMP,
            mpd=mpd,
            mpp=SPIKE_MIN_QPROM
        )

        # Compute firing metrics
        if ispikes.size == 0:  # if no spike, assign -1
            return -1
        elif ispikes.size == 1:  # if only 1 spike, assign 0
            return 0
        else:  # if more than 1 spike, assign firing rate
            FRs = 1 / np.diff(t[ispikes])
            return np.mean(FRs)

    @staticmethod
    def correctAmp(A):
        return np.round(A * 1e-3, 1) * 1e3

    def compute(self):
        logger.info('Generating activation map for %s neuron', self.pneuron.name)
        actmap = np.empty((self.amps.size, self.DCs.size))
        nfiles = self.DCs.size * self.amps.size
        for i, A in enumerate(self.amps):
            for j, DC in enumerate(self.DCs):
                fname = '{}.pkl'.format(self.nbls.filecode(
                    self.Fdrive, self.correctAmp(A), self.tstim, 0., self.PRF, DC, 'sonic'))
                fpath = os.path.join(self.root, fname)
                if not os.path.isfile(fpath):
                    print(fpath)
                    logger.error('"{}" file not found'.format(fname))
                    actmap[i, j] = np.nan
                else:
                    # Load data
                    logger.debug('Loading file {}/{}: "{}"'.format(
                        i * self.amps.size + j + 1, nfiles, fname))
                    df, _ = loadData(fpath)
                    actmap[i, j] = self.classify(df)
        return actmap

    @staticmethod
    def adjustFRbounds(actmap):
        ''' Check firing rate bounding. '''
        minFR, maxFR = (actmap[actmap > 0].min(), actmap.max())
        logger.info('FR range: %.0f - %.0f Hz', minFR, maxFR)
        if FRbounds is None:
            FRbounds = (minFR, maxFR)
        else:
            if minFR < FRbounds[0]:
                logger.warning(
                    'Minimal firing rate (%.0f Hz) is below defined lower bound (%.0f Hz)',
                    minFR, FRbounds[0])
            if maxFR > FRbounds[1]:
                logger.warning(
                    'Maximal firing rate (%.0f Hz) is above defined upper bound (%.0f Hz)',
                    maxFR, FRbounds[1])

    @staticmethod
    def fit2Colormap(actmap, cmap):
        actmap[actmap == -1] = np.nan
        actmap[actmap == 0] = 1e-3
        cmap.set_bad('silver')
        cmap.set_under('k')
        return actmap, cmap

    def addThresholdCurve(self, ax):
        Athrs_fname = 'Athrs_{}_{:.0f}nm_{}Hz_PRF{}Hz_{}s.xlsx'.format(
            self.pneuron.name, self.a * 1e9,
            *si_format([self.Fdrive, self.PRF, self.tstim], 0, space=''))
        fpath = os.path.join(self.root, Athrs_fname)
        if os.path.isfile(fpath):
            df = pd.read_excel(fpath, sheet_name='Data')
            DCs = df['Duty factor'].values
            Athrs = df['Adrive (kPa)'].values
            iDCs = np.argsort(DCs)
            DCs = DCs[iDCs]
            Athrs = Athrs[iDCs]
            ax.plot(DCs * 1e2, Athrs, '-', color='#F26522', linewidth=2,
                    label='threshold amplitudes')
            ax.legend(loc='lower center', frameon=False, fontsize=8)
        else:
            logger.warning('%s file not found -> cannot draw threshold curve', fpath)

    def render(self, Ascale='log', FRscale='log', FRbounds=None, fs=8, cmap='viridis',
               interactive=False, Vbounds=None, trange=None, thresholds=False):

        # Compute FR normalizer
        mymap = plt.get_cmap(cmap)
        norm, sm = setNormalizer(mymap, FRbounds, FRscale)

        # Compute mesh edges
        xedges = self.computeMeshEdges(self.DCs, 'lin')
        yedges = self.computeMeshEdges(self.amps, self.Ascale)

        # Create figure
        fig, ax = plt.subplots(figsize=cm2inch(8, 5.8))
        fig.subplots_adjust(left=0.15, bottom=0.15, right=0.8, top=0.92)
        ax.set_title(self.title, fontsize=fs)
        ax.set_xlabel('Duty cycle (%)', fontsize=fs, labelpad=-0.5)
        ax.set_ylabel('Amplitude (kPa)', fontsize=fs)
        ax.set_xlim(np.array([self.DCs.min(), self.DCs.max()]) * 1e2)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)

        # Plot activation map with specific color code
        actmap, cmap = self.fit2Colormap(self.data, plt.get_cmap(cmap))
        if Ascale == 'log':
            ax.set_yscale('log')
        ax.pcolormesh(xedges * 1e2, yedges * 1e-3, actmap, cmap=mymap, norm=norm)
        if thresholds:
            self.addThresholdCurve(ax)

        # Plot firing rate colorbar
        pos1 = ax.get_position()  # get the map axis position
        cbarax = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.03, pos1.height])
        fig.colorbar(sm, cax=cbarax)
        cbarax.set_ylabel('Firing rate (Hz)', fontsize=fs)
        for item in cbarax.get_yticklabels():
            item.set_fontsize(fs)

        if interactive:
            fig.canvas.mpl_connect(
                'button_press_event',
                lambda event: self.onClick(event, (xedges, yedges), trange, Vbounds))

        return fig

    def onClick(self, event, meshedges, trange, Vbounds):
        ''' Retrieve the specific input parameters of the x and y dimensions
            when the user clicks on a cell in the 2D map, and define filename from it.
        '''

        # Get DC and A from x and y coordinates
        x, y = event.xdata, event.ydata
        DC = self.DCs[np.searchsorted(meshedges[0], x * 1e-2) - 1]
        Adrive = self.amps[np.searchsorted(meshedges[1], y * 1e3) - 1]

        # Define filepath
        fname = '{}.pkl'.format(self.nbls.filecode(
            self.Fdrive, self.correctAmp(Adrive), self.tstim, 0., self.PRF, DC, 'sonic'))
        fpath = os.path.join(self.root, fname)

        # Plot Q-trace
        try:
            self.plotQVeff(fpath, trange=trange, ybounds=Vbounds)
            plt.show()
        except FileNotFoundError as err:
            logger.error(err)

    def plotQVeff(self, filepath, tonset=10e-3, trange=None, ybounds=None, fs=8, lw=1):
        ''' Plot superimposed profiles of membrane charge density and
            effective membrane potential.

            :param filepath: full path to the data file
            :param tonset: pre-stimulus onset to add to profiles (s)
            :param trange: time lower and upper bounds on graph (s)
            :param ybounds: y-axis bounds (mV / nC/cm2)
            :return: handle to the generated figure
        '''
        # Check file existence
        fname = os.path.basename(filepath)
        if not os.path.isfile(filepath):
            raise FileNotFoundError('Error: "{}" file does not exist'.format(fname))

        # Load data (with optional time restriction)
        logger.debug('Loading data from "%s"', fname)
        with open(filepath, 'rb') as fh:
            frame = pickle.load(fh)
            df = frame['data']
        if trange is not None:
            tmin, tmax = trange
            df = df.loc[(df['t'] >= tmin) & (df['t'] <= tmax)]

        # Load variables, add onset and rescale
        t, Qm, Vm = [df[k].values for k in ['t', 'Qm', 'Vm']]
        t = np.hstack((np.array([-tonset, t[0]]), t))  # s
        Vm = np.hstack((np.array([self.pneuron.Vm0] * 2), Vm))  # mV
        Qm = np.hstack((np.array([self.pneuron.Qm0()] * 2), Qm))  # C/m2
        t *= 1e3  # ms
        Qm *= 1e5  # nC/cm2

        # Determine axes bounds
        if ybounds is None:
            ybounds = (min(Vm.min(), Qm.min()), max(Vm.max(), Qm.max()))

        # Create figure
        fig, ax = plt.subplots(figsize=cm2inch(7, 3))
        fig.canvas.set_window_title(fname)
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
        for key in ['top', 'right']:
            ax.spines[key].set_visible(False)
        for key in ['bottom', 'left']:
            ax.spines[key].set_position(('axes', -0.03))
            ax.spines[key].set_linewidth(2)
        ax.yaxis.set_tick_params(width=2)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xticks([])
        ax.set_xlabel('{}s'.format(si_format(np.ptp(t), space=' ')), fontsize=fs)
        ax.set_ylabel('mV - $\\rm nC/cm^2$', fontsize=fs, labelpad=-15)
        ax.set_ylim(ybounds)
        ax.set_yticks(ybounds)
        for item in ax.get_yticklabels():
            item.set_fontsize(fs)

        # Plot Qm and Vmeff profiles
        ax.plot(t, Vm, color='darkgrey', linewidth=lw)
        ax.plot(t, Qm, color='k', linewidth=lw)

        # fig.tight_layout()
        return fig
