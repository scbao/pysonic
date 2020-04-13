# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-04 18:24:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-13 12:31:25

import abc
import pandas as pd
import csv
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from ..core import NeuronalBilayerSonophore, PulsedProtocol, AcousticDrive, LogBatch
from ..utils import logger, si_format, isIterable
from .pltutils import cm2inch, setNormalizer
from ..postpro import detectSpikes


class XYMap(LogBatch):
    ''' Generic 2D map object interface. '''

    def __init__(self, root, xvec, yvec):
        self.root = root
        self.xvec = xvec
        self.yvec = yvec
        super().__init__([list(pair) for pair in product(self.xvec, self.yvec)], root=root)

    def checkVector(self, name, value):
        if not isIterable(value):
            raise ValueError(f'{name} vector must be an iterable')
        if not isinstance(value, np.ndarray):
            value = np.asarray(value)
        if len(value.shape) > 1:
            raise ValueError(f'{name} vector must be one-dimensional')
        return value

    @property
    def in_key(self):
        return self.xkey

    @property
    def unit(self):
        return self.xunit

    @property
    def xvec(self):
        return self._xvec

    @xvec.setter
    def xvec(self, value):
        self._xvec = self.checkVector('x', value)

    @property
    def yvec(self):
        return self._yvec

    @yvec.setter
    def yvec(self, value):
        self._yvec = self.checkVector('x', value)

    @property
    @abc.abstractmethod
    def xkey(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def xfactor(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def xunit(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def ykey(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def yfactor(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def yunit(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def zkey(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def zunit(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def zfactor(self):
        raise NotImplementedError

    @property
    def out_keys(self):
        return [f'{self.zkey} ({self.zunit})']

    @property
    def in_labels(self):
        return [f'{self.xkey} ({self.xunit})', f'{self.ykey} ({self.yunit})']

    def getLogData(self):
        ''' Retrieve the batch log file data (inputs and outputs) as a dataframe. '''
        return pd.read_csv(self.fpath, sep=self.delimiter).sort_values(self.in_labels)

    def getInput(self):
        ''' Retrieve the logged batch inputs as an array. '''
        return self.getLogData()[self.in_labels].values

    def getOutput(self):
        return np.reshape(super().getOutput(), (self.xvec.size, self.yvec.size)).T

    def writeLabels(self):
        with open(self.fpath, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.delimiter)
            writer.writerow([*self.in_labels, *self.out_keys])

    def writeEntry(self, entry, output):
        with open(self.fpath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.delimiter)
            writer.writerow([*entry, output])

    def isEntry(self, comb):
        ''' Check if a given input is logged in the batch log file. '''
        inputs = self.getInput()
        if len(inputs) == 0:
            return False
        imatches_x = np.where(np.isclose(inputs[:, 0], comb[0], rtol=self.rtol, atol=self.atol))[0]
        imatches_y = np.where(np.isclose(inputs[:, 1], comb[1], rtol=self.rtol, atol=self.atol))[0]
        imatches = list(set(imatches_x).intersection(imatches_y))
        if len(imatches) == 0:
            return False
        return True

    @property
    def inputscode(self):
        ''' String describing the batch inputs. '''
        xcode = self.rangecode(self.xvec * self.xfactor, self.xkey, self.xunit)
        ycode = self.rangecode(self.yvec * self.yfactor, self.ykey, self.yunit)
        return '_'.join([xcode, ycode])

    @staticmethod
    def getScaleType(x):
        xmin, xmax, nx = x.min(), x.max(), x.size
        if np.all(np.isclose(x, np.logspace(np.log10(xmin), np.log10(xmax), nx))):
            return 'log'
        else:
            return 'lin'
        # elif np.all(np.isclose(x, np.linspace(xmin, xmax, nx))):
        #     return 'lin'
        # else:
        #     raise ValueError('Unknown distribution type')

    @property
    def xscale(self):
        return self.getScaleType(self.xvec)

    @property
    def yscale(self):
        return self.getScaleType(self.yvec)

    @staticmethod
    def computeMeshEdges(x, scale):
        ''' Compute the appropriate edges of a mesh that quads a linear or logarihtmic distribution.

            :param x: the input vector
            :param scale: the type of distribution ('lin' for linear, 'log' for logarihtmic)
            :return: the edges vector
        '''
        if scale == 'log':
            x = np.log10(x)
            range_func = np.logspace
        else:
            range_func = np.linspace
        dx = x[1] - x[0]
        n = x.size + 1
        return range_func(x[0] - dx / 2, x[-1] + dx / 2, n)

    @abc.abstractmethod
    def compute(self, x):
        ''' Compute the necessary output(s) for a given inputs combination. '''
        raise NotImplementedError

    def getOnClickXY(self, event):
        ''' Get x and y values from from x and y click event coordinates. '''
        x = self.xvec[np.searchsorted(self.xedges, event.xdata / self.xfactor) - 1]
        y = self.yvec[np.searchsorted(self.yedges, event.ydata / self.yfactor) - 1]
        return x, y

    def onClick(self, event):
        ''' Exexecute specific action when the user clicks on a cell in the 2D map. '''
        pass

    @property
    @abc.abstractmethod
    def title(self):
        raise NotImplementedError

    def getZBounds(self):
        matrix = self.getOutput()
        zmin, zmax = np.nanmin(matrix), np.nanmax(matrix)
        logger.info(
            f'{self.zkey} range: {zmin * self.zfactor:.0f} - {zmax * self.zfactor:.0f} {self.zunit}')
        return zmin, zmax

    def checkZbounds(self, zbounds):
        zmin, zmax = self.getZBounds()
        if zmin < zbounds[0]:
            logger.warning(
                f'Minimal {self.zkey} ({zmin:.0f} {self.zunit}) is below defined lower bound ({zbounds[0]:.0f} {self.zunit})')
        if zmax > zbounds[1]:
            logger.warning(
                f'Maximal {self.zkey} ({zmax:.0f} {self.zunit}) is above defined upper bound ({zbounds[1]:.0f} {self.zunit})')

    def render(self, xscale='lin', yscale='lin', zscale='lin', zbounds=None, fs=8, cmap='viridis',
               interactive=False, figsize=None):
        # Get figure size
        if figsize is None:
            figsize = cm2inch(12, 7)

        # Compute Z normalizer
        mymap = plt.get_cmap(cmap)
        mymap.set_bad('silver')
        if zbounds is None:
            zbounds = self.getZBounds()
        else:
            self.checkZbounds(zbounds)
        norm, sm = setNormalizer(mymap, zbounds, zscale)

        # Compute mesh edges
        self.xedges = self.computeMeshEdges(self.xvec, xscale)
        self.yedges = self.computeMeshEdges(self.yvec, yscale)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=0.15, bottom=0.15, right=0.8, top=0.92)
        ax.set_title(self.title, fontsize=fs)
        ax.set_xlabel(f'{self.xkey} ({self.xunit})', fontsize=fs, labelpad=-0.5)
        ax.set_ylabel(f'{self.ykey} ({self.yunit})', fontsize=fs)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fs)
        if xscale == 'log':
            ax.set_xscale('log')
        if yscale == 'log':
            ax.set_yscale('log')

        # Plot map with specific color code
        data = self.getOutput()
        ax.pcolormesh(self.xedges * self.xfactor, self.yedges * self.yfactor, data,
                      cmap=mymap, norm=norm)

        # Plot z-scale colorbar
        pos1 = ax.get_position()  # get the map axis position
        cbarax = fig.add_axes([pos1.x1 + 0.02, pos1.y0, 0.03, pos1.height])
        fig.colorbar(sm, cax=cbarax)
        cbarax.set_ylabel(f'{self.zkey} ({self.zunit})', fontsize=fs)
        for item in cbarax.get_yticklabels():
            item.set_fontsize(fs)

        if interactive:
            fig.canvas.mpl_connect('button_press_event', lambda event: self.onClick(event))

        return fig


class ActivationMap(XYMap):

    xkey = 'Duty cycle'
    xfactor = 1e2
    xunit = '%'
    ykey = 'Amplitude'
    yfactor = 1e-3
    yunit = 'kPa'
    zkey = 'Firing rate'
    zunit = 'Hz'
    zfactor = 1e0
    suffix = 'actmap'

    def __init__(self, root, pneuron, a, fs, f, tstim, PRF, amps, DCs):
        self.nbls = NeuronalBilayerSonophore(a, pneuron)
        self.drive = AcousticDrive(f, None)
        self.pp = PulsedProtocol(tstim, 0., PRF, 1.)
        self.fs = fs
        super().__init__(root, DCs, amps)

    @property
    def sim_args(self):
        return [self.drive, self.pp, self.fs, 'sonic', None]

    @property
    def title(self):
        s = '{} neuron @ {}Hz, {}Hz PRF ({}m sonophore'.format(
            self.nbls.pneuron.name, *si_format([self.drive.f, self.pp.PRF, self.nbls.a]))
        if self.fs < 1:
            s = f'{s}, {self.fs * 1e2:.0f}% coverage'
        return f'{s})'

    def corecode(self):
        corecodes = self.nbls.filecodes(*self.sim_args)
        corecodes['PRF'] = f'PRF{self.pp.PRF:.0f}Hz'
        del corecodes['nature']
        return '_'.join(filter(lambda x: x is not None, corecodes.values()))

    def compute(self, x):
        ''' Compute firing rate from simulation output '''
        # Adapt drive and pulsed protocol
        self.pp.DC = x[0]
        self.drive.A = x[1]

        # Get model output, running simulation if needed
        data, meta = self.nbls.getOutput(*self.sim_args, outputdir=self.root)

        # Detect spikes in data
        ispikes, _ = detectSpikes(data)

        # Compute firing metrics
        if ispikes.size > 1:
            t = data['t'].values
            sr = 1 / np.diff(t[ispikes])
            return np.mean(sr)
        else:
            return np.nan

    def addThresholdCurve(self, ax, fs):
        Athrs = []
        for DC in self.xvec:
            self.pp.DC = DC
            Athrs.append(self.nbls.titrate(*self.sim_args))
        Athrs = np.array(Athrs)
        ax.plot(self.xvec * self.xfactor, Athrs * self.yfactor, '-', color='#F26522', linewidth=3,
                label='threshold amplitudes')
        ax.legend(loc='lower center', frameon=False, fontsize=fs)

    def render(self, Ascale='log', FRscale='log', FRbounds=None, thresholds=False, **kwargs):
        kwargs['yscale'] = Ascale
        kwargs['zscale'] = FRscale
        kwargs['zbounds'] = FRbounds
        fig = super().render(**kwargs)
        if thresholds:
            self.addThresholdCurve(fig.axes[0], fs=8)
        return fig

    def onClick(self, event):
        ''' Retrieve the specific input parameters of the x and y dimensions
            when the user clicks on a cell in the 2D map, and define filename from it.
        '''
        # Define filepath
        DC, A = self.getOnClickXY(event)
        self.drive.A = A
        self.pp.DC = DC

        # Get model output, running simulation if needed
        data, meta = self.nbls.getOutput(*self.sim_args, outputdir=self.root)

        # Plot Q-trace
        self.plotQVeff(data, meta)
        plt.show()

    def plotQVeff(self, data, meta, tonset=10e-3, trange=None, ybounds=None, fs=8, lw=1):
        ''' Plot superimposed profiles of membrane charge density and
            effective membrane potential.

            :param data: simulation output dataframe
            :param tonset: pre-stimulus onset to add to profiles (s)
            :param trange: time lower and upper bounds on graph (s)
            :param ybounds: y-axis bounds (mV / nC/cm2)
            :return: handle to the generated figure
        '''
        # Bound time if needede
        if trange is not None:
            tmin, tmax = trange
            data = data.loc[(data['t'] >= tmin) & (data['t'] <= tmax)]

        # Load variables, add onset and rescale
        t, Qm, Vm = [data[k].values for k in ['t', 'Qm', 'Vm']]
        t = np.hstack((np.array([-tonset, t[0]]), t))  # s
        Vm = np.hstack((np.array([self.nbls.pneuron.Vm0] * 2), Vm))  # mV
        Qm = np.hstack((np.array([self.nbls.pneuron.Qm0] * 2), Qm))  # C/m2
        t *= 1e3  # ms
        Qm *= 1e5  # nC/cm2

        # Determine axes bounds
        if ybounds is None:
            ybounds = (min(Vm.min(), Qm.min()), max(Vm.max(), Qm.max()))

        # Create figure
        fig, ax = plt.subplots(figsize=cm2inch(12, 5))
        fig.canvas.set_window_title(self.nbls.desc(meta))
        plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.95)
        for key in ['top', 'right']:
            ax.spines[key].set_visible(False)
        for key in ['bottom', 'left']:
            ax.spines[key].set_position(('axes', -0.03))
            ax.spines[key].set_linewidth(2)
        ax.yaxis.set_tick_params(width=2)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xticks([])
        ax.set_xlabel(f'{si_format(np.ptp(t))}s', fontsize=fs)
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
