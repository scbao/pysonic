# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-12 18:04:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-12-11 18:25:22

import abc
import numpy as np
import matplotlib.pyplot as plt
import itertools
from ..utils import isIterable
from .stimobj import StimObject
from .batches import Batch


class TimeProtocol(StimObject):

    @property
    @abc.abstractmethod
    def nature(self):
        raise NotImplementedError

    @abc.abstractmethod
    def stimEvents(self):
        ''' Return time-value pairs for each transition in stimulation state. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def tstop(self):
        ''' Stopping time. '''
        raise NotImplementedError

    def stimProfile(self):
        events = self.stimEvents()
        l = [(0., 0)]
        for e in events:
            l.append((e[0], l[-1][1]))
            l.append(e)
        t, x = zip(*l)
        return np.array(t), np.array(x)

    def plot(self):
        t, x = self.stimProfile()
        fig, ax = plt.subplots()
        ax.set_title('stim vector')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('amplitude')
        ax.plot(t * 1e3, x)
        return fig


class CustomProtocol(TimeProtocol):

    def __init__(self, tevents, xevents, tstop):
        ''' Class constructor.

            :param tevents: vector of time events occurences (s)
            :param xevents: vector of stimulus modulation values
            :param tstop: stopping time (s)
        '''
        self.tevents = tevents
        self.xevents = xevents
        self.tstop = tstop

    @property
    def nature(self):
        return 'custom'

    @property
    def tevents(self):
        return self._tevents

    @tevents.setter
    def tevents(self, value):
        if not isIterable(value):
            value = [value]
        value = np.asarray([self.checkFloat('tevents', v) for v in value])
        self.checkPositiveOrNull('tevents', value.min())
        self._tevents = value

    @property
    def xevents(self):
        return self._xevents

    @xevents.setter
    def xevents(self, value):
        if not isIterable(value):
            value = [value]
        value = np.asarray([self.checkFloat('xevents', v) for v in value])
        self._xevents = value

    @property
    def tstop(self):
        return self._tstop

    @tstop.setter
    def tstop(self, value):
        value = self.checkFloat('tstop', value)
        self.checkBounded('tstop', value, (self.tevents.max(), np.inf))
        self._tstop = value

    def copy(self):
        return self.__class__(self.tevents, self.xevents, self.tstop)

    @staticmethod
    def inputs():
        return {
            'tevents': {
                'desc': 'events times',
                'label': 't_{events}',
                'unit': 's',
                'factor': 1e0,
                'precision': 2
            },
            'xevents': {
                'desc': 'events modulation factors',
                'label': 'x_{events}',
                'precision': 1
            },
            'tstop': {
                'desc': 'stopping time',
                'label': 't_{stop}',
                'unit': 's',
                'factor': 1e0,
                'precision': 0
            }
        }

    def stimEvents(self):
        return sorted(list(zip(self.tevents, self.xevents)), key=lambda x: x[0])


class PulsedProtocol(TimeProtocol):

    def __init__(self, tstim, toffset, PRF=100., DC=1., tstart=0.):
        ''' Class constructor.

            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
        '''
        self.tstim = tstim
        self.toffset = toffset
        self.DC = DC
        self.PRF = PRF
        self.tstart = tstart

    @property
    def tstim(self):
        return self._tstim

    @tstim.setter
    def tstim(self, value):
        value = self.checkFloat('tstim', value)
        self.checkPositiveOrNull('tstim', value)
        self._tstim = value

    @property
    def toffset(self):
        return self._toffset

    @toffset.setter
    def toffset(self, value):
        value = self.checkFloat('toffset', value)
        self.checkPositiveOrNull('toffset', value)
        self._toffset = value

    @property
    def DC(self):
        return self._DC

    @DC.setter
    def DC(self, value):
        value = self.checkFloat('DC', value)
        self.checkBounded('DC', value, (0., 1.))
        self._DC = value

    @property
    def PRF(self):
        return self._PRF

    @PRF.setter
    def PRF(self, value):
        value = self.checkFloat('PRF', value)
        self.checkPositiveOrNull('PRF', value)
        if self.DC < 1.:
            self.checkBounded('PRF', value, (1 / self.tstim, np.inf))
        self._PRF = value

    @property
    def tstart(self):
        return self._tstart

    @tstart.setter
    def tstart(self, value):
        value = self.checkFloat('tstart', value)
        self.checkPositiveOrNull('tstart', value)
        self._tstart = value

    def copy(self):
        return self.__class__(
            self.tstim, self.toffset, PRF=self.PRF, DC=self.DC, tstart=self.tstart)

    @property
    def tstop(self):
        return self.tstim + self.toffset

    def pdict(self, **kwargs):
        d = super().pdict(**kwargs)
        if 'toffset' in d and self.toffset == 0.:
            del d['toffset']
        if self.isCW:
            del d['PRF']
            del d['DC']
        if self.tstart == 0.:
            del d['tstart']
        return d

    @property
    def T_ON(self):
        return self.DC / self.PRF

    @property
    def T_OFF(self):
        return (1 - self.DC) / self.PRF

    @property
    def npulses(self):
        return int(np.round(self.tstim * self.PRF))

    @property
    def isCW(self):
        return self.DC == 1.

    @property
    def nature(self):
        return 'CW' if self.isCW else 'PW'

    @staticmethod
    def inputs():
        return {
            'tstim': {
                'desc': 'stimulus duration',
                'label': 't_{stim}',
                'unit': 's',
                'factor': 1e0,
                'precision': 0
            },
            'toffset': {
                'desc': 'offset duration',
                'label': 't_{offset}',
                'unit': 's',
                'factor': 1e0,
                'precision': 0
            },
            'PRF': {
                'desc': 'pulse repetition frequency',
                'label': 'PRF',
                'unit': 'Hz',
                'factor': 1e0,
                'precision': 2
            },
            'DC': {
                'desc': 'duty cycle',
                'label': 'DC',
                'unit': '%',
                'factor': 1e2,
                'precision': 1,
                'minfigs': 2
            },
            'tstart': {
                'desc': 'stimulus start time',
                'label': 't_{start}',
                'unit': 's',
                'precision': 0
            },
        }

    def tOFFON(self):
        ''' Return vector of times of OFF-ON transitions (in s). '''
        if self.isCW:
            return np.array([self.tstart])
        else:
            return np.arange(self.npulses) / self.PRF + self.tstart

    def tONOFF(self):
        ''' Return vector of times of ON-OFF transitions (in s). '''
        if self.isCW:
            return np.array([self.tstart + self.tstim])
        else:
            return (np.arange(self.npulses) + self.DC) / self.PRF + self.tstart

    def stimEvents(self):
        t_on_off = self.tONOFF()
        t_off_on = self.tOFFON()
        pairs = list(zip(t_off_on, [1] * len(t_off_on))) + list(zip(t_on_off, [0] * len(t_on_off)))
        return sorted(pairs, key=lambda x: x[0])

    @classmethod
    def createQueue(cls, durations, offsets, PRFs, DCs):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps, while avoiding repetition of CW protocols for a given PRF sweep.

            :param durations: list (or 1D-array) of stimulus durations
            :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
            :param PRFs: list (or 1D-array) of pulse-repetition frequencies
            :param DCs: list (or 1D-array) of duty cycle values
            :return: list of parameters (list) for each simulation
        '''
        DCs = np.array(DCs)
        queue = []
        if 1.0 in DCs:
            queue += Batch.createQueue(durations, offsets, min(PRFs), 1.0)
        if np.any(DCs != 1.0):
            queue += Batch.createQueue(durations, offsets, PRFs, DCs[DCs != 1.0])
        queue = [cls(*item) for item in queue]
        return queue


class BurstProtocol(PulsedProtocol):

    def __init__(self, tburst, PRF=100., DC=1., BRF=None, nbursts=1, tstart=0.):
        ''' Class constructor.

            :param tburst: burst duration (s)
            :param BRF: burst repetition frequency (Hz)
            :param nbursts: number of bursts
        '''
        if BRF is None:
            BRF = 1 / (2 * tburst)
        self.checkBounded('BRF', BRF, (0, 1 / tburst))  # BRF pre-check
        super().__init__(tburst, 1 / BRF - tburst, PRF=PRF, DC=DC, tstart=tstart)
        self.BRF = BRF
        self.nbursts = nbursts

    def copy(self):
        return self.__class__(
            self.tburst, PRF=self.PRF, DC=self.DC, BRF=self.BRF, nbursts=self.nbursts)

    @property
    def tburst(self):
        return self.tstim

    @property
    def tstop(self):
        return self.nbursts / self.BRF

    @property
    def BRF(self):
        return self._BRF

    @BRF.setter
    def BRF(self, value):
        value = self.checkFloat('BRF', value)
        self.checkPositiveOrNull('BRF', value)
        self.checkBounded('BRF', value, (0, 1 / self.tburst))
        self._BRF = value

    @staticmethod
    def inputs():
        d = PulsedProtocol.inputs()
        for k in ['tstim', 'toffset']:
            del d[k]
        d.update({
            'tburst': {
                'desc': 'burst duration',
                'label': 't_{burst}',
                'unit': 's',
                'factor': 1e0,
                'precision': 0
            },
            'BRF': {
                'desc': 'burst repetition frequency',
                'label': 'BRF',
                'unit': 'Hz',
                'precision': 1
            },
            'nbursts': {
                'desc': 'number of bursts',
                'label': 'n_{bursts}'
            }
        })
        return {
            'tburst': d.pop('tburst'),
            **{k: v for k, v in d.items()}
        }

    def repeatBurstArray(self, tburst):
        return np.ravel(np.array([tburst + i / self.BRF for i in range(self.nbursts)]))

    def tOFFON(self):
        return self.repeatBurstArray(super().tOFFON())

    def tONOFF(self):
        return self.repeatBurstArray(super().tONOFF())

    @classmethod
    def createQueue(cls, durations, PRFs, DCs, BRFs, nbursts):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps, while avoiding repetition of CW protocols for a given PRF sweep.

            :param durations: list (or 1D-array) of stimulus durations
            :param PRFs: list (or 1D-array) of pulse-repetition frequencies
            :param DCs: list (or 1D-array) of duty cycle values
            :param BRFs: list (or 1D-array) of burst-repetition frequencies
            :param nbursts: list (or 1D-array) of number of bursts
            :return: list of parameters (list) for each simulation
        '''
        # Get pulsed protocols queue (with unique zero offset)
        pp_queue = PulsedProtocol.createQueue(durations, [0.], PRFs, DCs)

        # Extract parameters (without offset)
        pp_queue = [[x.tstim, x.PRF, x.DC] for x in pp_queue]

        # Complete queue with each BRF-nburts combination
        queue = []
        for item in pp_queue:
            for nb in nbursts:
                for BRF in BRFs:
                    queue.append(item + [BRF, nb])

        # Construct and return objects queue
        return [cls(*item) for item in queue]


class BalancedPulsedProtocol(PulsedProtocol):

    def __init__(self, tpulse, xratio, toffset, tstim=None, PRF=100, tstart=0.):
        self.tpulse = tpulse
        self.xratio = xratio
        if tstim is None:
            tstim = self.ttotal
            PRF = 1 / tstim
        super().__init__(tstim, toffset, PRF=PRF, DC=self.tpulse * PRF, tstart=tstart)

    @property
    def tpulse(self):
        return self._tpulse

    @tpulse.setter
    def tpulse(self, value):
        value = self.checkFloat('tpulse', value)
        self.checkPositiveOrNull('tpulse', value)
        self._tpulse = value

    @property
    def xratio(self):
        return self._xratio

    @xratio.setter
    def xratio(self, value):
        value = self.checkFloat('xratio', value)
        self.checkBounded('xratio', value, (0., 1.))
        self._xratio = value

    @property
    def PRF(self):
        return self._PRF

    @PRF.setter
    def PRF(self, value):
        value = self.checkFloat('PRF', value)
        self.checkPositiveOrNull('PRF', value)
        if self.tstim != self.ttotal:
            self.checkBounded('PRF', value, (1 / self.tstim, 1 / self.ttotal))
        self._PRF = value

    @property
    def treversal(self):
        return self.tpulse / self.xratio

    @property
    def ttotal(self):
        return self.tpulse + self.treversal

    def copy(self):
        return self.__class__(
            self.tpulse, self.xratio, self.toffset, tstim=self.tstim, PRF=self.PRF)

    @staticmethod
    def inputs():
        d = PulsedProtocol.inputs()
        del d['DC']
        return {
            'tpulse': {
                'desc': 'pulse width',
                'label': 't_{pulse}',
                'unit': 's',
                'factor': 1e0,
                'precision': 2
            },
            'xratio': {
                'desc': 'balance amplitude factor',
                'label': 'x_{ratio}',
                'factor': 1e2,
                'unit': '%',
                'precision': 1
            },
            **d
        }

    def tRev(self):
        return self.tOFFON() + self.tpulse

    def tONOFF(self):
        return self.tOFFON() + self.ttotal

    def stimEvents(self):
        pairs = list(itertools.chain.from_iterable([
            list(zip(t, [x] * len(t))) for t, x in [
                (self.tOFFON(), 1), (self.tRev(), -self.xratio), (self.tONOFF(), 0)
            ]
        ]))
        return sorted(pairs, key=lambda x: x[0])


def getPulseTrainProtocol(PD, npulses, PRF):
    ''' Get a pulse train protocol for a given pulse duration, number of pulses and PRF.

        :param PD: pulse duration (s)
        :param npulses: number of pulses
        :param PRF: pulse repetition frequency (Hz)
        :return: PulsedProtocol object
    '''
    DC = PD * PRF
    tstim = npulses / PRF
    tstart = 1 / PRF - PD
    return PulsedProtocol(tstim + tstart, 0., PRF=PRF, DC=DC, tstart=tstart)
