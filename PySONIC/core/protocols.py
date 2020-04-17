# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-12 18:04:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-17 18:31:40

import numpy as np
from ..utils import si_format, StimObject
from .batches import Batch


class TimeProtocol(StimObject):

    def __init__(self, tstim, toffset):
        ''' Class constructor.

            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
        '''
        self.tstim = tstim
        self.toffset = toffset

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
    def ttotal(self):
        return self.tstim + self.toffset

    @property
    def nature(self):
        return 'CW'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.tstim == other.tstim and self.toffset == other.toffset

    def paramStr(self, k, strict_nfigs=False):
        val = getattr(self, k) * self.inputs()[k].get('factor', 1.)
        precision = self.inputs()[k].get('precision', 0)
        unit = self.inputs()[k].get('unit', '')
        formatted_val = si_format(val, precision=precision, space='')
        if strict_nfigs:
            minfigs = self.inputs()[k].get('minfigs', None)
            if minfigs is not None:
                nfigs = len(formatted_val.split('.')[0])
                if nfigs < minfigs:
                    formatted_val = '0' * (minfigs - nfigs) + formatted_val
        return f'{formatted_val}{unit}'

    def pdict(self, sf='{key}={value}', **kwargs):
        d = {k: sf.format(key=k, value=self.paramStr(k, **kwargs)) for k in self.inputs().keys()}
        if self.toffset == 0.:
            del d['toffset']
        return d

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(self.pdict().values())})'

    @property
    def desc(self):
        return ', '.join(self.pdict(sf='{value} {key}').values())

    @property
    def filecodes(self):
        return self.pdict(sf='{key}_{value}', strict_nfigs=True)

    @staticmethod
    def inputs():
        return {
            'tstim': {
                'desc': 'stimulus duration',
                'label': 't_{stim}',
                'unit': 'ms',
                'factor': 1e3,
                'precision': 0
            },
            'toffset': {
                'desc': 'offset duration',
                'label': 't_{offset}',
                'unit': 'ms',
                'factor': 1e3,
                'precision': 0
            }
        }

    def tOFFON(self):
        ''' Return vector of times of OFF-ON transitions (in s). '''
        return np.array([0.])

    def tONOFF(self):
        ''' Return vector of times of ON-OFF transitions (in s). '''
        return np.array([self.tstim])

    def stimEvents(self):
        ''' Return time-value pairs for each transition in stimulation state. '''
        t_on_off = self.tONOFF()
        t_off_on = self.tOFFON()
        pairs = list(zip(t_off_on, [1] * len(t_off_on))) + list(zip(t_on_off, [0] * len(t_on_off)))
        return sorted(pairs, key=lambda x: x[0])

    @classmethod
    def createQueue(cls, durations, offsets):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps.

            :param durations: list (or 1D-array) of stimulus durations
            :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
            :return: list of parameters (list) for each simulation
        '''
        return [cls(*item) for item in Batch.createQueue(durations, offsets)]


class PulsedProtocol(TimeProtocol):

    def __init__(self, tstim, toffset, PRF=100., DC=1.):
        ''' Class constructor.

            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
        '''
        super().__init__(tstim, toffset)
        self.DC = DC
        self.PRF = PRF

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

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return super().__eq__(other) and self.PRF == other.PRF and self.DC == other.DC

    def pdict(self, **kwargs):
        d = super().pdict(**kwargs)
        if self.isCW:
            del d['PRF']
            del d['DC']
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
        d = {
            'PRF': {
                'desc': 'pulse repetition frequency',
                'label': 'PRF',
                'unit': 'Hz',
                'factor': 1e0,
                'precision': 0
            },
            'DC': {
                'desc': 'duty cycle',
                'label': 'DC',
                'unit': '%',
                'factor': 1e2,
                'precision': 2,
                'minfigs': 2
            }
        }
        return {**TimeProtocol.inputs(), **d}

    def tOFFON(self):
        if self.isCW:
            return super().tOFFON()
        else:
            return np.arange(self.npulses) / self.PRF

    def tONOFF(self):
        if self.isCW:
            return super().tONOFF()
        else:
            return (np.arange(self.npulses) + self.DC) / self.PRF

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

    def __init__(self, tburst, BRF, nbursts=1, **kwargs):
        ''' Class constructor.

            :param tburst: burst duration (s)
            :param BRF: burst repetition frequency (Hz)
            :param nbursts: number of bursts
        '''
        super().__init__(tburst, 1 / BRF - tburst, **kwargs)
        self.BRF = BRF
        self.nbursts = nbursts

    @property
    def tburst(self):
        return self.tstim

    @property
    def ttotal(self):
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

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return super().__eq__(other) and self.BRF == other.BRF and self.nbursts == other.nbursts

    @staticmethod
    def inputs():
        d = PulsedProtocol.inputs()
        for k in ['tstim', 'toffset']:
            del d[k]
        d.update({
            'tburst': {
                'desc': 'burst duration',
                'label': 't_{burst}',
                'unit': 'ms',
                'factor': 1e3,
                'precision': 0
            },
            'BRF': {
                'desc': 'burst repetition frequency',
                'label': 'BRF',
                'unit': 'Hz',
                'precision': 0
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
        for nb in nbursts:
            for BRF in BRFs:
                queue.append(pp_queue + [BRF, nbursts])

        # Construct and return objects queue
        return [cls(*item) for item in queue]
