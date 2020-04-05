# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-12 18:04:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-05 15:10:51

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

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.tstim == other.tstim and self.toffset == other.toffset

    def __repr__(self):
        params = [f'{si_format(x, 1, space="")}s' for x in [self.tstim, self.toffset]]
        return f'{self.__class__.__name__}({", ".join(params)})'

    @property
    def desc(self):
        return f'{si_format(self.tstim, 1)}s stim, {si_format(self.toffset, 1)}s offset'

    @property
    def filecodes(self):
        return {'tstim': f'{(self.tstim * 1e3):.0f}ms', 'toffset': None}

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

    @classmethod
    def createQueue(cls, durations, offsets):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps.

            :param durations: list (or 1D-array) of stimulus durations
            :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
            :return: list of parameters (list) for each simulation
        '''
        return [cls(*item) for item in Batch.createQueue(durations, offsets)]

    def eventsON(self):
        return np.array([0.])

    def eventsOFF(self):
        return np.array([self.tstim])


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

    def __repr__(self):
        params = [f'{si_format(self.PRF, 1, space="")}Hz', f'{self.DC:.2f}']
        return f'{super().__repr__()[:-1]}, {", ".join(params)})'

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
    def desc(self):
        s = super().desc
        if self.DC < 1:
            s += f', {si_format(self.PRF, 2)}Hz PRF, {(self.DC * 1e2):.1f}% DC'
        return s

    @property
    def isCW(self):
        return self.DC == 1.

    @property
    def nature(self):
        return 'CW' if self.isCW else 'PW'

    @property
    def filecodes(self):
        if self.isCW:
            d = {'PRF': None, 'DC': None}
        else:
            d = {'PRF': f'PRF{self.PRF:.2f}Hz', 'DC': f'DC{self.DC * 1e2:04.1f}%'}
        return {**super().filecodes, **d}

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
                'precision': 2
            }
        }
        return {**TimeProtocol.inputs(), **d}

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

    def eventsON(self):
        if self.DC == 1:
            return super().eventsON()
        else:
            return np.arange(self.npulses) / self.PRF

    def eventsOFF(self):
        if self.DC == 1:
            return super().eventsOFF()
        else:
            return self.eventsON() + self.DC / self.PRF
