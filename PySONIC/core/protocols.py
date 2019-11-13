# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-12 18:04:45
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-11-13 11:20:36

import numpy as np
from ..utils import si_format
from .batches import Batch


class TimeProtocol:

    def __init__(self, tstim, toffset):
        ''' Class constructor.

            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
        '''
        for k, v in {'stimulus': tstim, 'offset': toffset}.items():
            if not isinstance(v, float):
                raise TypeError(f'Invalid {k} duration value (must be float typed)')
            if v < 0:
                raise ValueError(f'Invalid {k} duration: {(v * 1e3)} ms (must be positive or null)')

        # Assing attributes
        self.tstim = tstim
        self.toffset = toffset

    def __repr__(self):
        params = [f'{si_format(x, 1, space="")}s' for x in [self.tstim, self.toffset]]
        return f'{self.__class__.__name__}({", ".join(params)})'

    def pprint(self):
        return f'{si_format(self.tstim, 1)}s stim, {si_format(self.toffset, 1)}s offset'

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


class PulsedProtocol(TimeProtocol):

    def __init__(self, tstim, toffset, PRF=100., DC=1.):
        ''' Class constructor.

            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
        '''
        super().__init__(tstim, toffset)
        if not isinstance(DC, float):
            raise TypeError('Invalid duty cycle value (must be float typed)')
        if DC <= 0.0 or DC > 1.0:
            raise ValueError(f'Invalid duty cycle: {DC} (must be within ]0; 1])')
        if DC < 1.0:
            if not isinstance(PRF, float):
                raise TypeError('Invalid PRF value (must be float typed)')
            if PRF is None:
                raise AttributeError('Missing PRF value (must be provided when DC < 1)')
            if PRF < 1 / tstim:
                raise ValueError(f'Invalid PRF: {PRF} Hz (PR interval exceeds stimulus duration)')

        # Assing attributes
        self.PRF = PRF
        self.DC = DC

        # Derived attributes
        self.T_ON = self.DC / self.PRF
        self.T_OFF = (1 - self.DC) / self.PRF
        self.npulses = int(np.round(self.tstim * self.PRF))

    def __repr__(self):
        params = [f'{si_format(self.PRF, 1, space="")}Hz', f'{self.DC:.2f}']
        return f'{super().__repr__()[:-1]}, {", ".join(params)})'

    def pprint(self):
        s = super().pprint()
        if self.DC < 1:
            s += f', {si_format(self.PRF, 2)}Hz PRF, {(self.DC * 1e2):.1f}% DC'
        return s

    def isCW(self):
        return self.DC == 1.

    def filecodes(self):
        if self.isCW():
            d = {'PRF': None, 'DC': None}
        else:
            d = {'PRF': 'PRF{:.2f}Hz'.format(self.PRF), 'DC': 'DC{:04.1f}%'.format(self.DC * 1e2)}
        return {**super().filecodes(), **d}

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


def createPulsedProtocols(durations, offsets, PRFs, DCs):
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
    queue = [PulsedProtocol(*item) for item in queue]
    return queue