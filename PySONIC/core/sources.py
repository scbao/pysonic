# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-30 11:46:47
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-01 20:05:30

import numpy as np

from ..utils import si_format
from ..constants import NPC_DENSE, NPC_SPARSE
from .batches import Batch


class AcousticSource:

    def __init__(self, Fdrive, Adrive, phi=np.pi):
        ''' Constructor.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param phi: acoustic drive phase (rad)
        '''

        # Check inputs
        self.checkInputs(Fdrive, Adrive, phi)

        # Assign attributes
        self.Fdrive = Fdrive
        self.Adrive = Adrive
        self.phi = phi

    def __repr__(self):
        params = [f'{si_format(self.Fdrive, 1, space="")}Hz']
        if self.Adrive is not None:
            params.append(f'{si_format(self.Adrive, 1, space="")}Pa')
        return f'{self.__class__.__name__}({", ".join(params)})'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.Fdrive == other.Fdrive and self.Adrive == other.Adrive and self.phi == other.phi

    def copy(self):
        return self.__class__(self.Fdrive, self.Adrive, phi=self.phi)

    def checkInputs(self, Fdrive, Adrive, phi):
        ''' Check inputs. '''
        floatvars = {'Fdrive': Fdrive, 'phi': phi}
        if Adrive is not None:
            floatvars['Adrive'] = Adrive
        for k, v in floatvars.items():
            if not isinstance(v, float):
                raise TypeError(f'Invalid {k} parameter (must be float typed)')
        if Fdrive <= 0:
            d = self.inputs['Fdrive']
            raise ValueError(f'Invalid {d["desc"]}: {Fdrive * d.get("factor", 1)} {d["unit"]} (must be strictly positive)')
        if Adrive is not None and Adrive < 0:
            d = self.inputs['Adrive']
            raise ValueError(f'Invalid {d["desc"]}: {Adrive * d.get("factor", 1)} {d["unit"]} (must be positive or null)')

    @property
    @staticmethod
    def inputs():
        return {
            'Fdrive': {
                'desc': 'US drive frequency',
                'label': 'f',
                'unit': 'kHz',
                'factor': 1e-3,
                'precision': 0
            },
            'Adrive': {
                'desc': 'US pressure amplitude',
                'label': 'A',
                'unit': 'kPa',
                'factor': 1e-3,
                'precision': 2
            },
            'phi': {
                'desc': 'US drive phase',
                'label': '\Phi',
                'unit': 'rad',
                'precision': 2
            }
        }

    @property
    def meta(self):
        return {
            'Fdrive': self.Fdrive,
            'Adrive': self.Adrive
        }

    @property
    def desc(self):
        return 'f = {}Hz, A = {}Pa'.format(*si_format([self.Fdrive, self.Adrive], 2))

    @property
    def filecodes(self):
        return {
            'Fdrive': f'{self.Fdrive * 1e-3:.0f}kHz',
            'Adrive': f'{self.Adrive * 1e-3:.2f}kPa'
        }

    @property
    def dt(self):
        ''' Determine integration time step. '''
        return 1 / (NPC_DENSE * self.Fdrive)

    @property
    def dt_sparse(self):
        return 1 / (NPC_SPARSE * self.Fdrive)

    @property
    def periodicity(self):
        ''' Determine source periodicity. '''
        return 1. / self.Fdrive

    @property
    def nPerCycle(self):
        return NPC_DENSE

    @property
    def modulationFrequency(self):
        return self.Fdrive

    def computePressure(self, t):
        ''' Compute the acoustic acoustic pressure at a specific time.

            :param t: time (s)
            :return: acoustic pressure (Pa)
        '''
        return self.Adrive * np.sin(2 * np.pi * self.Fdrive * t - self.phi)


class AcousticSourceArray:

    def __init__(self, sources):
        self.sources = {f'source {i + 1}': s for i, s in enumerate(sources)}

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.nsources != other.nsources:
            return False
        if list(self.sources.keys()) != list(other.sources.keys()):
            return False
        for k, v in self.sources.items():
            if other.sources[k] != v:
                return False
        return True

    def copy(self):
        return self.__class__([x.copy() for x in self.sources.values()])

    @property
    def nsources(self):
        return len(self.sources)

    @property
    def meta(self):
        return {k: s.meta for k, s in self.sources.items()}

    @property
    def desc(self):
        descs = [f'[{s.desc}]' for k, s in self.sources.items()]
        return ', '.join(descs)

    @property
    def filecodes(self):
        return {k: s.filecodes for k, s in self.sources.items()}

    @property
    def fmax(self):
        return max(s.Fdrive for s in self.sources.values())

    @property
    def fmin(self):
        return min(s.Fdrive for s in self.sources.values())

    @property
    def dt(self):
        return 1 / (NPC_DENSE * self.fmax)

    @property
    def dt_sparse(self):
        return 1 / (NPC_SPARSE * self.fmax)

    @property
    def periodicity(self):
        ''' Determine source periodicity. '''
        if self.nsources > 2:
            raise ValueError('cannot compute periodicity for more than two sources')
        return 1 / (self.fmax - self.fmin)

    @property
    def nPerCycle(self):
        return int(self.periodicity // self.dt)

    @property
    def modulationFrequency(self):
        return np.mean([s.Fdrive for s in self.sources.values()])

    def computePressure(self, t):
        return sum(s.computePressure(t) for s in self.sources.values())


def createSources(freqs, amps):
    ''' Create a list of acoustic source objects for combinations of acoustic frequencies and amplitudes.

        :param freqs: list (or 1D-array) of US frequencies
        :param amps: list (or 1D-array) of US amplitudes
        :return: list of AcousticSource objects for each simulation
    '''
    queue = Batch.createQueue(freqs, amps)
    queue = [AcousticSource(*item) for item in queue]
    return queue