# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-01-30 11:46:47
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-03 22:59:11

import abc
import numpy as np

from ..utils import si_format
from ..constants import NPC_DENSE, NPC_SPARSE
from .batches import Batch


class Drive(metaclass=abc.ABCMeta):
    ''' Generic interface to drive object. '''

    @abc.abstractmethod
    def __repr__(self):
        ''' String representation. '''
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other):
        ''' Equality operator. '''
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        ''' String representation. '''
        raise NotImplementedError

    @abc.abstractmethod
    def checkInputs(self, *args):
        ''' Check inputs. '''
        raise NotImplementedError

    @property
    @staticmethod
    @abc.abstractmethod
    def inputs():
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def meta(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def desc(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def filecodes(self):
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self, t):
        ''' Compute the input drive at a specific time.

            :param t: time (s)
            :return: specific input drive
        '''
        raise NotImplementedError

    @classmethod
    def createQueue(cls, *args):
        ''' Create a list of Drive objects for combinations of input parameters. '''
        if len(args) == 1:
            return [cls(item) for item in args[0]]
        else:
            return [cls(*item) for item in Batch.createQueue(*args)]

    @property
    def isResolved(self):
        return True


class XDrive(Drive):
    ''' Drive object that can be titrated to find the threshold value of one of its inputs. '''

    @property
    def xvar(self):
        raise NotImplementedError

    @xvar.setter
    def xvar(self, value):
        raise NotImplementedError

    def updatedX(self, value):
        other = self.copy()
        other.xvar = value
        return other

    @property
    def is_resolved(self):
        return self.xvar is not None



class ElectricDrive(XDrive):
    ''' Electric drive object with constant amplitude. '''

    xkey = 'A'

    def __init__(self, A):
        ''' Constructor.

            :param A: amplitude (mA/m2)
        '''
        self.checkInputs(A)
        self.A = A

    @property
    def xvar(self):
        return self.A

    @xvar.setter
    def xvar(self, value):
        self.checkInputs(value)
        self.A = value

    def __repr__(self):
        params = []
        if self.A is not None:
            params.append(f'{si_format(self.A * 1e-3, 1, space="")}A/m2')
        return f'{self.__class__.__name__}({", ".join(params)})'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.A == other.A

    def copy(self):
        return self.__class__(self.A)

    def checkInputs(self, A):
        if A is not None:
            if not isinstance(A, float):
                raise TypeError('Invalid amplitude (must be float typed)')

    @staticmethod
    def inputs():
        return {
            'A': {
                'desc': 'current density amplitude',
                'label': 'A',
                'unit': 'mA/m2',
                'factor': 1e0,
                'precision': 1
            }
        }

    @property
    def meta(self):
        return {
            'A': self.A
        }

    @property
    def desc(self):
        return f'A = {si_format(self.A * 1e-3, 2)}A/m2'

    @property
    def filecodes(self):
        return {'A': f'{self.A:.2f}mAm2'}

    def compute(self, t):
        return self.A


class VoltageDrive(Drive):
    ''' Voltage drive object with a held potential and a step potential. '''

    def __init__(self, Vhold, Vstep):
        ''' Constructor.

            :param Vhold: held membrane potential (mV)
            :param Vstep: step membrane potential (mV)
        '''
        self.checkInputs(Vhold, Vstep)
        self.Vhold = Vhold
        self.Vstep = Vstep

    def __repr__(self):
        return f'{self.__class__.__name__}({self.desc})'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.Vhold == other.Vhold and self.Vstep == other.Vstep

    def copy(self):
        return self.__class__(self.Vhold, self.Vstep)

    def checkInputs(self, Vhold, Vstep):
        for k, v in {'Vhold': Vhold, 'Vstep': Vstep}.items():
            if not isinstance(v, float):
                raise TypeError(f'Invalid {k} parameter (must be float typed)')

    @staticmethod
    def inputs():
        return {
            'Vhold': {
                'desc': 'held membrane potential',
                'label': 'V_{hold}',
                'unit': 'mV',
                'precision': 0
            },
            'Vstep': {
                'desc': 'step membrane potential',
                'label': 'V_{step}',
                'unit': 'mV',
                'precision': 0
            }
        }

    @property
    def meta(self):
        return {
            'Vhold': self.Vhold,
            'Vstep': self.Vstep,
        }

    @property
    def desc(self):
        return f'Vhold = {self.Vhold:.1f}mV, Vstep = {self.Vstep:.1f}mV'

    @property
    def filecodes(self):
        return {
            'Vhold': f'{self.Vhold:.1f}mV',
            'Vstep': f'{self.Vstep:.1f}mV',
        }

    def compute(self, t):
        return self.Vstep


class AcousticDrive(XDrive):
    ''' Acoustic drive object with intrinsic frequency and amplitude. '''

    xkey = 'A'

    def __init__(self, f, A, phi=np.pi):
        ''' Constructor.

            :param f: carrier frequency (Hz)
            :param A: peak pressure amplitude (Pa)
            :param phi: phase (rad)
        '''
        self.checkInputs(f, A, phi)
        self.f = f
        self.A = A
        self.phi = phi

    @property
    def xvar(self):
        return self.A

    @xvar.setter
    def xvar(self, value):
        self.checkInputs(self.f, value, self.phi)
        self.A = value

    def __repr__(self):
        params = [f'{si_format(self.f, 1, space="")}Hz']
        if self.A is not None:
            params.append(f'{si_format(self.A, 1, space="")}Pa')
        return f'{self.__class__.__name__}({", ".join(params)})'

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.f == other.f and self.A == other.A and self.phi == other.phi

    def copy(self):
        return self.__class__(self.f, self.A, phi=self.phi)

    def checkInputs(self, f, A, phi):
        floatvars = {'f': f, 'phi': phi}
        if A is not None:
            floatvars['A'] = A
        for k, v in floatvars.items():
            if not isinstance(v, float):
                raise TypeError(f'Invalid {k} parameter (must be float typed)')
        if f <= 0:
            d = self.inputs()['f']
            raise ValueError(f'Invalid {d["desc"]}: {f * d.get("factor", 1)} {d["unit"]} (must be strictly positive)')
        if A is not None and A < 0:
            d = self.inputs()['A']
            raise ValueError(f'Invalid {d["desc"]}: {A * d.get("factor", 1)} {d["unit"]} (must be positive or null)')

    @staticmethod
    def inputs():
        return {
            'f': {
                'desc': 'US drive frequency',
                'label': 'f',
                'unit': 'kHz',
                'factor': 1e-3,
                'precision': 0
            },
            'A': {
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
            'f': self.f,
            'A': self.A
        }

    @property
    def desc(self):
        return 'f = {}Hz, A = {}Pa'.format(*si_format([self.f, self.A], 2))

    @property
    def filecodes(self):
        return {
            'f': f'{self.f * 1e-3:.0f}kHz',
            'A': f'{self.A * 1e-3:.2f}kPa'
        }

    @property
    def dt(self):
        ''' Determine integration time step. '''
        return 1 / (NPC_DENSE * self.f)

    @property
    def dt_sparse(self):
        return 1 / (NPC_SPARSE * self.f)

    @property
    def periodicity(self):
        ''' Determine drive periodicity. '''
        return 1. / self.f

    @property
    def nPerCycle(self):
        return NPC_DENSE

    @property
    def modulationFrequency(self):
        return self.f

    def compute(self, t):
        return self.A * np.sin(2 * np.pi * self.f * t - self.phi)


class AcousticDriveArray(Drive):

    def __init__(self, drives):
        self.drives = {f'source {i + 1}': s for i, s in enumerate(drives)}

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        if self.ndrives != other.ndrives:
            return False
        if list(self.drives.keys()) != list(other.drives.keys()):
            return False
        for k, v in self.drives.items():
            if other.drives[k] != v:
                return False
        return True

    def copy(self):
        return self.__class__([x.copy() for x in self.drives.values()])

    @property
    def ndrives(self):
        return len(self.drives)

    @property
    def meta(self):
        return {k: s.meta for k, s in self.drives.items()}

    @property
    def desc(self):
        descs = [f'[{s.desc}]' for k, s in self.drives.items()]
        return ', '.join(descs)

    @property
    def filecodes(self):
        return {k: s.filecodes for k, s in self.drives.items()}

    @property
    def fmax(self):
        return max(s.f for s in self.drives.values())

    @property
    def fmin(self):
        return min(s.f for s in self.drives.values())

    @property
    def dt(self):
        return 1 / (NPC_DENSE * self.fmax)

    @property
    def dt_sparse(self):
        return 1 / (NPC_SPARSE * self.fmax)

    @property
    def periodicity(self):
        if self.ndrives > 2:
            raise ValueError('cannot compute periodicity for more than two drives')
        return 1 / (self.fmax - self.fmin)

    @property
    def nPerCycle(self):
        return int(self.periodicity // self.dt)

    @property
    def modulationFrequency(self):
        return np.mean([s.f for s in self.drives.values()])

    def compute(self, t):
        return sum(s.compute(t) for s in self.drives.values())
