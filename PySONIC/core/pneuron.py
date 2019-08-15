# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-03 11:53:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-14 19:31:06

import abc
import inspect
import numpy as np
import pandas as pd

from .batches import Batch
from .model import Model
from .lookups import SmartLookup
from .simulators import PWSimulator
from ..postpro import findPeaks, computeFRProfile
from ..constants import *
from ..utils import *


class PointNeuron(Model):
    ''' Generic point-neuron model interface. '''

    tscale = 'ms'  # relevant temporal scale of the model
    simkey = 'ESTIM'  # keyword used to characterize simulations made with this model

    def __repr__(self):
        return self.__class__.__name__

    @property
    @classmethod
    @abc.abstractmethod
    def name(cls):
        ''' Neuron name. '''
        raise NotImplementedError

    @property
    @classmethod
    @abc.abstractmethod
    def Cm0(cls):
        ''' Neuron's resting capacitance (F/cm2). '''
        raise NotImplementedError

    @property
    @classmethod
    @abc.abstractmethod
    def Vm0(cls):
        ''' Neuron's resting membrane potential(mV). '''
        raise NotImplementedError

    @classmethod
    def Qm0(cls):
        return cls.Cm0 * cls.Vm0 * 1e-3  # C/cm2

    @staticmethod
    def inputs():
        return {
            'Astim': {
                'desc': 'current density amplitude',
                'label': 'A',
                'unit': 'mA/m2',
                'factor': 1e0,
                'precision': 1
            },
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
            },
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

    @classmethod
    def filecodes(cls, Astim, tstim, toffset, PRF, DC):
        is_CW = DC == 1.
        return {
            'simkey': cls.simkey,
            'neuron': cls.name,
            'nature': 'CW' if is_CW else 'PW',
            'Astim': '{:.1f}mAm2'.format(Astim),
            'tstim': '{:.0f}ms'.format(tstim * 1e3),
            'toffset': None,
            'PRF': 'PRF{:.2f}Hz'.format(PRF) if not is_CW else None,
            'DC': 'DC{:.2f}%'.format(DC * 1e2) if not is_CW else None
        }

    @classmethod
    def getPltVars(cls, wrapleft='df["', wrapright='"]'):
        pltvars = {
            'Qm': {
                'desc': 'membrane charge density',
                'label': 'Q_m',
                'unit': 'nC/cm^2',
                'factor': 1e5,
                'bounds': (-100, 50)
            },

            'Vm': {
                'desc': 'membrane potential',
                'label': 'V_m',
                'unit': 'mV',
                'bounds': (-150, 70)
            },

            'ELeak': {
                'constant': 'obj.ELeak',
                'desc': 'non-specific leakage current resting potential',
                'label': 'V_{leak}',
                'unit': 'mV',
                'ls': '--',
                'color': 'k'
            }
        }

        for cname in cls.getCurrentsNames():
            cfunc = getattr(cls, cname)
            cargs = inspect.getargspec(cfunc)[0][1:]
            pltvars[cname] = {
                'desc': inspect.getdoc(cfunc).splitlines()[0],
                'label': 'I_{{{}}}'.format(cname[1:]),
                'unit': 'A/m^2',
                'factor': 1e-3,
                'func': '{}({})'.format(cname, ', '.join(['{}{}{}'.format(wrapleft, a, wrapright)
                                                          for a in cargs]))
            }
            for var in cargs:
                if var != 'Vm':
                    pltvars[var] = {
                        'desc': cls.states[var],
                        'label': var,
                        'bounds': (-0.1, 1.1)
                    }

        pltvars['iNet'] = {
            'desc': inspect.getdoc(getattr(cls, 'iNet')).splitlines()[0],
            'label': 'I_{net}',
            'unit': 'A/m^2',
            'factor': 1e-3,
            'func': 'iNet({0}Vm{1}, {2}{3}{4})'.format(
                wrapleft, wrapright, wrapleft[:-1], cls.statesNames(), wrapright[1:]),
            'ls': '--',
            'color': 'black'
        }

        pltvars['dQdt'] = {
            'desc': inspect.getdoc(getattr(cls, 'dQdt')).splitlines()[0],
            'label': 'dQ_m/dt',
            'unit': 'A/m^2',
            'factor': 1e-3,
            'func': 'dQdt({0}Vm{1}, {2}{3}{4})'.format(
                wrapleft, wrapright, wrapleft[:-1], cls.statesNames(), wrapright[1:]),
            'ls': '--',
            'color': 'black'
        }

        pltvars['iCap'] = {
            'desc': inspect.getdoc(getattr(cls, 'iCap')).splitlines()[0],
            'label': 'I_{cap}',
            'unit': 'A/m^2',
            'factor': 1e-3,
            'func': 'iCap({0}t{1}, {0}Vm{1})'.format(wrapleft, wrapright)
        }

        for rate in cls.rates:
            if 'alpha' in rate:
                prefix, suffix = 'alpha', rate[5:]
            else:
                prefix, suffix = 'beta', rate[4:]
            pltvars['{}'.format(rate)] = {
                'label': '\\{}_{{{}}}'.format(prefix, suffix),
                'unit': 'ms^{-1}',
                'factor': 1e-3
            }

        pltvars['FR'] = {
            'desc': 'riring rate',
            'label': 'FR',
            'unit': 'Hz',
            'factor': 1e0,
            # 'bounds': (0, 1e3),
            'func': 'firingRateProfile({0}t{1}.values, {0}Qm{1}.values)'.format(wrapleft, wrapright)
        }

        return pltvars

    @classmethod
    def iCap(cls, t, Vm):
        ''' Capacitive current. '''
        dVdt = np.insert(np.diff(Vm) / np.diff(t), 0, 0.)
        return cls.Cm0 * dVdt

    @classmethod
    def getPltScheme(cls):
        pltscheme = {
            'Q_m': ['Qm'],
            'V_m': ['Vm']
        }
        pltscheme['I'] = cls.getCurrentsNames() + ['iNet']
        for cname in cls.getCurrentsNames():
            if 'Leak' not in cname:
                key = 'i_{{{}}}\ kin.'.format(cname[1:])
                cargs = inspect.getargspec(getattr(cls, cname))[0][1:]
                pltscheme[key] = [var for var in cargs if var not in ['Vm', 'Cai']]

        return pltscheme

    @classmethod
    def statesNames(cls):
        ''' Return a list of names of all state variables of the model. '''
        return list(cls.states.keys())

    @classmethod
    @abc.abstractmethod
    def derStates(cls):
        ''' Dictionary of states derivatives functions '''
        raise NotImplementedError

    @classmethod
    def getDerStates(cls, Vm, states):
        ''' Compute states derivatives array given a membrane potential and states dictionary '''
        return np.array([cls.derStates()[k](Vm, states) for k in cls.statesNames()])

    @classmethod
    @abc.abstractmethod
    def steadyStates(cls):
        ''' Return a dictionary of steady-states functions '''
        raise NotImplementedError

    @classmethod
    def getSteadyStates(cls, Vm):
        ''' Compute array of steady-states for a given membrane potential '''
        return np.array([cls.steadyStates()[k](Vm) for k in cls.statesNames()])

    @classmethod
    def getDerEffStates(cls, lkp, states):
        ''' Compute effective states derivatives array given lookups and states dictionaries. '''
        return np.array([
            cls.derEffStates()[k](lkp, states) for k in cls.statesNames()])

    @classmethod
    def getEffRates(cls, Vm):
        ''' Compute array of effective rate constants for a given membrane potential vector. '''
        return {k: np.mean(np.vectorize(v)(Vm)) for k, v in cls.effRates().items()}

    @classmethod
    def getLookup(cls):
        ''' Get lookup of membrane potential rate constants interpolated along the neuron's
            charge physiological range. '''
        Qref = np.arange(*cls.Qbounds(), 1e-5)  # C/m2
        Vref = Qref / cls.Cm0 * 1e3  # mV
        tables = {k: np.vectorize(v)(Vref) for k, v in cls.effRates().items()}
        return SmartLookup({'Q': Qref}, {**{'V': Vref}, **tables})

    @classmethod
    @abc.abstractmethod
    def currents(cls):
        ''' Dictionary of ionic currents functions (returning current densities in mA/m2) '''

    @classmethod
    def iNet(cls, Vm, states):
        ''' net membrane current

            :param Vm: membrane potential (mV)
            :states: states of ion channels gating and related variables
            :return: current per unit area (mA/m2)
        '''
        return sum([cfunc(Vm, states) for cfunc in cls.currents().values()])

    @classmethod
    def dQdt(cls, Vm, states):
        ''' membrane charge density variation rate

            :param Vm: membrane potential (mV)
            :states: states of ion channels gating and related variables
            :return: variation rate (mA/m2)
        '''
        return -cls.iNet(Vm, states)

    @classmethod
    def titrationFunc(cls, *args, **kwargs):
        ''' Default titration function. '''
        return cls.isExcited(*args, **kwargs)

    @staticmethod
    def currentToConcentrationRate(z_ion, depth):
        ''' Compute the conversion factor from a specific ionic current (in mA/m2)
            into a variation rate of submembrane ion concentration (in M/s).

            :param: z_ion: ion valence
            :param depth: submembrane depth (m)
            :return: conversion factor (Mmol.m-1.C-1)
        '''
        return 1e-6 / (z_ion * depth * FARADAY)

    @staticmethod
    def nernst(z_ion, Cion_in, Cion_out, T):
        ''' Nernst potential of a specific ion given its intra and extracellular concentrations.

            :param z_ion: ion valence
            :param Cion_in: intracellular ion concentration
            :param Cion_out: extracellular ion concentration
            :param T: temperature (K)
            :return: ion Nernst potential (mV)
        '''
        return (Rg * T) / (z_ion * FARADAY) * np.log(Cion_out / Cion_in) * 1e3

    @staticmethod
    def vtrap(x, y):
        ''' Generic function used to compute rate constants. '''
        return x / (np.exp(x / y) - 1)

    @staticmethod
    def efun(x):
        ''' Generic function used to compute rate constants. '''
        return x / (np.exp(x) - 1)

    @classmethod
    def ghkDrive(cls, Vm, Z_ion, Cion_in, Cion_out, T):
        ''' Use the Goldman-Hodgkin-Katz equation to compute the electrochemical driving force
            of a specific ion species for a given membrane potential.

            :param Vm: membrane potential (mV)
            :param Cin: intracellular ion concentration (M)
            :param Cout: extracellular ion concentration (M)
            :param T: temperature (K)
            :return: electrochemical driving force of a single ion particle (mC.m-3)
        '''
        x = Z_ion * FARADAY * Vm / (Rg * T) * 1e-3   # [-]
        eCin = Cion_in * cls.efun(-x)  # M
        eCout = Cion_out * cls.efun(x)  # M
        return FARADAY * (eCin - eCout) * 1e6  # mC/m3

    @classmethod
    def getCurrentsNames(cls):
        return list(cls.currents().keys())

    def firingRateProfile(*args, **kwargs):
        return computeFRProfile(*args, **kwargs)

    @classmethod
    def Qbounds(cls):
        ''' Determine bounds of membrane charge physiological range for a given neuron. '''
        return np.array([np.round(cls.Vm0 - 25.0), 50.0]) * cls.Cm0 * 1e-3  # C/m2

    @classmethod
    def isVoltageGated(cls, state):
        ''' Determine whether a given state is purely voltage-gated or not.'''
        return 'alpha{}'.format(state.lower()) in cls.rates

    @classmethod
    def simQueue(cls, amps, durations, offsets, PRFs, DCs, outputdir=None):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps, while avoiding repetition of CW protocols for a given PRF sweep.

            :param amps: list (or 1D-array) of acoustic amplitudes
            :param durations: list (or 1D-array) of stimulus durations
            :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
            :param PRFs: list (or 1D-array) of pulse-repetition frequencies
            :param DCs: list (or 1D-array) of duty cycle values
            :return: list of parameters (list) for each simulation
        '''
        if amps is None:
            amps = [np.nan]
        DCs = np.array(DCs)
        queue = []
        if 1.0 in DCs:
            queue += Batch.createQueue(amps, durations, offsets, min(PRFs), 1.0)
        if np.any(DCs != 1.0):
            queue += Batch.createQueue(amps, durations, offsets, PRFs, DCs[DCs != 1.0])
        for item in queue:
            if np.isnan(item[0]):
                item[0] = None
        return cls.checkOutputDir(queue, outputdir)

    @staticmethod
    def checkInputs(Astim, tstim, toffset, PRF, DC):
        ''' Check validity of electrical stimulation parameters.

            :param Astim: pulse amplitude (mA/m2)
            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
        '''
        if not all(isinstance(param, float) for param in [Astim, tstim, toffset, DC]):
            raise TypeError('Invalid stimulation parameters (must be float typed)')
        if tstim <= 0:
            raise ValueError('Invalid stimulus duration: {} ms (must be strictly positive)'
                             .format(tstim * 1e3))
        if toffset < 0:
            raise ValueError('Invalid stimulus offset: {} ms (must be positive or null)'
                             .format(toffset * 1e3))
        if DC <= 0.0 or DC > 1.0:
            raise ValueError('Invalid duty cycle: {} (must be within ]0; 1])'.format(DC))
        if DC < 1.0:
            if not isinstance(PRF, float):
                raise TypeError('Invalid PRF value (must be float typed)')
            if PRF is None:
                raise AttributeError('Missing PRF value (must be provided when DC < 1)')
            if PRF < 1 / tstim:
                raise ValueError('Invalid PRF: {} Hz (PR interval exceeds stimulus duration)'
                                 .format(PRF))

    @classmethod
    def derivatives(cls, t, y, Cm=None, Iinj=0.):
        ''' Compute system derivatives for a given mambrane capacitance and injected current.

            :param t: specific instant in time (s)
            :param y: vector of HH system variables at time t
            :param Cm: membrane capacitance (F/m2)
            :param Iinj: injected current (mA/m2)
            :return: vector of system derivatives at time t
        '''
        if Cm is None:
            Cm = cls.Cm0
        Qm, *states = y
        Vm = Qm / Cm * 1e3  # mV
        states_dict = dict(zip(cls.statesNames(), states))
        dQmdt = (Iinj - cls.iNet(Vm, states_dict)) * 1e-3  # A/m2
        return [dQmdt, *cls.getDerStates(Vm, states_dict)]

    @Model.logNSpikes
    @Model.checkTitrate('Astim')
    @Model.addMeta
    def simulate(self, Astim, tstim, toffset, PRF=100., DC=1.0):
        ''' Simulate a specific neuron model for a specific set of electrical parameters,
            and return output data in a dataframe.

            :param Astim: pulse amplitude (mA/m2)
            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :return: 2-tuple with the output dataframe and computation time.
        '''
        logger.info(
            '%s: simulation @ A = %sA/m2, t = %ss (%ss offset)%s',
            self, si_format(Astim, 2, space=' '), *si_format([tstim, toffset], 1, space=' '),
            (', PRF = {}Hz, DC = {:.2f}%'.format(
                si_format(PRF, 2, space=' '), DC * 1e2) if DC < 1.0 else ''))

        # Check validity of stimulation parameters
        self.checkInputs(Astim, tstim, toffset, PRF, DC)

        # Set initial conditions
        y0 = np.array((self.Qm0(), *self.getSteadyStates(self.Vm0)))

        # Initialize simulator and compute solution
        logger.debug('Computing solution')
        simulator = PWSimulator(
            lambda t, y: self.derivatives(t, y, Iinj=Astim),
            lambda t, y: self.derivatives(t, y, Iinj=0.))
        t, y, stim = simulator(
            y0, DT_EFFECTIVE, tstim, toffset, PRF, DC)

        # Prepend initial conditions (prior to stimulation)
        t, y, stim = simulator.prependSolution(t, y, stim)

        # Store output in dataframe and return
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Qm': y[:, 0],
            'Vm': y[:, 0] / self.Cm0 * 1e3,
        })
        for i in range(len(self.states)):
            data[self.statesNames()[i]] = y[:, i + 1]
        return data

    @classmethod
    def meta(cls, Astim, tstim, toffset, PRF, DC):
        return {
            'simkey': cls.simkey,
            'neuron': cls.name,
            'Astim': Astim,
            'tstim': tstim,
            'toffset': toffset,
            'PRF': PRF,
            'DC': DC
        }

    @staticmethod
    def getNSpikes(data):
        ''' Compute number of spikes in charge profile of simulation output.

            :param data: dataframe containing output time series
            :return: number of detected spikes
        '''
        dt = np.diff(data.ix[1:2, 't'].values)[0]
        ipeaks, *_ = findPeaks(
            data['Qm'].values,
            SPIKE_MIN_QAMP,
            int(np.ceil(SPIKE_MIN_DT / dt)),
            SPIKE_MIN_QPROM
        )
        return ipeaks.size

    @staticmethod
    def getStabilizationValue(data):
        ''' Determine stabilization value from the charge profile of a simulation output.

            :param data: dataframe containing output time series
            :return: charge stabilization value (or np.nan if no stabilization detected)
        '''

        # Extract charge signal posterior to observation window
        t, Qm = [data[key].values for key in ['t', 'Qm']]
        if t.max() <= TMIN_STABILIZATION:
            raise ValueError('solution length is too short to assess stabilization')
        Qm = Qm[t > TMIN_STABILIZATION]

        # Compute variation range
        Qm_range = np.ptp(Qm)
        logger.debug('%.2f nC/cm2 variation range over the last %.0f ms, Qmf = %.2f nC/cm2',
                     Qm_range * 1e5, TMIN_STABILIZATION * 1e3, Qm[-1] * 1e5)

        # Return final value only if stabilization is detected
        if np.ptp(Qm) < QSS_Q_DIV_THR:
            return Qm[-1]
        else:
            return np.nan

    @classmethod
    def isExcited(cls, data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        return cls.getNSpikes(data) > 0

    @classmethod
    def isSilenced(cls, data):
        ''' Determine if neuron is silenced from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is silenced or not
        '''
        return not np.isnan(cls.getStabilizationValue(data))

    def titrate(self, tstim, toffset, PRF, DC, xfunc=None, Arange=(0., 2 * AMP_UPPER_BOUND_ESTIM)):
        ''' Use a binary search to determine the threshold amplitude needed
            to obtain neural excitation for a given duration, PRF and duty cycle.

            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param xfunc: function determining whether condition is reached from simulation output
            :param Arange: search interval for Astim, iteratively refined
            :return: excitation threshold amplitude (mA/m2)
        '''
        # Default output function
        if xfunc is None:
            xfunc = self.titrationFunc

        return binarySearch(
            lambda x: xfunc(self.simulate(*x)[0]),
            [tstim, toffset, PRF, DC], 0, Arange, THRESHOLD_CONV_RANGE_ESTIM
        )
