# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-03 11:53:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-01-14 13:47:44

import abc
import inspect
import numpy as np
import pandas as pd

from .protocols import PulsedProtocol, createPulsedProtocols
from .model import Model
from .lookups import SmartLookup
from .simulators import PWSimulator
from ..postpro import detectSpikes, computeFRProfile
from ..constants import *
from ..utils import *
from ..threshold import threshold


class PointNeuron(Model):
    ''' Generic point-neuron model interface. '''

    tscale = 'ms'  # relevant temporal scale of the model
    simkey = 'ESTIM'  # keyword used to characterize simulations made with this model
    titration_var = 'Astim'  # name of the titration parameter

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
        return {**{
            'Astim': {
                'desc': 'current density amplitude',
                'label': 'A',
                'unit': 'mA/m2',
                'factor': 1e0,
                'precision': 1
            }}, **PulsedProtocol.inputs()}

    @classmethod
    def filecodes(cls, Astim, pp):
        return {**{
            'simkey': cls.simkey,
            'neuron': cls.name,
            'nature': 'CW' if pp.isCW() else 'PW',
            'Astim': '{:.1f}mAm2'.format(Astim)},
            **pp.filecodes()
        }

    @classmethod
    def getPltVars(cls, wrapleft='df["', wrapright='"]'):
        pltvars = {
            'Qm': {
                'desc': 'membrane charge density',
                'label': 'Q_m',
                'unit': 'nC/cm^2',
                'factor': 1e5,
                'bounds': ((cls.Vm0 - 20.0) * cls.Cm0 * 1e2, 60)
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
            'func': f'firingRateProfile({wrapleft[:-2]})'
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
        Qmin, Qmax = expandRange(*cls.Qbounds(), 10.)
        Qref = np.arange(Qmin, Qmax, 1e-5)  # C/m2
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
    def xBG(cls, Vref, Vm):
        ''' Compute dimensionless Borg-Graham ratio for a given voltage.

            :param Vref: reference voltage membrane (mV)
            :param Vm: membrane potential (mV)
            :return: dimensionless ratio
        '''
        return (Vm - Vref) * FARADAY / (Rg * cls.T) * 1e-3  # [-]

    @classmethod
    def alphaBG(cls, alpha0, zeta, gamma, Vref,  Vm):
        ''' Compute the activation rate constant for a given voltage and temperature, using
            a Borg-Graham formalism.

            :param alpha0: pre-exponential multiplying factor
            :param zeta: effective valence of the gating particle
            :param gamma: normalized position of the transition state within the membrane
            :param Vref: membrane voltage at which alpha = alpha0 (mV)
            :param Vm: membrane potential (mV)
            :return: rate constant (in alpha0 units)
        '''
        return alpha0 * np.exp(-zeta * gamma * cls.xBG(Vref, Vm))

    @classmethod
    def betaBG(cls, beta0, zeta, gamma, Vref, Vm):
        ''' Compute the inactivation rate constant for a given voltage and temperature, using
            a Borg-Graham formalism.

            :param beta0: pre-exponential multiplying factor
            :param zeta: effective valence of the gating particle
            :param gamma: normalized position of the transition state within the membrane
            :param Vref: membrane voltage at which beta = beta0 (mV)
            :param Vm: membrane potential (mV)
            :return: rate constant (in beta0 units)
        '''
        return beta0 * np.exp(zeta * (1 - gamma) * cls.xBG(Vref, Vm))

    @classmethod
    def getCurrentsNames(cls):
        return list(cls.currents().keys())

    @staticmethod
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
    @Model.checkOutputDir
    def simQueue(cls, amps, durations, offsets, PRFs, DCs, **kwargs):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps.

            :param amps: list (or 1D-array) of acoustic amplitudes
            :param durations: list (or 1D-array) of stimulus durations
            :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
            :param PRFs: list (or 1D-array) of pulse-repetition frequencies
            :param DCs: list (or 1D-array) of duty cycle values
            :return: list of parameters (list) for each simulation
        '''
        if amps is None:
            amps = [None]
        ppqueue = createPulsedProtocols(durations, offsets, PRFs, DCs)
        queue = []
        for A in amps:
            for item in ppqueue:
                queue.append([A, item])
        return queue

    @staticmethod
    def checkInputs(Astim, pp):
        ''' Check validity of electrical stimulation parameters.

            :param Astim: pulse amplitude (mA/m2)
            :param pp: pulse protocol object
        '''
        if not isinstance(Astim, float):
            raise TypeError('Invalid simulation amplitude (must be float typed)')
        if not isinstance(pp, PulsedProtocol):
            raise TypeError('Invalid pulsed protocol (must be "PulsedProtocol" instance)')

    def chooseTimeStep(self):
        ''' Determine integration time step based on intrinsic temporal properties. '''
        return DT_EFFECTIVE

    @classmethod
    def derivatives(cls, t, y, Cm=None, Iinj=0.):
        ''' Compute system derivatives for a given membrane capacitance and injected current.

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
    @Model.checkTitrate
    @Model.addMeta
    @Model.logDesc
    @Model.checkSimParams
    def simulate(self, Astim, pp):
        ''' Simulate a specific neuron model for a set of simulation parameters,
            and return output data in a dataframe.

            :param Astim: pulse amplitude (mA/m2)
            :param pp: pulse protocol object
            :return: output dataframe
        '''

        # Set initial conditions
        y0 = np.array((self.Qm0(), *self.getSteadyStates(self.Vm0)))

        # Initialize simulator and compute solution
        logger.debug('Computing solution')
        simulator = PWSimulator(
            lambda t, y: self.derivatives(t, y, Iinj=Astim),
            lambda t, y: self.derivatives(t, y, Iinj=0.))
        t, y, stim = simulator(
            y0, self.chooseTimeStep(), pp)

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
    def meta(cls, Astim, pp):
        return {
            'simkey': cls.simkey,
            'neuron': cls.name,
            'Astim': Astim,
            'pp': pp
        }

    def desc(self, meta):
        return '{}: simulation @ A = {}A/m2, {}'.format(
            self, si_format(meta["Astim"] * 1e-3, 2), meta['pp'].pprint())

    @staticmethod
    def getNSpikes(data):
        ''' Compute number of spikes in charge profile of simulation output.

            :param data: dataframe containing output time series
            :return: number of detected spikes
        '''
        return detectSpikes(data)[0].size

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

    def titrate(self, pp, xfunc=None, Arange=(0., ESTIM_AMP_UPPER_BOUND)):
        ''' Use a binary search to determine the threshold amplitude needed
            to obtain neural excitation for a given duration, PRF and duty cycle.

            :param pp: pulsed protocol object
            :param xfunc: function determining whether condition is reached from simulation output
            :param Arange: search interval for Astim, iteratively refined
            :return: excitation threshold amplitude (mA/m2)
        '''
        # Default output function
        if xfunc is None:
            xfunc = self.titrationFunc

        return threshold(
            lambda x: xfunc(self.simulate(x, pp)[0]),
            Arange, x0=ESTIM_AMP_INITIAL, rel_eps_thr=ESTIM_REL_CONV_THR, precheck=False)
