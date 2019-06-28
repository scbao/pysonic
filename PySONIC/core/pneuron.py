# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-03 11:53:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-28 16:10:04

import abc
import re
import pprint
import inspect
import numpy as np
import pandas as pd

from .batches import createQueue
from .model import Model
from .lookup import SmartLookup
from .simulators import PWSimulator
from ..postpro import findPeaks, computeFRProfile
from ..constants import *
from ..utils import si_format, logger, plural, binarySearch


class PointNeuron(Model):
    ''' Generic point-neuron model interface. '''

    tscale = 'ms'  # relevant temporal scale of the model
    simkey = 'ESTIM'  # keyword used to characterize simulations made with this model

    def __init__(self):
        # Determine neuron's resting charge density
        self.Qm0 = self.Cm0 * self.Vm0 * 1e-3  # C/cm2

        # Parse pneuron object to create derEffStates and effRates methods
        self.parse()

    def __repr__(self):
        return self.__class__.__name__

    @property
    @abc.abstractmethod
    def name(self):
        ''' Neuron name. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Cm0(self):
        ''' Neuron's resting capacitance (F/cm2). '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def Vm0(self):
        ''' Neuron's resting membrane potential(mV). '''
        raise NotImplementedError

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

    def filecodes(self, Astim, tstim, toffset, PRF, DC):
        is_CW = DC == 1.
        return {
            'simkey': self.simkey,
            'neuron': self.name,
            'nature': 'CW' if is_CW else 'PW',
            'Astim': '{:.1f}mAm2'.format(Astim),
            'tstim': '{:.0f}ms'.format(tstim * 1e3),
            'toffset': None,
            'PRF': 'PRF{:.2f}Hz'.format(PRF) if not is_CW else None,
            'DC': 'DC{:.2f}%'.format(DC * 1e2) if not is_CW else None
        }

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
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
                'y0': self.Vm0,
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

        for cname in self.getCurrentsNames():
            cfunc = getattr(self, cname)
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
                        'desc': self.states[var],
                        'label': var,
                        'bounds': (-0.1, 1.1)
                    }

        pltvars['iNet'] = {
            'desc': inspect.getdoc(getattr(self, 'iNet')).splitlines()[0],
            'label': 'I_{net}',
            'unit': 'A/m^2',
            'factor': 1e-3,
            'func': 'iNet({0}Vm{1}, {2}{3}{4})'.format(
                wrapleft, wrapright, wrapleft[:-1], self.statesNames(), wrapright[1:]),
            'ls': '--',
            'color': 'black'
        }

        pltvars['dQdt'] = {
            'desc': inspect.getdoc(getattr(self, 'dQdt')).splitlines()[0],
            'label': 'dQ_m/dt',
            'unit': 'A/m^2',
            'factor': 1e-3,
            'func': 'dQdt({0}Vm{1}, {2}{3}{4})'.format(
                wrapleft, wrapright, wrapleft[:-1], self.statesNames(), wrapright[1:]),
            'ls': '--',
            'color': 'black'
        }

        for rate in self.rates:
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

    def getPltScheme(self):
        pltscheme = {
            'Q_m': ['Qm'],
            'V_m': ['Vm']
        }
        pltscheme['I'] = self.getCurrentsNames() + ['iNet']
        for cname in self.getCurrentsNames():
            if 'Leak' not in cname:
                key = 'i_{{{}}}\ kin.'.format(cname[1:])
                cargs = inspect.getargspec(getattr(self, cname))[0][1:]
                pltscheme[key] = [var for var in cargs if var not in ['Vm', 'Cai']]

        return pltscheme

    def statesNames(self):
        ''' Return a list of names of all state variables of the model. '''
        return list(self.states.keys())

    @abc.abstractmethod
    def derStates(self):
        ''' Dictionary of states derivatives functions '''
        raise NotImplementedError

    def getDerStates(self, Vm, states):
        ''' Compute states derivatives array given a membrane potential and states dictionary '''
        return np.array([self.derStates()[k](Vm, states) for k in self.statesNames()])

    @abc.abstractmethod
    def steadyStates(self):
        ''' Return a dictionary of steady-states functions '''
        raise NotImplementedError

    def getSteadyStates(self, Vm):
        ''' Compute array of steady-states for a given membrane potential '''
        return np.array([self.steadyStates()[k](Vm) for k in self.statesNames()])

    def getDerEffStates(self, lkp, states):
        ''' Compute effective states derivatives array given lookups and states dictionaries. '''
        return np.array([
            self.derEffStates()[k](lkp, states) for k in self.statesNames()])

    def getEffRates(self, Vm):
        ''' Compute array of effective rate constants for a given membrane potential vector. '''
        return {k: np.mean(np.vectorize(v)(Vm)) for k, v in self.effRates().items()}

    def getLookup(self):
        ''' Get lookup of membrane potential rate constants interpolated along the neuron's
            charge physiological range. '''
        Qref = np.arange(*self.Qbounds(), 1e-5)  # C/m2
        Vref = Qref / self.Cm0 * 1e3  # mV
        tables = {k: np.vectorize(v)(Vref) for k, v in self.effRates().items()}
        return SmartLookup({'Q': Qref}, {**{'V': Vref}, **tables})

    @abc.abstractmethod
    def currents(self):
        ''' Dictionary of ionic currents functions (returning current densities in mA/m2) '''

    def iNet(self, Vm, states):
        ''' net membrane current

            :param Vm: membrane potential (mV)
            :states: states of ion channels gating and related variables
            :return: current per unit area (mA/m2)
        '''
        return sum([cfunc(Vm, states) for cfunc in self.currents().values()])

    def dQdt(self, Vm, states):
        ''' membrane charge density variation rate

            :param Vm: membrane potential (mV)
            :states: states of ion channels gating and related variables
            :return: variation rate (mA/m2)
        '''
        return -self.iNet(Vm, states)

    def titrationFunc(self, *args, **kwargs):
        ''' Default titration function. '''
        return self.isExcited(*args, **kwargs)

    def currentToConcentrationRate(self, z_ion, depth):
        ''' Compute the conversion factor from a specific ionic current (in mA/m2)
            into a variation rate of submembrane ion concentration (in M/s).

            :param: z_ion: ion valence
            :param depth: submembrane depth (m)
            :return: conversion factor (Mmol.m-1.C-1)
        '''
        return 1e-6 / (z_ion * depth * FARADAY)

    def nernst(self, z_ion, Cion_in, Cion_out, T):
        ''' Nernst potential of a specific ion given its intra and extracellular concentrations.

            :param z_ion: ion valence
            :param Cion_in: intracellular ion concentration
            :param Cion_out: extracellular ion concentration
            :param T: temperature (K)
            :return: ion Nernst potential (mV)
        '''
        return (Rg * T) / (z_ion * FARADAY) * np.log(Cion_out / Cion_in) * 1e3

    def vtrap(self, x, y):
        ''' Generic function used to compute rate constants. '''
        return x / (np.exp(x / y) - 1)

    def efun(self, x):
        ''' Generic function used to compute rate constants. '''
        return x / (np.exp(x) - 1)

    def ghkDrive(self, Vm, Z_ion, Cion_in, Cion_out, T):
        ''' Use the Goldman-Hodgkin-Katz equation to compute the electrochemical driving force
            of a specific ion species for a given membrane potential.

            :param Vm: membrane potential (mV)
            :param Cin: intracellular ion concentration (M)
            :param Cout: extracellular ion concentration (M)
            :param T: temperature (K)
            :return: electrochemical driving force of a single ion particle (mC.m-3)
        '''
        x = Z_ion * FARADAY * Vm / (Rg * T) * 1e-3   # [-]
        eCin = Cion_in * self.efun(-x)  # M
        eCout = Cion_out * self.efun(x)  # M
        return FARADAY * (eCin - eCout) * 1e6  # mC/m3

    def getCurrentsNames(self):
        return list(self.currents().keys())

    def firingRateProfile(*args, **kwargs):
        return computeFRProfile(*args, **kwargs)

    def Qbounds(self):
        ''' Determine bounds of membrane charge physiological range for a given neuron. '''
        return np.array([np.round(self.Vm0 - 25.0), 50.0]) * self.Cm0 * 1e-3  # C/m2

    def isVoltageGated(self, state):
        ''' Determine whether a given state is purely voltage-gated or not.'''
        return 'alpha{}'.format(state.lower()) in self.rates

    def qsStates(self, lkp, states):
        ''' Compute a collection of quasi steady states using the standard
            xinf = ax / (ax + Bx) equation.

            :param lkp: dictionary of 1D vectors of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :return: dictionary of quasi-steady states
        '''
        return {
            x: lkp['alpha{}'.format(x)] / (lkp['alpha{}'.format(x)] + lkp['beta{}'.format(x)])
            for x in states
        }

    def quasiSteadyStates(self, lkp):
        ''' Compute the quasi-steady states of a neuron for a range of membrane charge densities,
            based on 1-dimensional lookups interpolated at a given sonophore diameter, US frequency,
            US amplitude and duty cycle.

            :param lkp: dictionary of 1D vectors of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :return: dictionary of quasi-steady states
        '''
        return self.qsStates(lkp, self.statesNames())
        # return {k: func(lkp['Vm']) for k, func in self.steadyStates().items()}

    @staticmethod
    def simQueue(amps, durations, offsets, PRFs, DCs, outputdir=None):
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
            queue += createQueue(amps, durations, offsets, min(PRFs), 1.0)
        if np.any(DCs != 1.0):
            queue += createQueue(amps, durations, offsets, PRFs, DCs[DCs != 1.0])
        for item in queue:
            if np.isnan(item[0]):
                item[0] = None
        if outputdir is not None:
            for item in queue:
                item.insert(0, outputdir)
        return queue

    @staticmethod
    def checkInputs(Astim, tstim, toffset, PRF, DC):
        ''' Check validity of electrical stimulation parameters.

            :param Astim: pulse amplitude (mA/m2)
            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
        '''

        # Check validity of stimulation parameters
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

    def derivatives(self, t, y, Cm=None, Iinj=0.):
        ''' Compute system derivatives for a given mambrane capacitance and injected current.

            :param t: specific instant in time (s)
            :param y: vector of HH system variables at time t
            :param Cm: membrane capacitance (F/m2)
            :param Iinj: injected current (mA/m2)
            :return: vector of system derivatives at time t
        '''
        if Cm is None:
            Cm = self.Cm0
        Qm, *states = y
        Vm = Qm / Cm * 1e3  # mV
        states_dict = dict(zip(self.statesNames(), states))
        dQmdt = (Iinj - self.iNet(Vm, states_dict)) * 1e-3  # A/m2
        return [dQmdt, *self.getDerStates(Vm, states_dict)]

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
        y0 = np.array((self.Qm0, *self.getSteadyStates(self.Vm0)))

        # Initialize simulator and compute solution
        logger.debug('Computing solution')
        simulator = PWSimulator(
            lambda t, y: self.derivatives(t, y, Iinj=Astim),
            lambda t, y: self.derivatives(t, y, Iinj=0.))
        (t, y, stim), tcomp = simulator(
            y0, DT_EFFECTIVE, tstim, toffset, PRF, DC, monitor_time=True)
        logger.debug('completed in %ss', si_format(tcomp, 1))

        # Store output in dataframe
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Qm': y[:, 0],
            'Vm': y[:, 0] / self.Cm0 * 1e3,
        })
        # data['Qm'] = data['Vm'].values * self.Cm0 * 1e-3
        for i in range(len(self.states)):
            data[self.statesNames()[i]] = y[:, i + 1]

        # Log number of detected spikes
        nspikes = self.getNSpikes(data)
        logger.debug('{} spike{} detected'.format(nspikes, plural(nspikes)))

        # Return dataframe and computation time
        return data, tcomp

    def meta(self, Astim, tstim, toffset, PRF, DC):
        return {
            'simkey': self.simkey,
            'neuron': self.name,
            'Astim': Astim,
            'tstim': tstim,
            'toffset': toffset,
            'PRF': PRF,
            'DC': DC
        }

    def getNSpikes(self, data):
        ''' Compute number of spikes in charge profile of simulation output.

            :param data: dataframe containing output time series
            :return: number of detected spikes
        '''
        dt = np.diff(data.ix[:1, 't'].values)[0]
        ipeaks, *_ = findPeaks(
            data['Qm'].values,
            SPIKE_MIN_QAMP,
            int(np.ceil(SPIKE_MIN_DT / dt)),
            SPIKE_MIN_QPROM
        )
        return ipeaks.size

    def getStabilizationValue(self, data):
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

    def isExcited(self, data):
        ''' Determine if neuron is excited from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is excited or not
        '''
        return self.getNSpikes(data) > 0

    def isSilenced(self, data):
        ''' Determine if neuron is silenced from simulation output.

            :param data: dataframe containing output time series
            :return: boolean stating whether neuron is silenced or not
        '''
        return not np.isnan(self.getStabilizationValue(data))

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

    def parse(self):
        ''' Parse neuron's inner methods to construct adapted methods for SONIC simulations. '''

        def getLambdaSource(dict_entry):
            clean_dict_entry = re.sub(
                ' +', ' ',
                inspect.getsource(dict_entry).replace('\n', ' ')).strip()
            if clean_dict_entry[-1] == ',':
                clean_dict_entry = clean_dict_entry[:-1]
            fsource = clean_dict_entry.split(':', 1)[-1].strip()
            lambda_pattern = 'lambda ([a-z_A-Z,0-9\s]*): (.+)'
            return re.match(lambda_pattern, fsource).groups()

        def getFuncCalls(s):
            func_pattern = '([a-z_A-Z]*).([a-z_A-Z][a-z_A-Z0-9]*)\(([^\)]*)\)'
            return re.finditer(func_pattern, s)

        def createStateEffectiveDerivative(expr):
            lambda_expr = 'lambda lkp, x: {}'.format(expr)
            lambda_expr_self = 'lambda self, lkp, x: {}'.format(expr)
            f = eval(lambda_expr_self)
            return lambda_expr, lambda *args: f(self, *args)

        def defineConstLambda(const):
            return lambda _: const

        def addEffRates(expr, d, dstr):
            # Define patterns
            suffix_pattern = '[A-Za-z0-9_]+'
            xinf_pattern = re.compile('^({})inf$'.format(suffix_pattern))
            taux_pattern = re.compile('^tau({})$'.format(suffix_pattern))
            alphax_pattern = re.compile('^alpha({})$'.format(suffix_pattern))
            betax_pattern = re.compile('^beta({})$'.format(suffix_pattern))
            err_str = 'gating states must be defined via the alphaX-betaX or Xinf-tauX paradigm'

            # If expression matches alpha or beta rate -> return corresponding
            # effective rate function
            if alphax_pattern.match(expr) or betax_pattern.match(expr):
                try:
                    d[expr] = getattr(self, expr)
                except AttributeError:
                    raise ValueError(err_str)
                dstr[expr] = 'self.{}'.format(expr)

            # If expression matches xinf or taux -> add corresponding alpha and beta
            # effective rates functions
            else:
                for pattern in [taux_pattern, xinf_pattern]:
                    m = pattern.match(expr)
                    if m:
                        k = m.group(1)
                        alphax_str, betax_str = ['{}{}'.format(p, k) for p in ['alpha', 'beta']]
                        xinf_str, taux_str = ['{}inf'.format(k), 'tau{}'.format(k)]
                        try:
                            xinf, taux = [getattr(self, s) for s in [xinf_str, taux_str]]
                            # If taux is a constant, define a lambda function that returns it
                            if not callable(taux):
                                taux = defineConstLambda(taux)
                            d[alphax_str] = lambda Vm: xinf(Vm) / taux(Vm)
                            d[betax_str] = lambda Vm: (1 - xinf(Vm)) / taux(Vm)
                        except AttributeError:
                            raise ValueError(err_str)
                        dstr.update({
                            alphax_str: 'lambda Vm: self.{}(Vm) / self.{}(Vm)'.format(
                                xinf_str, taux_str),
                            betax_str: 'lambda Vm: (1 - self.{}(Vm)) / self.{}(Vm)'.format(
                                xinf_str, taux_str)
                        })

        # Initialize empty dictionaries to gather effective derivatives functions
        # and effective rates functions
        eff_dstates, eff_dstates_str = {}, {}
        eff_rates, eff_rates_str = {}, {}

        # For each state derivative
        for k, dfunc in self.derStates().items():
            # Get derivative function source code
            dfunc_args, dfunc_exp = getLambdaSource(dfunc)

            # For each internal function call in the derivative function expression
            matches = getFuncCalls(dfunc_exp)
            for m in matches:
                # Determine function arguments
                fprefix, fname, fargs = m.groups()
                fcall = '{}({})'.format(fname, fargs)
                if fprefix:
                    fcall = '{}.{}'.format(fprefix, fcall)
                args_list = fargs.split(',')

                # If sole argument is Vm
                if len(args_list) == 1 and args_list[0] == 'Vm':
                    # Replace function call by lookup retrieval in expression
                    dfunc_exp = dfunc_exp.replace(fcall, "lkp['{}']".format(fname))

                    # Add the corresponding effective rate function(s) to the dictionnary
                    addEffRates(fname, eff_rates, eff_rates_str)

            # Replace Vm by lkp['V'] in expression
            dfunc_exp = dfunc_exp.replace('Vm', "lkp['V']")

            # Create the modified lambda expression and evaluate it
            eff_dstates_str[k], eff_dstates[k] = createStateEffectiveDerivative(dfunc_exp)

        self.rates = list(eff_rates.keys())

        # Define methods that return dictionaries of effective states derivatives
        # and effective rates functions, along with their corresponding string equivalents
        def derEffStates():
            return eff_dstates

        def printDerEffStates():
            pprint.PrettyPrinter(indent=4).pprint(eff_dstates_str)

        def effRates():
            return eff_rates

        def printEffRates():
            pprint.PrettyPrinter(indent=4).pprint(eff_rates_str)

        if not hasattr(self, 'derEffStates'):
            self.derEffStates = derEffStates
            self.printDerEffStates = printDerEffStates
        if not hasattr(self, 'effRates'):
            self.effRates = effRates
            self.printEffRates = printEffRates
