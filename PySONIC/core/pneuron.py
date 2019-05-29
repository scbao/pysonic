#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-03 11:53:04
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-29 13:31:11

import time
import pickle
import abc
import inspect
import re
import numpy as np
from scipy.integrate import odeint
import pandas as pd

from ..postpro import findPeaks
from ..constants import *
from ..utils import si_format, logger, titrate
from .simulators import PWSimulator


class PointNeuron(metaclass=abc.ABCMeta):
    ''' Abstract class defining the common API (i.e. mandatory attributes and methods) of all
        subclasses implementing the channels mechanisms of specific point neurons.
    '''

    tscale = 'ms'  # relevant temporal scale of the model
    defvar = 'V'  # default plot variable

    def __repr__(self):
        return self.__class__.__name__

    def pprint(self):
        return '{} neuron'.format(self.__class__.__name__)

    def filecode(self, Astim, tstim, PRF, DC):
        ''' File naming convention. '''
        return 'ESTIM_{}_{}_{:.1f}mA_per_m2_{:.0f}ms{}'.format(
            self.name, 'CW' if DC == 1 else 'PW', Astim, tstim * 1e3,
            '_PRF{:.2f}Hz_DC{:.2f}%'.format(PRF, DC * 1e2) if DC < 1. else '')

    @property
    @abc.abstractmethod
    def name(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def Cm0(self):
        return 'Should never reach here'

    @property
    @abc.abstractmethod
    def Vm0(self):
        return 'Should never reach here'

    @abc.abstractmethod
    def currents(self, Vm, states):
        ''' Compute all ionic currents per unit area.

            :param Vm: membrane potential (mV)
            :states: state probabilities of the ion channels
            :return: dictionary of ionic currents per unit area (mA/m2)
        '''

    def iNet(self, Vm, states):
        ''' net membrane current

            :param Vm: membrane potential (mV)
            :states: states of ion channels gating and related variables
            :return: current per unit area (mA/m2)
        '''
        return sum(self.currents(Vm, states).values())

    def dQdt(self, Vm, states):
        ''' membrane charge density variation rate

            :param Vm: membrane potential (mV)
            :states: states of ion channels gating and related variables
            :return: variation rate (mA/m2)
        '''
        return -self.iNet(Vm, states)

    def isTitratable(self):
        ''' Simple method returning whether the neuron can be titrated (defaults to True). '''
        return True

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

    def getDesc(self):
        return inspect.getdoc(self).splitlines()[0]

    def getCurrentsNames(self):
        return list(self.currents(np.nan, [np.nan] * len(self.states)).keys())


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


    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        ''' Return a dictionary with information about all plot variables related to the neuron. '''

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
                if var not in ['Vm', 'Cai']:
                    vfunc = getattr(self, 'der{}{}'.format(var[0].upper(), var[1:]))
                    desc = cname + re.sub('^Evolution of', '', inspect.getdoc(vfunc).splitlines()[0])
                    pltvars[var] = {
                        'desc': desc,
                        'label': var,
                        'bounds': (-0.1, 1.1)
                    }

        pltvars['iNet'] = {
            'desc': inspect.getdoc(getattr(self, 'iNet')).splitlines()[0],
            'label': 'I_{net}',
            'unit': 'A/m^2',
            'factor': 1e-3,
            'func': 'iNet({0}Vm{1}, {2}{3}{4}.values.T)'.format(
                wrapleft, wrapright, wrapleft[:-1], self.states, wrapright[1:]),
            'ls': '--',
            'color': 'black'
        }

        pltvars['dQdt'] = {
            'desc': inspect.getdoc(getattr(self, 'dQdt')).splitlines()[0],
            'label': 'dQ_m/dt',
            'unit': 'A/m^2',
            'factor': 1e-3,
            'func': 'dQdt({0}Vm{1}, {2}{3}{4}.values.T)'.format(
                wrapleft, wrapright, wrapleft[:-1], self.states, wrapright[1:]),
            'ls': '--',
            'color': 'black'
        }

        for x in self.getGates():
            for rate in ['alpha', 'beta']:
                pltvars['{}{}'.format(rate, x)] = {
                    'label': '\\{}_{{{}}}'.format(rate, x),
                    'unit': 'ms^{-1}',
                    'factor': 1e-3
                }

        return pltvars

    def getRatesNames(self, states):
        return list(sum(
            [['alpha{}'.format(x.lower()), 'beta{}'.format(x.lower())] for x in states],
            []
        ))

    def Qm0(self):
        ''' Return the resting charge density (in C/m2). '''
        return self.Cm0 * self.Vm0 * 1e-3  # C/cm2


    @abc.abstractmethod
    def steadyStates(self, Vm):
        ''' Compute the steady-state values for a specific membrane potential value.

            :param Vm: membrane potential (mV)
            :return: dictionary of steady-states
        '''

    @abc.abstractmethod
    def derStates(self, Vm, states):
        ''' Compute the derivatives of channel states.

            :param Vm: membrane potential (mV)
            :states: state probabilities of the ion channels
            :return: current per unit area (mA/m2)
        '''


    @abc.abstractmethod
    def computeEffRates(self, Vm):
        ''' Get the effective rate constants of ion channels, averaged along an acoustic cycle,
            for future use in effective simulations.

            :param Vm: array of membrane potential values for an acoustic cycle (mV)
            :return: a dictionary of rate average constants (s-1)
        '''

    def interpEffRates(self, Qm, lkp, keys=None):
        ''' Interpolate effective rate constants for a given charge density using
            reference lookup vectors.

            :param Qm: membrane charge density (C/m2)
            :states: state probabilities of the ion channels
            :param lkp: dictionary of 1D vectors of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :return: dictionary of interpolated rate constants
        '''
        if keys is None:
            keys = self.rates
        return {k: np.interp(Qm, lkp['Q'], lkp[k], left=np.nan, right=np.nan) for k in keys}

    def interpVmeff(self, Qm, lkp):
        ''' Interpolate the effective membrane potential for a given charge density
            using reference lookup vectors.

            :param Qm: membrane charge density (C/m2)
            :param lkp: dictionary of 1D vectors of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :return: dictionary of interpolated rate constants
        '''
        return np.interp(Qm, lkp['Q'], lkp['V'], left=np.nan, right=np.nan)

    @abc.abstractmethod
    def derEffStates(self, Qm, states, lkp):
        ''' Compute the effective derivatives of channel states, based on
            1-dimensional linear interpolation of "effective" coefficients
            that summarize the system's behaviour over an acoustic cycle.

            :param Qm: membrane charge density (C/m2)
            :states: state probabilities of the ion channels
            :param lkp: dictionary of 1D vectors of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
        '''

    def Qbounds(self):
        ''' Determine bounds of membrane charge physiological range for a given neuron. '''
        return np.array([np.round(self.Vm0 - 25.0), 50.0]) * self.Cm0 * 1e-3  # C/m2

    def isVoltageGated(self, state):
        ''' Determine whether a given state is purely voltage-gated or not.'''
        return 'alpha{}'.format(state.lower()) in self.rates

    def getGates(self):
        ''' Retrieve the names of the neuron's states that match an ion channel gating. '''
        gates = []
        for x in self.states:
            if self.isVoltageGated(x):
                gates.append(x)
        return gates

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

    @abc.abstractmethod
    def quasiSteadyStates(self, lkp):
        ''' Compute the quasi-steady states of a neuron for a range of membrane charge densities,
            based on 1-dimensional lookups interpolated at a given sonophore diameter, US frequency,
            US amplitude and duty cycle.

            :param lkp: dictionary of 1D vectors of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :return: dictionary of quasi-steady states
        '''

    def getRates(self, Vm):
        ''' Compute the ion channels rate constants for a given membrane potential.

            :param Vm: membrane potential (mV)
            :return: a dictionary of rate constants and their values at the given potential.
        '''
        rates = {}
        for x in self.getGates():
            x = x.lower()
            alpha_str, beta_str = ['{}{}'.format(s, x.lower()) for s in ['alpha', 'beta']]
            inf_str, tau_str = ['{}inf'.format(x.lower()), 'tau{}'.format(x.lower())]
            if hasattr(self, 'alpha{}'.format(x)):
                alphax = getattr(self, alpha_str)(Vm)
                betax = getattr(self, beta_str)(Vm)
            elif hasattr(self, '{}inf'.format(x)):
                xinf = getattr(self, inf_str)(Vm)
                taux = getattr(self, tau_str)(Vm)
                alphax = xinf / taux
                betax = 1 / taux - alphax
            rates[alpha_str] = alphax
            rates[beta_str] = betax
        return rates


    def Vderivatives(self, y, t, Iinj):
        ''' Compute the derivatives of a V-cast HH system for a
            specific value of injected current.

            :param y: vector of HH system variables at time t
            :param t: time value (s, unused)
            :param Iinj: injected current (mA/m2)
            :return: vector of HH system derivatives at time t
        '''
        Vm, *states = y
        Iionic = self.iNet(Vm, states)  # mA/m2
        dVmdt = (- Iionic + Iinj) / self.Cm0  # mV/s
        dstates = self.derStates(Vm, states)
        return [dVmdt, *[dstates[k] for k in self.states]]

    def Qderivatives(self, y, t, Cm=None):
        ''' Compute the derivatives of the n-ODE HH system variables,
            based on a value of membrane capacitance.

            :param y: vector of HH system variables at time t
            :param t: specific instant in time (s)
            :param Cm: membrane capacitance (F/m2)
            :return: vector of HH system derivatives at time t
        '''
        if Cm is None:
            Cm = self.Cm0
        Qm, *states = y
        Vm = Qm / Cm * 1e3  # mV
        dQmdt = - self.iNet(Vm, states) * 1e-3  # A/m2
        dstates = self.derStates(Vm, states)
        return [dQmdt, *[dstates[k] for k in self.states]]


    def checkInputs(self, Astim, tstim, toffset, PRF, DC):
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

    def simulate(self, Astim, tstim, toffset, PRF=None, DC=1.0, dt=DT_ESTIM):
        ''' Compute solutions of a neuron's HH system for a specific set of
            electrical stimulation parameters, using a classic integration scheme.

            :param Astim: pulse amplitude (mA/m2)
            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param dt: integration time step (s)
            :return: 3-tuple with the time profile and solution matrix and a state vector
        '''

        # Check validity of stimulation parameters
        self.checkInputs(Astim, tstim, toffset, PRF, DC)

        # Set initial conditions
        steady_states = self.steadyStates(self.Vm0)
        y0 = np.array([self.Vm0, *[steady_states[k] for k in self.states]])

        # Initialize simulator and compute solution
        logger.debug('Computing solution')
        simulator = PWSimulator(
            lambda y, t: self.Vderivatives(y, t, Astim),
            lambda y, t: self.Vderivatives(y, t, 0.)
        )
        t, y, stim = simulator.compute(y0, dt, tstim, toffset, PRF, DC)

        # Return output variables
        return t, y.T, stim

    def isExcited(self, Astim, tstim, toffset, PRF, DC):
        ''' Run a simulation and determine if neuron is excited.

            :param Astim: current amplitude (mA/m2)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :return: boolean stating whether neuron is excited or not
        '''
        t, y, _ = self.simulate(Astim, tstim, toffset, PRF, DC)
        dt = t[1] - t[0]
        ipeaks, *_ = findPeaks(y[0, :], SPIKE_MIN_VAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                               SPIKE_MIN_VPROM)
        nspikes = ipeaks.size
        logger.debug('A = %sA/m2 ---> %s spike%s detected',
                     si_format(Astim * 1e-3, 2, space=' '),
                     nspikes, "s" if nspikes > 1 else "")
        return nspikes > 0

    def titrate(self, tstim, toffset, PRF=None, DC=1.0, Arange=(0., 2 * TITRATION_ESTIM_A_MAX),
                xfunc=None):
        ''' Use a binary search to determine the threshold amplitude needed
            to obtain neural excitation for a given duration, PRF and duty cycle.

            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param Arange: search interval for Astim, iteratively refined
            :return: excitation threshold amplitude (mA/m2)
        '''
        # Determine output function
        if xfunc is None:
            xfunc = self.isExcited
        return titrate(xfunc, (tstim, toffset, PRF, DC), Arange, TITRATION_ESTIM_DA_MAX)


    def logNSpikes(self, data):
        ''' Detect spikes on Qm signal. '''
        dt = np.diff(data.ix[:1, 't'].values)[0]
        ipeaks, *_ = findPeaks(
            data['Qm'].values,
            SPIKE_MIN_QAMP,
            int(np.ceil(SPIKE_MIN_DT / dt)),
            SPIKE_MIN_QPROM
        )
        nspikes = ipeaks.size
        logger.debug('{} spike{} detected'.format(nspikes, 's' if nspikes > 1 else ''))


    def run(self, tstim, toffset, PRF=None, DC=1.0, Astim=None):
        ''' Run a simulation of the point-neuron Hodgkin-Huxley system with specific parameters,
            and return output data and metadata.

            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle (-)
            :param Astim: stimulus amplitude (mA/m2)
        '''

        logger.info(
            '%s: %s @ %st = %ss (%ss offset)%s',
            self,
            'titration' if Astim is None else 'simulation',
            'A = {}A/m2, '.format(si_format(Astim, 2, space=' ')) if Astim is not None else '',
            *si_format([tstim, toffset], 1, space=' '),
            (', PRF = {}Hz, DC = {:.2f}%'.format(
                si_format(PRF, 2, space=' '), DC * 1e2) if DC < 1.0 else ''))

        if Astim is None:
            Astim = self.titrate(tstim, toffset, PRF, DC)
            if np.isnan(Astim):
                logger.error('Could not find threshold excitation amplitude')
                return None

        # Run simulation
        tstart = time.time()
        t, y, stimstate = self.simulate(Astim, tstim, toffset, PRF, DC)
        Vm, *channels = y
        tcomp = time.time() - tstart
        logger.debug('completed in %ss', si_format(tcomp, 1))

        # Store dataframe and metadata
        data = pd.DataFrame({
            't': t,
            'stimstate': stimstate,
            'Vm': Vm,
            'Qm': Vm * self.Cm0 * 1e-3
        })
        for j in range(len(self.states)):
            data[self.states[j]] = channels[j]

        meta = {
            'neuron': self.name,
            'Astim': Astim,
            'tstim': tstim,
            'toffset': toffset,
            'PRF': PRF,
            'DC': DC,
            'tcomp': tcomp
        }

        # Log number of detected spikes
        self.logNSpikes(data)

        return data, meta

    def runAndSave(self, outdir, tstim, toffset, PRF=None, DC=1.0, Astim=None):
        ''' Run a simulation of the point-neuron Hodgkin-Huxley system with specific parameters,
            and save the results in a PKL file.

            :param outdir: full path to output directory
            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle (-)
            :param Astim: stimulus amplitude (mA/m2)
        '''
        data, meta = self.run(tstim, toffset, PRF, DC, Astim)
        simcode = self.filecode(Astim, tstim, PRF, DC)
        outpath = '{}/{}.pkl'.format(outdir, simcode)
        with open(outpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': data}, fh)
        logger.debug('simulation data exported to "%s"', outpath)
        return outpath
