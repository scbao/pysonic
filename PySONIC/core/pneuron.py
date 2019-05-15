#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-03 11:53:04
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-15 14:01:49

import os
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
from ..utils import si_format, logger, ESTIM_filecode, titrate, resolveDependencies
from ..batches import xlslog


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
    def getEffRates(self, Vm):
        ''' Get the effective rate constants of ion channels, averaged along an acoustic cycle,
            for future use in effective simulations.

            :param Vm: array of membrane potential values for an acoustic cycle (mV)
            :return: a dictionary of rate average constants (s-1)
        '''

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

    def simulate(self, Astim, tstim, toffset, PRF=None, DC=1.0):
        ''' Compute solutions of a neuron's HH system for a specific set of
            electrical stimulation parameters, using a classic integration scheme.

            :param Astim: pulse amplitude (mA/m2)
            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :return: 3-tuple with the time profile and solution matrix and a state vector
        '''

        # Check validity of stimulation parameters
        self.checkInputs(Astim, tstim, toffset, PRF, DC)

        # Determine system time step
        dt = DT_ESTIM

        # if CW stimulus: divide integration during stimulus into single interval
        if DC == 1.0:
            PRF = 1 / tstim

        # Compute vector sizes
        npulses = int(np.round(PRF * tstim))
        Tpulse_on = DC / PRF
        Tpulse_off = (1 - DC) / PRF

        # For high-PRF pulsed protocols: adapt time step to ensure minimal
        # number of samples during TON or TOFF
        dt_warning_msg = 'high-PRF protocol: lowering time step to %.2e s to properly integrate %s'
        for key, Tpulse in {'TON': Tpulse_on, 'TOFF': Tpulse_off}.items():
            if Tpulse > 0 and Tpulse / dt < MIN_SAMPLES_PER_PULSE_INT:
                dt = Tpulse / MIN_SAMPLES_PER_PULSE_INT
                logger.warning(dt_warning_msg, dt, key)

        n_pulse_on = int(np.round(Tpulse_on / dt))
        n_pulse_off = int(np.round(Tpulse_off / dt))

        # Compute offset size
        n_off = int(np.round(toffset / dt))

        # Set initial conditions
        steady_states = self.steadyStates(self.Vm0)
        y0 = [self.Vm0, *[steady_states[k] for k in self.states]]
        nvar = len(y0)

        # Initialize global arrays
        t = np.array([0.])
        stimstate = np.array([1])
        y = np.array([y0]).T

        # Initialize pulse time and stimstate vectors
        t_pulse0 = np.linspace(0, Tpulse_on + Tpulse_off, n_pulse_on + n_pulse_off)
        stimstate_pulse = np.concatenate((np.ones(n_pulse_on), np.zeros(n_pulse_off)))

        # Loop through all pulse (ON and OFF) intervals
        for i in range(npulses):

            # Construct and initialize arrays
            t_pulse = t_pulse0 + t[-1]
            y_pulse = np.empty((nvar, n_pulse_on + n_pulse_off))

            # Integrate ON system
            y_pulse[:, :n_pulse_on] = odeint(
                self.Vderivatives, y[:, -1], t_pulse[:n_pulse_on], args=(Astim,)).T

            # Integrate OFF system
            if n_pulse_off > 0:
                y_pulse[:, n_pulse_on:] = odeint(
                    self.Vderivatives, y_pulse[:, n_pulse_on - 1], t_pulse[n_pulse_on:],
                    args=(0.0,)).T

            # Append pulse arrays to global arrays
            stimstate = np.concatenate([stimstate, stimstate_pulse[1:]])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)

        # Integrate offset interval
        if n_off > 0:
            t_off = np.linspace(0, toffset, n_off) + t[-1]
            stimstate_off = np.zeros(n_off)
            y_off = odeint(self.Vderivatives, y[:, -1], t_off, args=(0.0, )).T

            # Concatenate offset arrays to global arrays
            stimstate = np.concatenate([stimstate, stimstate_off[1:]])
            t = np.concatenate([t, t_off[1:]])
            y = np.concatenate([y, y_off[:, 1:]], axis=1)

        # Return output variables
        return (t, y, stimstate)

    def nSpikes(self, Astim, tstim, toffset, PRF, DC):
        ''' Run a simulation and determine number of spikes in the response.

            :param Astim: current amplitude (mA/m2)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :return: number of spikes found in response
        '''
        t, y, _ = self.simulate(Astim, tstim, toffset, PRF, DC)
        dt = t[1] - t[0]
        ipeaks, *_ = findPeaks(y[0, :], SPIKE_MIN_VAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                               SPIKE_MIN_VPROM)
        nspikes = ipeaks.size
        logger.debug('A = %sA/m2 ---> %s spike%s detected',
                     si_format(Astim * 1e-3, 2, space=' '),
                     nspikes, "s" if nspikes > 1 else "")
        return nspikes

    def titrate(self, tstim, toffset, PRF=None, DC=1.0, Arange=(0., 2 * TITRATION_ESTIM_A_MAX)):
        ''' Use a binary search to determine the threshold amplitude needed
            to obtain neural excitation for a given duration, PRF and duty cycle.

            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param Arange: search interval for Astim, iteratively refined
            :return: excitation threshold amplitude (mA/m2)
        '''
        return titrate(self.nSpikes, (tstim, toffset, PRF, DC), Arange, TITRATION_ESTIM_DA_MAX)

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

        # Get date and time info
        date_str = time.strftime("%Y.%m.%d")
        daytime_str = time.strftime("%H:%M:%S")

        logger.info(
            '%s: %s @ %st = %ss (%ss offset)%s',
            self,
            'titration' if Astim is None else 'simulation',
            'A = {}A/m2, '.format(si_format(Astim, 2, space=' ')) if Astim is not None else '',
            *si_format([tstim, toffset], 1, space=' '),
            (', PRF = {}Hz, DC = {:.2f}%'.format(si_format(PRF, 2, space=' '), DC * 1e2) if DC < 1.0 else ''))

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

        # Detect spikes on Vm signal
        dt = t[1] - t[0]
        ipeaks, *_ = findPeaks(Vm, SPIKE_MIN_VAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                               SPIKE_MIN_VPROM)
        nspikes = ipeaks.size
        lat = t[ipeaks[0]] if nspikes > 0 else 'N/A'
        outstr = '{} spike{} detected'.format(nspikes, 's' if nspikes > 1 else '')
        logger.debug('completed in %ss, %s', si_format(tcomp, 1), outstr)
        sr = np.mean(1 / np.diff(t[ipeaks])) if nspikes > 1 else None

        # Store dataframe and metadata
        df = pd.DataFrame({
            't': t,
            'stimstate': stimstate,
            'Vm': Vm,
            'Qm': Vm * self.Cm0 * 1e-3
        })
        for j in range(len(self.states)):
            df[self.states[j]] = channels[j]

        meta = {
            'neuron': self.name,
            'Astim': Astim,
            'tstim': tstim,
            'toffset': toffset,
            'PRF': PRF,
            'DC': DC,
            'tcomp': tcomp
        }

        # Export into to PKL file
        simcode = ESTIM_filecode(self.name, Astim, tstim, PRF, DC)
        outpath = '{}/{}.pkl'.format(outdir, simcode)
        with open(outpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': df}, fh)
        logger.debug('simulation data exported to "%s"', outpath)

        # Export key metrics to log file
        logpath = os.path.join(outdir, 'log_ESTIM.xlsx')
        logentry = {
            'Date': date_str,
            'Time': daytime_str,
            'Neuron Type': self.name,
            'Astim (mA/m2)': Astim,
            'Tstim (ms)': tstim * 1e3,
            'PRF (kHz)': PRF * 1e-3 if DC < 1 else 'N/A',
            'Duty factor': DC,
            '# samples': t.size,
            'Comp. time (s)': round(tcomp, 2),
            '# spikes': nspikes,
            'Latency (ms)': lat * 1e3 if isinstance(lat, float) else 'N/A',
            'Spike rate (sp/ms)': sr * 1e-3 if isinstance(sr, float) else 'N/A'
        }

        if xlslog(logpath, logentry) == 1:
            logger.debug('log exported to "%s"', logpath)
        else:
            logger.error('log export to "%s" aborted', self.logpath)

        return outpath
