#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-03 11:53:04
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-13 20:08:49

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
from ..utils import si_format, logger, ESTIM_filecode
from ..batches import xlslog


class PointNeuron(metaclass=abc.ABCMeta):
    ''' Abstract class defining the common API (i.e. mandatory attributes and methods) of all
        subclasses implementing the channels mechanisms of specific point neurons.

        The mandatory attributes are:
            - **name**: a string defining the name of the mechanism.
            - **Cm0**: a float defining the membrane resting capacitance (in F/m2)
            - **Vm0**: a float defining the membrane resting potential (in mV)
            - **states_names**: a list of strings defining the names of the different state
              probabilities governing the channels behaviour (i.e. the differential HH variables).
            - **states0**: a 1D array of floats (NOT integers !!!) defining the initial values of
              the different state probabilities.
            - **coeff_names**: a list of strings defining the names of the different coefficients
              to be used in effective simulations.

        The mandatory methods are:
            - **iNet**: compute the net ionic current density (in mA/m2) across the membrane,
              given a specific membrane potential (in mV) and channel states.
            - **steadyStates**: compute the channels steady-state values for a specific membrane
              potential value (in mV).
            - **derStates**: compute the derivatives of channel states, given a specific membrane
              potential (in mV) and channel states. This method must return a list of derivatives
              ordered identically as in the states0 attribute.
            - **getEffRates**: get the effective rate constants of ion channels to be used in
              effective simulations. This method must return an array of effective rates ordered
              identically as in the coeff_names attribute.
            - **derStatesEff**: compute the effective derivatives of channel states, based on
              1-dimensional linear interpolators of "effective" coefficients. This method must
              return a list of derivatives ordered identically as in the states0 attribute.
    '''

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
        ''' Net membrane current

            :param Vm: membrane potential (mV)
            :states: state probabilities of the ion channels
            :return: current per unit area (mA/m2)
        '''
        return sum(self.currents(Vm, states).values())

    def currentToConcentrationRate(self, z_ion, depth):
        ''' Compute the conversion factor from a specific ionic current (in mA/m2)
            into a variation rate of submembrane ion concentration (in M/s).

            :param: z_ion: ion valence
            :param depth: submembrane depth (m)
            :return: time derivative of submembrane ion concentration (M/s)
        '''
        return 1e-6 / (z_ion * depth * FARADAY)


    def getPltVars(self):
        ''' Return a dictionary containing information about all plot variables
            related to the neuron (description, label, unit, factor, possible alias). '''

        all_pltvars = {
            'Qm': {
                'desc': 'charge density',
                'label': 'Q_m',
                'unit': 'nC/cm^2',
                'factor': 1e5,
                'min': -100,
                'max': 50
            },

            'Vm': {
                'desc': 'membrane potential',
                'label': 'V_m',
                'unit': 'mV',
                'factor': 1,
            },

            'ELeak': {
                'constant': 'neuron.ELeak',
                'desc': 'non-specific leakage current resting potential',
                'label': 'V_{leak}',
                'unit': 'mV',
                'factor': 1e0
            }
        }

        for cname in self.currents(np.nan, [np.nan] * len(self.states_names)).keys():
            cfunc = getattr(self, cname)
            cargs = inspect.getargspec(cfunc)[0][1:]
            all_pltvars[cname] = dict(
                desc=inspect.getdoc(cfunc).splitlines()[0],
                label='I_{{{}}}'.format(cname[1:]),
                unit='A/m^2',
                factor=1e-3,
                alias='neuron.{}({})'.format(cname, ', '.join(['df["{}"]'.format(a) for a in cargs]))
            )
            for var in cargs:
                if var not in ['Vm', 'Cai']:
                    vfunc = getattr(self, 'der{}{}'.format(var[0].upper(), var[1:]))
                    desc = cname + re.sub('^Evolution of', '', inspect.getdoc(vfunc).splitlines()[0])
                    all_pltvars[var] = dict(
                        desc=desc,
                        label=var,
                        unit=None,
                        factor=1,
                        min=-0.1,
                        max=1.1
                    )

        all_pltvars['iNet'] = dict(
            desc=inspect.getdoc(getattr(self, 'iNet')).splitlines()[0],
            label='I_{net}',
            unit='A/m^2',
            factor=1e-3,
            alias='neuron.iNet(df["Vm"], neuron_states)'
        )

        return all_pltvars


    @abc.abstractmethod
    def steadyStates(self, Vm):
        ''' Compute the channels steady-state values for a specific membrane potential value.

            :param Vm: membrane potential (mV)
            :return: array of steady-states
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
            :return: an array of rate average constants (s-1)
        '''

    @abc.abstractmethod
    def derStatesEff(self, Qm, states, interp_data):
        ''' Compute the effective derivatives of channel states, based on
            1-dimensional linear interpolation of "effective" coefficients
            that summarize the system's behaviour over an acoustic cycle.

            :param Qm: membrane charge density (C/m2)
            :states: state probabilities of the ion channels
            :param interp_data: dictionary of 1D vectors of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
        '''


    def Qbounds(self):
        ''' Determine bounds of membrane charge physiological range for a given neuron. '''
        return np.array([np.round(self.Vm0 - 25.0), 50.0]) * self.Cm0 * 1e-3  # C/m2


    def getGates(self):
        ''' Retrieve the names of the neuron's states that match an ion channel gating. '''
        gates = []
        for x in self.states_names:
            if 'alpha{}'.format(x.lower()) in self.coeff_names:
                gates.append(x)
        return gates

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
        return [dVmdt, *dstates]

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
        dQm = - self.iNet(Vm, states) * 1e-3  # A/m2
        dstates = self.derStates(Vm, states)
        return [dQm, *dstates]

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
        y0 = [self.Vm0, *self.states0]
        nvar = len(y0)

        # Initialize global arrays
        t = np.array([0.])
        states = np.array([1])
        y = np.array([y0]).T

        # Initialize pulse time and states vectors
        t_pulse0 = np.linspace(0, Tpulse_on + Tpulse_off, n_pulse_on + n_pulse_off)
        states_pulse = np.concatenate((np.ones(n_pulse_on), np.zeros(n_pulse_off)))

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
            states = np.concatenate([states, states_pulse[1:]])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)

        # Integrate offset interval
        if n_off > 0:
            t_off = np.linspace(0, toffset, n_off) + t[-1]
            states_off = np.zeros(n_off)
            y_off = odeint(self.Vderivatives, y[:, -1], t_off, args=(0.0, )).T

            # Concatenate offset arrays to global arrays
            states = np.concatenate([states, states_off[1:]])
            t = np.concatenate([t, t_off[1:]])
            y = np.concatenate([y, y_off[:, 1:]], axis=1)

        # Return output variables
        return (t, y, states)

    def titrate(self, tstim, toffset, PRF=None, DC=1.0, Arange=(0., 2 * TITRATION_ESTIM_A_MAX)):
        ''' Use a dichotomic recursive search to determine the threshold amplitude needed
            to obtain neural excitation for a given duration, PRF and duty cycle.

            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param Arange: search interval for Astim, iteratively refined
            :return: 5-tuple with the determined threshold, time profile,
                 solution matrix, state vector and response latency
        '''
        Astim = (Arange[0] + Arange[1]) / 2

        # Run simulation and detect spikes
        t0 = time.time()
        (t, y, states) = self.simulate(Astim, tstim, toffset, PRF, DC)
        tcomp = time.time() - t0
        dt = t[1] - t[0]
        ipeaks, *_ = findPeaks(y[0, :], SPIKE_MIN_VAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                               SPIKE_MIN_VPROM)
        nspikes = ipeaks.size
        latency = t[ipeaks[0]] if nspikes > 0 else None
        logger.debug('A = %sA/m2 ---> %s spike%s detected',
                     si_format(Astim * 1e-3, 2, space=' '),
                     nspikes, "s" if nspikes > 1 else "")

        # If accurate threshold is found, return simulation results
        if (Arange[1] - Arange[0]) <= TITRATION_ESTIM_DA_MAX and nspikes == 1:
            return (Astim, t, y, states, latency, tcomp)

        # Otherwise, refine titration interval and iterate recursively
        else:
            if nspikes == 0:
                # if Astim too close to max then stop
                if (TITRATION_ESTIM_A_MAX - Astim) <= TITRATION_ESTIM_DA_MAX:
                    return (np.nan, t, y, states, latency, tcomp)
                Arange = (Astim, Arange[1])
            else:
                Arange = (Arange[0], Astim)
            return self.titrate(tstim, toffset, PRF, DC, Arange=Arange)

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

        if Astim is not None:
            logger.info('%s: simulation @ A = %sA/m2, t = %ss (%ss offset)%s',
                        self, si_format(Astim * 1e-3, 2, space=' '),
                        *si_format([tstim, toffset], 1, space=' '),
                        (', PRF = {}Hz, DC = {:.2f}%'.format(si_format(PRF, 2, space=' '), DC * 1e2)
                         if DC < 1.0 else ''))

            # Run simulation
            tstart = time.time()
            t, y, states = self.simulate(Astim, tstim, toffset, PRF, DC)
            Vm, *channels = y
            tcomp = time.time() - tstart

            # Detect spikes on Vm signal
            dt = t[1] - t[0]
            ipeaks, *_ = findPeaks(Vm, SPIKE_MIN_VAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                                   SPIKE_MIN_VPROM)
            nspikes = ipeaks.size
            lat = t[ipeaks[0]] if nspikes > 0 else 'N/A'
            outstr = '{} spike{} detected'.format(nspikes, 's' if nspikes > 1 else '')
        else:
            logger.info('%s: titration @ t = %ss%s',
                        self, si_format(tstim, 1, space=' '),
                        (', PRF = {}Hz, DC = {:.2f}%'.format(si_format(PRF, 2, space=' '), DC * 1e2)
                         if DC < 1.0 else ''))

            # Run titration
            Astim, t, y, states, lat, tcomp = self.titrate(tstim, toffset, PRF, DC)
            Vm, *channels = y
            nspikes = 1
            if Astim is np.nan:
                outstr = 'no spikes detected within titration interval'
                nspikes = 0
            else:
                nspikes = 1
                outstr = 'Athr = {}A/m2'.format(si_format(Astim * 1e-3, 2, space=' '))
        logger.debug('completed in %s, %s', si_format(tcomp, 1), outstr)
        sr = np.mean(1 / np.diff(t[ipeaks])) if nspikes > 1 else None

        # Store dataframe and metadata
        df = pd.DataFrame({
            't': t,
            'states': states,
            'Vm': Vm,
            'Qm': Vm * self.Cm0 * 1e-3
        })
        for j in range(len(self.states_names)):
            df[self.states_names[j]] = channels[j]

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

    def findRheobaseAmps(self, DCs, Vthr, curr='net'):
        ''' Find the rheobase amplitudes (i.e. threshold amplitudes of infinite duration
            that would result in excitation) of a specific neuron for various stimulation duty cycles.

            :param DCs: duty cycles vector (-)
            :param Vthr: threshold membrane potential above which the neuron necessarily fires (mV)
            :return: rheobase amplitudes vector (mA/m2)
        '''

        # Compute the pulse average net (or leakage) current along the amplitude space
        if curr == 'net':
            iNet = self.iNet(Vthr, self.steadyStates(Vthr))
        elif curr == 'leak':
            iNet = self.iLeak(Vthr)

        # Compute rheobase amplitudes
        return iNet / np.array(DCs)
