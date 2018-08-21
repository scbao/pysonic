#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-21 16:10:37


import warnings
import logging
import numpy as np
import scipy.integrate as integrate

from ..constants import *
from ..neurons import BaseMech
from ..utils import InputError

# Get package logger
logger = logging.getLogger('PySONIC')


class SolverElec:

    def __init__(self):
        # Do nothing
        pass


    def eqHH(self, y, _, neuron, Iinj):
        ''' Compute the derivatives of a HH system variables for a
            specific value of injected current.

            :param y: vector of HH system variables at time t
            :param t: time value (s, unused)
            :param neuron: neuron object
            :param Iinj: injected current (mA/m2)
            :return: vector of HH system derivatives at time t
        '''

        Vm, *states = y
        Iionic = neuron.currNet(Vm, states)  # mA/m2
        dVmdt = (- Iionic + Iinj) / neuron.Cm0  # mV/s
        dstates = neuron.derStates(Vm, states)
        return [dVmdt, *dstates]


    def run(self, neuron, Astim, tstim, toffset, PRF=None, DC=1.0):
        ''' Compute solutions of a neuron's HH system for a specific set of
            electrical stimulation parameters, using a classic integration scheme.

            :param neuron: neuron object
            :param Astim: pulse amplitude (mA/m2)
            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :return: 3-tuple with the time profile and solution matrix and a state vector
        '''

        # Check validity of stimulation parameters
        if not isinstance(neuron, BaseMech):
            raise InputError('Invalid neuron type: "{}" (must inherit from BaseMech class)'
                             .format(neuron.name))
        if not all(isinstance(param, float) for param in [Astim, tstim, toffset, DC]):
            raise InputError('Invalid stimulation parameters (must be float typed)')
        if tstim <= 0:
            raise InputError('Invalid stimulus duration: {} ms (must be strictly positive)'
                             .format(tstim * 1e3))
        if toffset < 0:
            raise InputError('Invalid stimulus offset: {} ms (must be positive or null)'
                             .format(toffset * 1e3))
        if DC <= 0.0 or DC > 1.0:
            raise InputError('Invalid duty cycle: {} (must be within ]0; 1])'.format(DC))
        if DC < 1.0:
            if not isinstance(PRF, float):
                raise InputError('Invalid PRF value (must be float typed)')
            if PRF is None:
                raise InputError('Missing PRF value (must be provided when DC < 1)')
            if PRF < 1 / tstim:
                raise InputError('Invalid PRF: {} Hz (PR interval exceeds stimulus duration)'
                                 .format(PRF))

        # Raise warnings as error
        warnings.filterwarnings('error')

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
        y0 = [neuron.Vm0, *neuron.states0]
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
            y_pulse[:, :n_pulse_on] = integrate.odeint(self.eqHH, y[:, -1], t_pulse[:n_pulse_on],
                                                       args=(neuron, Astim)).T

            # Integrate OFF system
            if n_pulse_off > 0:
                y_pulse[:, n_pulse_on:] = integrate.odeint(self.eqHH, y_pulse[:, n_pulse_on - 1],
                                                           t_pulse[n_pulse_on:],
                                                           args=(neuron, 0.0)).T

            # Append pulse arrays to global arrays
            states = np.concatenate([states, states_pulse[1:]])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)

        # Integrate offset interval
        if n_off > 0:
            t_off = np.linspace(0, toffset, n_off) + t[-1]
            states_off = np.zeros(n_off)
            y_off = integrate.odeint(self.eqHH, y[:, -1], t_off, args=(neuron, 0.0)).T

            # Concatenate offset arrays to global arrays
            states = np.concatenate([states, states_off[1:]])
            t = np.concatenate([t, t_off[1:]])
            y = np.concatenate([y, y_off[:, 1:]], axis=1)

        # Return output variables
        return (t, y, states)
