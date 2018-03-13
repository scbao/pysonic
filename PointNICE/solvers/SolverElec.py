#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-13 14:30:51


import warnings
import logging
import numpy as np
import scipy.integrate as integrate

from ..constants import *

# Get package logger
logger = logging.getLogger('PointNICE')


class SolverElec:

    def __init__(self):
        # Do nothing
        pass


    def eqHH(self, _, y, neuron, Iinj):
        ''' Compute the derivatives of a HH system variables for a
            specific value of injected current.

            :param t: time value (s, unused)
            :param y: vector of HH system variables at time t
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
        for param in [Astim, tstim, toffset, DC]:
            assert isinstance(param, float), 'stimulation parameters must be float typed'
        assert tstim > 0, 'Stimulus duration must be strictly positive'
        assert toffset >= 0, 'Stimulus offset must be positive or null'
        assert DC > 0 and DC <= 1, 'Duty cycle must be within [0; 1)'
        if DC < 1.0:
            assert isinstance(PRF, float), 'if provided, the PRF parameter must be float typed'
            assert PRF is not None, 'PRF must be provided when using duty cycles smaller than 1'
            assert PRF >= 1 / tstim, 'PR interval must be smaller than stimulus duration'

        # Raise warnings as error
        warnings.filterwarnings('error')

        # Initialize system solver
        solver = integrate.ode(self.eqHH)
        solver.set_integrator('lsoda', nsteps=1000)

        # Determine system time step
        dt = DT_ESTIM

        # if CW stimulus: divide integration during stimulus into single interval
        if DC == 1.0:
            PRF = 1 / tstim

        # Compute vector sizes
        npulses = int(np.round(PRF * tstim))
        Tpulse_on = DC / PRF
        Tpulse_off = (1 - DC) / PRF
        n_pulse_on = int(np.round(Tpulse_on / dt))
        n_pulse_off = int(np.round(Tpulse_off / dt))

        # For high-PRF pulsed protocols: adapt time step if greater than TON or TOFF
        dt_warning_msg = 'high-PRF protocol: lowering integration time step to %.2e ms to match %s'
        if Tpulse_on > 0 and n_pulse_on == 0:
            logger.warning(dt_warning_msg, Tpulse_on * 1e3, 'TON')
            dt = Tpulse_on
            n_pulse_on = int(np.round(Tpulse_on / dt))
            n_pulse_off = int(np.round(Tpulse_off / dt))
        if Tpulse_off > 0 and n_pulse_off == 0:
            logger.warning(dt_warning_msg, Tpulse_off * 1e3, 'TOFF')
            dt = Tpulse_off
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
            y_pulse[:, 0] = y[:, -1]

            # Initialize iterator
            k = 0

            # Integrate ON system
            solver.set_f_params(neuron, Astim)
            solver.set_initial_value(y_pulse[:, k], t_pulse[k])
            while solver.successful() and k < n_pulse_on - 1:
                k += 1
                solver.integrate(t_pulse[k])
                y_pulse[:, k] = solver.y

            # Integrate OFF system
            solver.set_f_params(neuron, 0.0)
            solver.set_initial_value(y_pulse[:, k], t_pulse[k])
            while solver.successful() and k < n_pulse_on + n_pulse_off - 1:
                k += 1
                solver.integrate(t_pulse[k])
                y_pulse[:, k] = solver.y

            # Append pulse arrays to global arrays
            states = np.concatenate([states, states_pulse[1:]])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)

        # Integrate offset interval
        if n_off > 0:
            t_off = np.linspace(0, toffset, n_off) + t[-1]
            states_off = np.zeros(n_off)
            y_off = np.empty((nvar, n_off))
            y_off[:, 0] = y[:, -1]
            solver.set_initial_value(y_off[:, 0], t_off[0])
            solver.set_f_params(neuron, 0.0)
            k = 0
            while solver.successful() and k < n_off - 1:
                k += 1
                solver.integrate(t_off[k])
                y_off[:, k] = solver.y


            # Concatenate offset arrays to global arrays
            states = np.concatenate([states, states_off[1:]])
            t = np.concatenate([t, t_off[1:]])
            y = np.concatenate([y, y_off[:, 1:]], axis=1)

        # Return output variables
        return (t, y, states)
