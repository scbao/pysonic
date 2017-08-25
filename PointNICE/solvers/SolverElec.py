#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-25 10:35:53


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
        logger.info('Elec solver initialization')


    def eqHH(self, _, y, channel_mech, Iinj):
        ''' Compute the derivatives of a HH system variables for a
            specific value of injected current.

            :param t: time value (s, unused)
            :param y: vector of HH system variables at time t
            :param channel_mech: channels mechanism object
            :param Iinj: injected current (mA/m2)
            :return: vector of HH system derivatives at time t
        '''

        Vm, *states = y
        Iionic = channel_mech.currNet(Vm, states)  # mA/m2
        dVmdt = (- Iionic + Iinj) / channel_mech.Cm0  # mV/s
        dstates = channel_mech.derStates(Vm, states)
        return [dVmdt, *dstates]


    def run(self, channel_mech, Astim, tstim, toffset, PRF, DF):
        ''' Compute solutions of a neuron's HH system for a specific set of
            electrical stimulation parameters, using a classic integration scheme.

            :param channel_mech: channels mechanism object
            :param Astim: pulse amplitude (mA/m2)
            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DF: pulse duty factor (-)
            :return: 3-tuple with the time profile and solution matrix and a state vector
        '''

        # Raise warnings as error
        warnings.filterwarnings('error')

        # Initialize system solver
        solver = integrate.ode(self.eqHH)
        solver.set_integrator('lsoda', nsteps=1000)

        # Determine system time step
        dt = DT_ESTIM

        # Determine proportion of tstim in total integration
        stim_prop = tstim / (tstim + toffset)

        # if CW stimulus: divide integration during stimulus into single interval
        if DF == 1.0:
            PRF = 1 / tstim

        # Compute vector sizes
        npulses = int(np.round(PRF * tstim))
        Tpulse_on = DF / PRF
        Tpulse_off = (1 - DF) / PRF
        n_pulse_on = int(np.round(Tpulse_on / dt))
        n_pulse_off = int(np.round(Tpulse_off / dt))
        n_off = int(np.round(toffset / dt))

        # Set initial conditions
        y0 = [channel_mech.Vm0, *channel_mech.states0]
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
            solver.set_f_params(channel_mech, Astim)
            solver.set_initial_value(y_pulse[:, k], t_pulse[k])
            while solver.successful() and k < n_pulse_on - 1:
                k += 1
                solver.integrate(t_pulse[k])
                y_pulse[:, k] = solver.y

            # Integrate OFF system
            solver.set_f_params(channel_mech, 0.0)
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
        t_off = np.linspace(0, toffset, n_off) + t[-1]
        states_off = np.zeros(n_off)
        y_off = np.empty((nvar, n_off))
        y_off[:, 0] = y[:, -1]
        solver.set_initial_value(y_off[:, 0], t_off[0])
        solver.set_f_params(channel_mech, 0.0)
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
