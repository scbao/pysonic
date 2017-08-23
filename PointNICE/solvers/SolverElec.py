#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-03 15:09:53


import logging
import numpy as np
import scipy.integrate as integrate

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


    def runSim(self, channel_mech, Astim, tstim, toffset, tonset=10e-3):
        ''' Compute solutions of a neuron's HH system for a specific set of
            electrical stimulation parameters, using a classic integration scheme.

            :param channel_mech: channels mechanism object
            :param Astim: pulse amplitude (mA/m2)
            :param tstim: pulse duration (s)
            :param toffset: offset duration (s)
            :param tonset: onset duration (s)
            :return: 2-tuple with the time profile and solution matrix
        '''

        # Set time vector
        ttot = tonset + tstim + toffset
        dt = 1e-4  # s
        nsamples = int(np.round(ttot / dt))
        t = np.linspace(0.0, ttot, nsamples) - tonset

        # Set pulse vector
        n_onset = int(np.round(tonset / dt))
        n_stim = int(np.round(tstim / dt))
        n_offset = int(np.round(toffset / dt))
        pulse = np.concatenate((np.zeros(n_onset), Astim * np.ones(n_stim), np.zeros(n_offset)))

        # Create solver
        solver = integrate.ode(self.eqHH)
        solver.set_integrator('lsoda', nsteps=1000)

        # Set initial conditions
        y0 = [channel_mech.Vm0, *channel_mech.states0]
        nvar = len(y0)

        # Run simulation
        y = np.empty((nsamples - 1, nvar))
        solver.set_initial_value(y0, t[0])
        k = 1
        while solver.successful() and k <= nsamples - 1:
            solver.set_f_params(channel_mech, pulse[k])
            solver.integrate(t[k])
            y[k - 1, :] = solver.y
            k += 1

        y = np.concatenate((np.atleast_2d(y0), y), axis=0)
        return (t, y)


    def eqHH_VClamp(self, _, y, channel_mech, Vc):
        ''' Compute the derivatives of a HH system variables for a
            specific value of clamped voltage.

            :param t: time value (s, unused)
            :param y: vector of HH system variables at time t
            :param channel_mech: channels mechanism object
            :param Vc: clamped voltage (mV)
            :return: vector of HH system derivatives at time t
        '''

        return channel_mech.derStates(Vc, y)



    def runVClamp(self, channel_mech, Vclamp, tclamp, toffset, tonset=10e-3):
        ''' Compute solutions of a neuron's HH system for a specific set of
            voltage clamp parameters, using a classic integration scheme.

            :param channel_mech: channels mechanism object
            :param Vclamp: clamped voltage (mV)
            :param toffset: offset duration (s)
            :param tclamp: clamp duration (s)
            :param tonset: onset duration (s)
            :return: 2-tuple with the time profile and solution matrix
        '''

        # Set time vector
        ttot = tonset + tclamp + toffset
        dt = 1e-4  # s
        nsamples = int(np.round(ttot / dt))
        t = np.linspace(0.0, ttot, nsamples) - tonset

        # Set clamp vector
        n_onset = int(np.round(tonset / dt))
        n_clamp = int(np.round(tclamp / dt))
        n_offset = int(np.round(toffset / dt))
        clamp = np.concatenate((np.zeros(n_onset), Vclamp * np.ones(n_clamp), np.zeros(n_offset)))

        # Create solver
        solver = integrate.ode(self.eqHH_VClamp)
        solver.set_integrator('lsoda', nsteps=1000)

        # Set initial conditions
        y0 = channel_mech.states0
        nvar = len(y0)

        # Run simulation
        y = np.empty((nsamples - 1, nvar))
        solver.set_initial_value(y0, t[0])
        k = 1
        while solver.successful() and k <= nsamples - 1:
            solver.set_f_params(channel_mech, clamp[k])
            solver.integrate(t[k])
            y[k - 1, :] = solver.y
            k += 1

        y = np.concatenate((np.atleast_2d(y0), y), axis=0)
        return (t, y)
