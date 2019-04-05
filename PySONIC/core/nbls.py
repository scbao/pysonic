#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-04-05 21:01:36

import os
import inspect
import time
import logging
import pickle
import progressbar as pb
import numpy as np
import pandas as pd
from scipy.integrate import ode, odeint
from scipy.interpolate import interp1d

from .bls import BilayerSonophore
from .pneuron import PointNeuron
from ..utils import logger, si_format, downsample, rmse, ASTIM_filecode, getLookups2D, isWithin, titrate
from ..constants import *
from ..postpro import findPeaks
from ..batches import xlslog


class NeuronalBilayerSonophore(BilayerSonophore):
    ''' This class inherits from the BilayerSonophore class and receives an PointNeuron instance
        at initialization, to define the electro-mechanical NICE model and its SONIC variant. '''

    tscale = 'ms'  # relevant temporal scale of the model
    defvar = 'Q'  # default plot variable

    def __init__(self, a, neuron, Fdrive=None, embedding_depth=0.0):
        ''' Constructor of the class.

            :param a: in-plane radius of the sonophore structure within the membrane (m)
            :param neuron: neuron object
            :param Fdrive: frequency of acoustic perturbation (Hz)
            :param embedding_depth: depth of the embedding tissue around the membrane (m)
        '''

        # Check validity of input parameters
        if not isinstance(neuron, PointNeuron):
            raise ValueError('Invalid neuron type: "{}" (must inherit from PointNeuron class)'
                             .format(neuron.name))
        self.neuron = neuron

        # Initialize BilayerSonophore parent object
        BilayerSonophore.__init__(self, a, neuron.Cm0, neuron.Cm0 * neuron.Vm0 * 1e-3,
                                  embedding_depth)

    def __repr__(self):
        return 'NeuronalBilayerSonophore({}m, {})'.format(
            si_format(self.a, precision=1, space=' '),
            self.neuron)

    def pprint(self):
        return '{}m radius NBLS - {} neuron'.format(
            si_format(self.a, precision=0, space=' '),
            self.neuron.name)

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        pltvars = super().getPltVars(wrapleft, wrapright)
        pltvars.update(self.neuron.getPltVars(wrapleft, wrapright))
        return pltvars

    def getPltScheme(self):
        return self.neuron.getPltScheme()

    def fullDerivatives(self, y, t, Adrive, Fdrive, phi):
        ''' Compute the derivatives of the (n+3) ODE full NBLS system variables.

            :param y: vector of state variables
            :param t: specific instant in time (s)
            :param Adrive: acoustic drive amplitude (Pa)
            :param Fdrive: acoustic drive frequency (Hz)
            :param phi: acoustic drive phase (rad)
            :return: vector of derivatives
        '''
        dydt_mech = BilayerSonophore.derivatives(self, y[:3], t, Adrive, Fdrive, y[3], phi)
        dydt_elec = self.neuron.Qderivatives(y[3:], t, self.Capct(y[1]))
        return dydt_mech + dydt_elec


    def effDerivatives(self, y, t, interp_data):
        ''' Compute the derivatives of the n-ODE effective HH system variables,
            based on 1-dimensional linear interpolation of "effective" coefficients
            that summarize the system's behaviour over an acoustic cycle.

            :param y: vector of HH system variables at time t
            :param t: specific instant in time (s)
            :param interp_data: dictionary of 1D data points of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :return: vector of effective system derivatives at time t
        '''

        # Split input vector explicitly
        Qm, *states = y

        # Compute charge and channel states variation
        Vm = np.interp(Qm, interp_data['Q'], interp_data['V'])  # mV
        dQmdt = - self.neuron.iNet(Vm, states) * 1e-3
        dstates = self.neuron.derStatesEff(Qm, states, interp_data)

        # Return derivatives vector
        return [dQmdt, *dstates]


    def runFull(self, Fdrive, Adrive, tstim, toffset, PRF, DC, phi=np.pi):
        ''' Compute solutions of the full electro-mechanical system for a specific set of
            US stimulation parameters, using a classic integration scheme.

            The first iteration uses the quasi-steady simplification to compute
            the initiation of motion from a flat leaflet configuration. Afterwards,
            the ODE system is solved iteratively until completion.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param phi: acoustic drive phase (rad)
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        '''

        # Determine system time step
        Tdrive = 1 / Fdrive
        dt = Tdrive / NPC_FULL

        # if CW stimulus: divide integration during stimulus into 100 intervals
        if DC == 1.0:
            PRF = 100 / tstim

        # Compute vector sizes
        npulses = int(np.round(PRF * tstim))
        Tpulse_on = DC / PRF
        Tpulse_off = (1 - DC) / PRF
        n_pulse_on = int(np.round(Tpulse_on / dt))
        n_pulse_off = int(np.round(Tpulse_off / dt))
        n_off = int(np.round(toffset / dt))

        # Solve quasi-steady equation to compute first deflection value
        Z0 = 0.0
        ng0 = self.ng0
        Qm0 = self.Qm0
        Pac1 = self.Pacoustic(dt, Adrive, Fdrive, phi)
        Z1 = self.balancedefQS(ng0, Qm0, Pac1)

        # Initialize global arrays
        stimstate = np.array([1, 1])
        t = np.array([0., dt])
        y_membrane = np.array([[0., (Z1 - Z0) / dt], [Z0, Z1], [ng0, ng0], [Qm0, Qm0]])
        y_channels = np.tile(self.neuron.steadyStates(self.neuron.Vm0), (2, 1)).T
        y = np.vstack((y_membrane, y_channels))
        nvar = y.shape[0]

        # Initialize pulse time and stimstate vectors
        t_pulse0 = np.linspace(0, Tpulse_on + Tpulse_off, n_pulse_on + n_pulse_off)
        stimstate_pulse = np.concatenate((np.ones(n_pulse_on), np.zeros(n_pulse_off)))

        logger.debug('Computing detailed solution')

        # Initialize progress bar
        if logger.getEffectiveLevel() <= logging.INFO:
            widgets = ['Running: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
            pbar = pb.ProgressBar(widgets=widgets,
                                  max_value=int(npulses * (toffset + tstim) / tstim))
            pbar.start()

        # Loop through all pulse (ON and OFF) intervals
        for i in range(npulses):

            # Construct and initialize arrays
            t_pulse = t_pulse0 + t[-1]
            y_pulse = np.empty((nvar, n_pulse_on + n_pulse_off))

            # Integrate ON system
            y_pulse[:, :n_pulse_on] = odeint(
                self.fullDerivatives, y[:, -1], t_pulse[:n_pulse_on],
                args=(Adrive, Fdrive, phi)).T

            # Integrate OFF system
            if n_pulse_off > 0:
                y_pulse[:, n_pulse_on:] = odeint(
                    self.fullDerivatives, y_pulse[:, n_pulse_on - 1], t_pulse[n_pulse_on:],
                    args=(0.0, 0.0, 0.0)).T

            # Append pulse arrays to global arrays
            stimstate = np.concatenate([stimstate, stimstate_pulse[1:]])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)

            # Update progress bar
            if logger.getEffectiveLevel() <= logging.INFO:
                pbar.update(i)

        # Integrate offset interval
        if n_off > 0:
            t_off = np.linspace(0, toffset, n_off) + t[-1]
            stimstate_off = np.zeros(n_off)
            y_off = odeint(self.fullDerivatives, y[:, -1], t_off, args=(0.0, 0.0, 0.0)).T

            # Concatenate offset arrays to global arrays
            stimstate = np.concatenate([stimstate, stimstate_off[1:]])
            t = np.concatenate([t, t_off[1:]])
            y = np.concatenate([y, y_off[:, 1:]], axis=1)

        # Terminate progress bar
        if logger.getEffectiveLevel() <= logging.INFO:
            pbar.finish()

        # Downsample arrays in time-domain according to target temporal resolution
        ds_factor = int(np.round(CLASSIC_TARGET_DT / dt))
        if ds_factor > 1:
            Fs = 1 / (dt * ds_factor)
            logger.info('Downsampling output arrays by factor %u (Fs = %.2f MHz)',
                        ds_factor, Fs * 1e-6)
            t = t[::ds_factor]
            y = y[:, ::ds_factor]
            stimstate = stimstate[::ds_factor]

        # Compute membrane potential vector (in mV)
        Vm = y[3, :] / self.v_Capct(y[1, :]) * 1e3  # mV

        # Return output variables with Vm
        return (t, np.vstack([y[1:4, :], Vm, y[4:, :]]), stimstate)


    def runSONIC(self, Fdrive, Adrive, tstim, toffset, PRF, DC, dt=DT_EFF):
        ''' Compute solutions of the system for a specific set of
            US stimulation parameters, using charge-predicted "effective"
            coefficients to solve the HH equations at each step.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param dt: integration time step (s)
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        '''

        # Load appropriate 2D lookups
        Aref, Qref, lookups2D, _ = getLookups2D(self.neuron.name, a=self.a, Fdrive=Fdrive)

        # Check that acoustic amplitude is within lookup range
        Adrive = isWithin('amplitude', Adrive, (Aref.min(), Aref.max()))

        # Interpolate 2D lookups at zero and US amplitude
        logger.debug('Interpolating lookups at A = %.2f kPa and A = 0', Adrive * 1e-3)
        lookups_on = {key: interp1d(Aref, y2D, axis=0)(Adrive) for key, y2D in lookups2D.items()}
        lookups_off = {key: interp1d(Aref, y2D, axis=0)(0.0) for key, y2D in lookups2D.items()}

        # Add reference charge vector to 1D lookup dictionaries
        lookups_on['Q'] = Qref
        lookups_off['Q'] = Qref

        # if CW stimulus: change PRF to have exactly one integration interval during stimulus
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

        n_pulse_on = int(np.round(Tpulse_on / dt)) + 1
        n_pulse_off = int(np.round(Tpulse_off / dt))

        # Compute offset size
        n_off = int(np.round(toffset / dt))

        # Initialize global arrays
        stimstate = np.array([1])
        t = np.array([0.0])
        y = np.atleast_2d(np.insert(self.neuron.steadyStates(self.neuron.Vm0), 0, self.Qm0)).T
        nvar = y.shape[0]

        # Initializing accurate pulse time vector
        t_pulse_on = np.linspace(0, Tpulse_on, n_pulse_on)
        t_pulse_off = np.linspace(dt, Tpulse_off, n_pulse_off) + Tpulse_on
        t_pulse0 = np.concatenate([t_pulse_on, t_pulse_off])
        stimstate_pulse = np.concatenate((np.ones(n_pulse_on), np.zeros(n_pulse_off)))

        logger.debug('Computing effective solution')

        # Loop through all pulse (ON and OFF) intervals
        for i in range(npulses):

            # Construct and initialize arrays
            t_pulse = t_pulse0 + t[-1]
            y_pulse = np.empty((nvar, n_pulse_on + n_pulse_off))
            y_pulse[:, 0] = y[:, -1]

            # Integrate ON system
            y_pulse[:, :n_pulse_on] = odeint(
                self.effDerivatives, y[:, -1], t_pulse[:n_pulse_on], args=(lookups_on, )).T

            # Integrate OFF system
            if n_pulse_off > 0:
                y_pulse[:, n_pulse_on:] = odeint(
                    self.effDerivatives, y_pulse[:, n_pulse_on - 1], t_pulse[n_pulse_on:],
                    args=(lookups_off, )).T

            # Append pulse arrays to global arrays
            stimstate = np.concatenate([stimstate[:-1], stimstate_pulse])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)

        # Integrate offset interval
        if n_off > 0:
            t_off = np.linspace(0, toffset, n_off) + t[-1]
            y_off = odeint(self.effDerivatives, y[:, -1], t_off, args=(lookups_off, )).T

            # Concatenate offset arrays to global arrays
            stimstate = np.concatenate([stimstate, np.zeros(n_off - 1)])
            t = np.concatenate([t, t_off[1:]])
            y = np.concatenate([y, y_off[:, 1:]], axis=1)

        # Compute effective gas content vector
        ngeff = np.zeros(stimstate.size)
        ngeff[stimstate == 0] = np.interp(
            y[0, stimstate == 0], lookups_on['Q'], lookups_on['ng'])  # mole
        ngeff[stimstate == 1] = np.interp(
            y[0, stimstate == 1], lookups_off['Q'], lookups_off['ng'])  # mole

        # Compute quasi-steady deflection vector
        Zeff = np.array([self.balancedefQS(ng, Qm) for ng, Qm in zip(ngeff, y[0, :])])  # m

        # Compute membrane potential vector (in mV)
        Vm = np.zeros(stimstate.size)
        Vm[stimstate == 1] = np.interp(y[0, stimstate == 1], lookups_on['Q'], lookups_on['V'])  # mV
        Vm[stimstate == 0] = np.interp(y[0, stimstate == 0], lookups_off['Q'], lookups_off['V'])  # mV

        # Add Zeff, ngeff and Vm to solution matrix
        y = np.vstack([Zeff, ngeff, y[0, :], Vm, y[1:, :]])

        # return output variables
        return (t, y, stimstate)


    def runHybrid(self, Fdrive, Adrive, tstim, toffset, phi=np.pi):
        ''' Compute solutions of the system for a specific set of
            US stimulation parameters, using a hybrid integration scheme.

            The first iteration uses the quasi-steady simplification to compute
            the initiation of motion from a flat leaflet configuration. Afterwards,
            the NBLS ODE system is solved iteratively for "slices" of N-microseconds,
            in a 2-steps scheme:

            - First, the full (n+3) ODE system is integrated for a few acoustic cycles
              until Z and ng reach a stable periodic solution (limit cycle)
            - Second, the signals of the 3 mechanical variables over the last acoustic
              period are selected and resampled to a far lower sampling rate
            - Third, the HH n-ODE system is integrated for the remaining time of the
              slice, using periodic expansion of the mechanical signals to precompute
              the values of capacitance.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param phi: acoustic drive phase (rad)
            :return: 3-tuple with the time profile, the solution matrix and a state vector

            .. warning:: This method cannot handle pulsed stimuli
        '''

        # Initialize full and HH systems solvers
        solver_full = ode(
            lambda t, y, Adrive, Fdrive, phi: self.fullDerivatives(y, t, Adrive, Fdrive, phi))
        solver_full.set_f_params(Adrive, Fdrive, phi)
        solver_full.set_integrator('lsoda', nsteps=SOLVER_NSTEPS)
        solver_hh = ode(lambda t, y, Cm: self.neuron.Qderivatives(y, t, Cm))
        solver_hh.set_integrator('dop853', nsteps=SOLVER_NSTEPS, atol=1e-12)

        # Determine full and HH systems time steps
        Tdrive = 1 / Fdrive
        dt_full = Tdrive / NPC_FULL
        dt_hh = Tdrive / NPC_HH
        n_full_per_hh = int(NPC_FULL / NPC_HH)
        t_full_cycle = np.linspace(0, Tdrive - dt_full, NPC_FULL)
        t_hh_cycle = np.linspace(0, Tdrive - dt_hh, NPC_HH)

        # Determine number of samples in prediction vectors
        npc_pred = NPC_FULL - n_full_per_hh + 1

        # Solve quasi-steady equation to compute first deflection value
        Z0 = 0.0
        ng0 = self.ng0
        Qm0 = self.Qm0
        Pac1 = self.Pacoustic(dt_full, Adrive, Fdrive, phi)
        Z1 = self.balancedefQS(ng0, Qm0, Pac1)

        # Initialize global arrays
        stimstate = np.array([1, 1])
        t = np.array([0., dt_full])
        y_membrane = np.array([[0., (Z1 - Z0) / dt_full], [Z0, Z1], [ng0, ng0], [Qm0, Qm0]])
        y_channels = np.tile(self.neuron.steadyStates(self.neuron.Vm0), (2, 1)).T
        y = np.vstack((y_membrane, y_channels))
        nvar = y.shape[0]

        # Initialize progress bar
        if logger.getEffectiveLevel() == logging.DEBUG:
            widgets = ['Running: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
            pbar = pb.ProgressBar(widgets=widgets, max_value=1000)
            pbar.start()

        # For each hybrid integration interval
        irep = 0
        sim_error = False
        while not sim_error and t[-1] < tstim + toffset:

            # Integrate full system for a few acoustic cycles until stabilization
            periodic_conv = False
            j = 0
            ng_last = None
            Z_last = None
            while not sim_error and not periodic_conv:
                if t[-1] > tstim:
                    solver_full.set_f_params(0.0, 0.0, 0.0)
                t_full = t_full_cycle + t[-1] + dt_full
                y_full = np.empty((nvar, NPC_FULL))
                y0_full = y[:, -1]
                solver_full.set_initial_value(y0_full, t[-1])
                k = 0
                while solver_full.successful() and k <= NPC_FULL - 1:
                    solver_full.integrate(t_full[k])
                    y_full[:, k] = solver_full.y
                    k += 1

                # Compare Z and ng signals over the last 2 acoustic periods
                if j > 0 and rmse(Z_last, y_full[1, :]) < Z_ERR_MAX \
                   and rmse(ng_last, y_full[2, :]) < NG_ERR_MAX:
                    periodic_conv = True

                # Update last vectors for next comparison
                Z_last = y_full[1, :]
                ng_last = y_full[2, :]

                # Concatenate time and solutions to global vectors
                stimstate = np.concatenate([stimstate, np.ones(NPC_FULL)], axis=0)
                t = np.concatenate([t, t_full], axis=0)
                y = np.concatenate([y, y_full], axis=1)

                # Increment loop index
                j += 1

            # Retrieve last period of the 3 mechanical variables to propagate in HH system
            t_last = t[-npc_pred:]
            mech_last = y[0:3, -npc_pred:]

            # Downsample signals to specified HH system time step
            (_, mech_pred) = downsample(t_last, mech_last, NPC_HH)

            # Integrate HH system until certain dQ or dT is reached
            Q0 = y[3, -1]
            dQ = 0.0
            t0_interval = t[-1]
            dt_interval = 0.0
            j = 0
            if t[-1] < tstim:
                tlim = tstim
            else:
                tlim = tstim + toffset
            while (not sim_error and t[-1] < tlim and
                   (np.abs(dQ) < DQ_UPDATE or dt_interval < DT_UPDATE)):
                t_hh = t_hh_cycle + t[-1] + dt_hh
                y_hh = np.empty((nvar - 3, NPC_HH))
                y0_hh = y[3:, -1]
                solver_hh.set_initial_value(y0_hh, t[-1])
                k = 0
                while solver_hh.successful() and k <= NPC_HH - 1:
                    solver_hh.set_f_params(self.Capct(mech_pred[1, k]))
                    solver_hh.integrate(t_hh[k])
                    y_hh[:, k] = solver_hh.y
                    k += 1

                # Concatenate time and solutions to global vectors
                stimstate = np.concatenate([stimstate, np.zeros(NPC_HH)], axis=0)
                t = np.concatenate([t, t_hh], axis=0)
                y = np.concatenate([y, np.concatenate([mech_pred, y_hh], axis=0)], axis=1)

                # Compute charge variation from interval beginning
                dQ = y[3, -1] - Q0
                dt_interval = t[-1] - t0_interval

                # Increment loop index
                j += 1

            # Update progress bar
            if logger.getEffectiveLevel() == logging.DEBUG:
                pbar.update(int(1000 * (t[-1] / (tstim + toffset))))

            irep += 1

        # Terminate progress bar
        if logger.getEffectiveLevel() == logging.DEBUG:
            pbar.finish()

        # Compute membrane potential vector (in mV)
        Vm = y[3, :] / self.v_Capct(y[1, :]) * 1e3  # mV

        # Return output variables with Vm
        return (t, np.vstack([y[1:4, :], Vm, y[4:, :]]), stimstate)


    def checkInputsFull(self, Fdrive, Adrive, tstim, toffset, PRF, DC,
                        method):
        ''' Check validity of simulation parameters.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param method: selected integration method
            :return: 3-tuple with the time profile, the solution matrix and a state vector
        '''

        BilayerSonophore.checkInputs(self, Fdrive, Adrive, 0.0, 0.0)
        self.neuron.checkInputs(Adrive, tstim, toffset, PRF, DC)

        # Check validity of simulation type
        if method not in ('full', 'hybrid', 'sonic'):
            raise ValueError('Invalid integration method: "{}"'.format(method))


    def simulate(self, Fdrive, Adrive, tstim, toffset, PRF=None, DC=1.0,
                 method='sonic'):
        ''' Run simulation of the system for a specific set of
            US stimulation parameters.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param method: selected integration method
            :return: 3-tuple with the time profile, the solution matrix and a state vector
        '''

        # Check validity of stimulation parameters
        self.checkInputsFull(Fdrive, Adrive, tstim, toffset, PRF, DC, method)

        # Call appropriate simulation function
        if method == 'full':
            return self.runFull(Fdrive, Adrive, tstim, toffset, PRF, DC)
        elif method == 'sonic':
            return self.runSONIC(Fdrive, Adrive, tstim, toffset, PRF, DC)
        elif method == 'hybrid':
            if DC < 1.0:
                raise ValueError('Pulsed protocol incompatible with hybrid integration method')
            return self.runHybrid(Fdrive, Adrive, tstim, toffset)


    def nSpikes(self, Adrive, Fdrive, tstim, toffset, PRF, DC, method):
        ''' Run a simulation and determine number of spikes in the response.

            :param Adrive: acoustic amplitude (Pa)
            :param Fdrive: US frequency (Hz)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :return: number of spikes found in response
        '''
        t, y, _ = self.simulate(Fdrive, Adrive, tstim, toffset, PRF, DC, method=method)
        dt = t[1] - t[0]
        ipeaks, *_ = findPeaks(y[2, :], SPIKE_MIN_QAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                               SPIKE_MIN_QPROM)
        nspikes = ipeaks.size
        logger.debug('A = %sPa ---> %s spike%s detected',
                     si_format(Adrive, 2, space=' '),
                     nspikes, "s" if nspikes > 1 else "")
        return nspikes


    def titrate(self, Fdrive, tstim, toffset, PRF=None, DC=1.0, Arange=None, method='sonic'):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given frequency, duration, PRF and duty cycle.

            :param Fdrive: US frequency (Hz)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param Arange: search interval for Adrive, iteratively refined
            :return: determined threshold amplitude (Pa)
        '''

        # Determine amplitude interval if needed
        if Arange is None:
            Arange = (0, getLookups2D(self.neuron.name, a=self.a, Fdrive=Fdrive)[0].max())

        # Titrate
        return titrate(self.nSpikes, (Fdrive, tstim, toffset, PRF, DC, method),
                       Arange, TITRATION_ASTIM_DA_MAX)


    def runAndSave(self, outdir, Fdrive, tstim, toffset, PRF=None, DC=1.0, Adrive=None,
                   method='sonic'):
        ''' Run a simulation of the full electro-mechanical system for a given neuron type
            with specific parameters, and save the results in a PKL file.

            :param outdir: full path to output directory
            :param Fdrive: US frequency (Hz)
            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle (-)
            :param Adrive: acoustic pressure amplitude (Pa)
            :param method: integration method
        '''

        # Get date and time info
        date_str = time.strftime("%Y.%m.%d")
        daytime_str = time.strftime("%H:%M:%S")

        logger.info(
            '%s: %s @ f = %sHz, %st = %ss (%ss offset)%s',
            self,
            'titration' if Adrive is None else 'simulation',
            si_format(Fdrive, 0, space=' '),
            'A = {}Pa, '.format(si_format(Adrive, 2, space=' ')) if Adrive is not None else '',
            *si_format([tstim, toffset], 1, space=' '),
            (', PRF = {}Hz, DC = {:.2f}%'.format(si_format(PRF, 2, space=' '), DC * 1e2) if DC < 1.0 else ''))

        if Adrive is None:
            Adrive = self.titrate(Fdrive, tstim, toffset, PRF, DC, method=method)
            if np.isnan(Adrive):
                logger.error('Could not find threshold excitation amplitude')
                return None

        # Run simulation
        tstart = time.time()
        t, y, stimstate = self.simulate(Fdrive, Adrive, tstim, toffset, PRF, DC, method=method)
        tcomp = time.time() - tstart
        Z, ng, Qm, Vm, *channels = y

        # Detect spikes on Qm signal
        dt = t[1] - t[0]
        ipeaks, *_ = findPeaks(Qm, SPIKE_MIN_QAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                               SPIKE_MIN_QPROM)
        nspikes = ipeaks.size
        lat = t[ipeaks[0]] if nspikes > 0 else 'N/A'
        outstr = '{} spike{} detected'.format(nspikes, 's' if nspikes > 1 else '')
        logger.debug('completed in %ss, %s', si_format(tcomp, 1), outstr)
        sr = np.mean(1 / np.diff(t[ipeaks])) if nspikes > 1 else None

        # Store dataframe and metadata
        U = np.insert(np.diff(Z) / np.diff(t), 0, 0.0)
        df = pd.DataFrame({
            't': t,
            'stimstate': stimstate,
            'U': U,
            'Z': Z,
            'ng': ng,
            'Qm': Qm,
            'Vm': Vm
        })
        for j in range(len(self.neuron.states)):
            df[self.neuron.states[j]] = channels[j]

        meta = {
            'neuron': self.neuron.name,
            'a': self.a,
            'd': self.d,
            'Fdrive': Fdrive,
            'Adrive': Adrive,
            'phi': np.pi,
            'tstim': tstim,
            'toffset': toffset,
            'PRF': PRF,
            'DC': DC,
            'tcomp': tcomp,
            'method': method
        }

        # Export into to PKL file
        simcode = ASTIM_filecode(self.neuron.name, self.a, Fdrive, Adrive, tstim, PRF, DC, method)
        outpath = '{}/{}.pkl'.format(outdir, simcode)
        with open(outpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': df}, fh)
        logger.debug('simulation data exported to "%s"', outpath)

        # Export key metrics to log file
        logpath = os.path.join(outdir, 'log_ASTIM.xlsx')
        logentry = {
            'Date': date_str,
            'Time': daytime_str,
            'Neuron Type': self.neuron.name,
            'Radius (nm)': self.a * 1e9,
            'Thickness (um)': self.d * 1e6,
            'Fdrive (kHz)': Fdrive * 1e-3,
            'Adrive (kPa)': Adrive * 1e-3,
            'Tstim (ms)': tstim * 1e3,
            'PRF (kHz)': PRF * 1e-3 if DC < 1 else 'N/A',
            'Duty factor': DC,
            'Sim. Type': method,
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


    def quasiSteadyStates(self, Fdrive, amps=None, charges=None, DCs=1.0):
        ''' Compute the quasi-steady state values of the neuron's gating variables
            for a combination of US amplitudes, charge densities and duty cycles,
            at a specific US frequency.

            :param Fdrive: US frequency (Hz)
            :param amps: US amplitudes (Pa)
            :param charges: membrane charge densities (C/m2)
            :param DCs: duty cycle value(s)
            :return: 4-tuple with reference values of US amplitude and charge density,
                as well as interpolated Vmeff and QSS gating variables
        '''

        # Get lookups for specific (a, f, A) combination
        Aref, Qref, lookups2D, _ = getLookups2D(self.neuron.name, a=self.a, Fdrive=Fdrive)

        # Derive inputs from lookups reference if not provided
        if amps is None:
            amps = Aref
        if charges is None:
            charges = Qref

        # Transform inputs into arrays if single value provided
        if isinstance(amps, float):
            amps = np.array([amps])
        if isinstance(charges, float):
            charges = np.array([charges])
        if isinstance(DCs, float):
            DCs = np.array([DCs])
        nA, nQ, nDC = amps.size, charges.size, DCs.size
        cs = {True: 's', False: ''}
        logger.debug('%u amplitude%s, %u charge%s, %u DC%s',
                     nA, cs[nA > 1], nQ, cs[nQ > 1], nDC, cs[nDC > 1])

        # Re-interpolate lookups at input charges
        lookups2D = {key: interp1d(Qref, y2D, axis=1)(charges) for key, y2D in lookups2D.items()}

        # Interpolate US-ON (for each input amplitude) and US-OFF (A = 0) lookups
        lookups_on = {key: interp1d(Aref, y2D, axis=0)(amps) for key, y2D in lookups2D.items()}
        lookups_off = {key: interp1d(Aref, y2D, axis=0)(0.0) for key, y2D in lookups2D.items()}

        # Compute DC-averaged Vmeff
        Vmeff_on, Vmeff_off = lookups_on['V'], lookups_off['V']
        Vmeff_avg = np.empty((nA, nQ, nDC))
        for iA, Adrive in enumerate(amps):
            for iDC, DC in enumerate(DCs):
                Vmeff_avg[iA, :, iDC] = Vmeff_on[iA, :] * DC + Vmeff_off * (1 - DC)

        # Compute QSS states
        QS_states = np.empty((len(self.neuron.states), nA, nQ, nDC))
        is_computed = np.zeros(len(self.neuron.states), dtype=int)
        while is_computed.sum() < len(self.neuron.states):

            for i, x in enumerate(self.neuron.states):

                # If voltage-gated state, compute DC-averaged QSS values
                # from interpolated rate constants
                if self.neuron.isVoltageGated(x):
                    alpha_str, beta_str = ['{}{}'.format(s, x.lower()) for s in ['alpha', 'beta']]
                    alphax_on, alphax_off = lookups_on[alpha_str], lookups_off[alpha_str]
                    betax_on, betax_off = lookups_on[beta_str], lookups_off[beta_str]
                    for iA, Adrive in enumerate(amps):
                        for iDC, DC in enumerate(DCs):
                            alphax = alphax_on[iA, :] * DC + alphax_off * (1 - DC)
                            betax = betax_on[iA, :] * DC + betax_off * (1 - DC)
                            QS_states[i, iA, :, iDC] = alphax / (alphax + betax)
                    is_computed[i] = 1

                # Otherwise, it must be a slowly evolving state -> compute QSS values using
                # steady-state function with DC-averaged parameters (states and Vmeff)
                else:
                    # get xinf method and ist arguments
                    xinf_func = getattr(self.neuron, '{}inf'.format(x))
                    xinf_args = inspect.getargspec(xinf_func)[0][1:]

                    # Check if method is Vm-dependent, and extract indexes of QSS states required
                    # to compute method
                    iVm = -1
                    if 'Vm' in xinf_args:
                        iVm = xinf_args.index('Vm')
                        xinf_args.remove('Vm')
                    i_xinf_args = [self.neuron.states.index(arg) for arg in xinf_args]

                    # If all required QSS states are already computed -> compute this QSS state
                    if np.sum(is_computed[i_xinf_args]) == len(i_xinf_args):
                        # Extract QSS values of required states
                        xinf_inputs = QS_states[i_xinf_args]

                        # If method is Vm-dependent, insert DC-averaged Vmeff
                        # to the dependencies array (at appropriate index)
                        if iVm > -1:
                            xinf_inputs = np.insert(xinf_inputs, iVm, Vmeff_avg, axis=0)

                        # Compute QSS value for each (A, Q, DC) combination
                        for iA, Adrive in enumerate(amps):
                            for iDC, DC in enumerate(DCs):
                                QS_states[i, iA, :, iDC] = np.array([
                                    xinf_func(*xinf_inputs[:, iA, iQ, iDC]) for iQ in range(nQ)])

                        is_computed[i] = 1

        # Return reference inputs and output arrays
        return amps, charges, np.squeeze(Vmeff_avg), np.squeeze(QS_states)


    def findRheobaseAmps(self, DCs, Fdrive, Vthr):
        ''' Find the rheobase amplitudes (i.e. threshold acoustic amplitudes of infinite duration
            that would result in excitation) of a specific neuron for various duty cycles.

            :param DCs: duty cycles vector (-)
            :param Fdrive: acoustic drive frequency (Hz)
            :param Vthr: threshold membrane potential above which the neuron necessarily fires (mV)
            :return: rheobase amplitudes vector (Pa)
        '''

        # Get threshold charge from neuron's spike threshold parameter
        Qthr = self.neuron.Cm0 * Vthr * 1e-3  # C/m2

        # Get QSS variables for each amplitude at threshold charge
        Aref, _, Vmeff, QS_states = self.quasiSteadyStates(Fdrive, charges=Qthr, DCs=DCs)

        if DCs.size == 1:
            QS_states = QS_states.reshape((*QS_states.shape, 1))
            Vmeff = Vmeff.reshape((*Vmeff.shape, 1))

        # Compute 2D QSS charge variation array at Qthr
        dQdt = -self.neuron.iNet(Vmeff, QS_states)

        # Find the threshold amplitude that cancels dQdt for each duty cycle
        Arheobase = np.array([np.interp(0, dQdt[:, i], Aref, left=0., right=np.nan)
                              for i in range(DCs.size)])

        # Check if threshold amplitude is found for all DCs
        inan = np.where(np.isnan(Arheobase))[0]
        if inan.size > 0:
            if inan.size == Arheobase.size:
                logger.error(
                    'No rheobase amplitudes within [%s - %sPa] for the provided duty cycles',
                    *si_format((Aref.min(), Aref.max())))
            else:
                minDC = DCs[inan.max() + 1]
                logger.warning(
                    'No rheobase amplitudes within [%s - %sPa] below %.1f%% duty cycle',
                    *si_format((Aref.min(), Aref.max())), minDC * 1e2)

        return Arheobase, Aref


    def computeEffVars(self, Fdrive, Adrive, Qm, fs=None, phi=np.pi):
        ''' Compute "effective" coefficients of the HH system for a specific
            combination of stimulus frequency, stimulus amplitude and charge density.

            A short mechanical simulation is run while imposing the specific charge density,
            until periodic stabilization. The HH coefficients are then averaged over the last
            acoustic cycle to yield "effective" coefficients.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param Qm: imposed charge density (C/m2)
            :param phi: acoustic drive phase (rad)
            :param fs: sonophore membrane coverage fraction
            :return: list with computation time and effective variables
        '''

        if fs is None:
            fs = 1.
            logfs = False
        else:
            logfs = True

        tstart = time.time()

        # Run simulation and retrieve deflection and gas content vectors from last cycle
        _, [Z, ng], _ = BilayerSonophore.simulate(self, Fdrive, Adrive, Qm, phi)
        Z_last = Z[-NPC_FULL:]  # m

        # Compute membrane capacitance profile (taking into account partial coverage)
        Cm = fs * self.v_Capct(Z_last) + (1 - fs) * self.Cm0  # F/m2

        # Compute membrane potential vector
        Vm = Qm / Cm * 1e3  # mV

        # Compute average cycle value for membrane potential and rate constants
        Vm_eff = np.mean(Vm)  # mV
        rates_eff = self.neuron.getEffRates(Vm)

        # Take final cycle value for gas content
        ng_eff = ng[-1]  # mole

        tcomp = time.time() - tstart

        logger.info(
            '%s: lookups @ %sHz, %sPa, %.2f nC/cm2%s: tcomp = %f s',
            self, *si_format([Fdrive, Adrive], precision=1, space=' '), Qm * 1e5,
            ', fs = {:.0f}%'.format(fs * 1e2) if logfs is True else '', tcomp)

        # Return effective coefficients
        return [tcomp, Vm_eff, ng_eff, *rates_eff]
