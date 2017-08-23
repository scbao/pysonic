#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-22 17:30:03

import os
import warnings
import pickle
import logging
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp2d

from ..bls import BilayerSonophore
from ..utils import *
from ..constants import *


# Get package logger
logger = logging.getLogger('PointNICE')


class SolverUS(BilayerSonophore):
    """ This class extends the BilayerSonophore class by adding a biophysical
        Hodgkin-Huxley model on top of the mechanical BLS model. """

    def __init__(self, geom, params, channel_mech, Fdrive):
        """ Constructor of the class.

            :param geom: BLS geometric constants dictionary
            :param params: BLS biomechanical and biophysical parameters dictionary
            :param channel_mech: channels mechanism object
            :param Fdrive: frequency of acoustic perturbation (Hz)
        """

        # Check validity of input parameters
        assert Fdrive >= 0., 'Driving frequency must be positive'

        # Initialize BLS object
        Cm0 = channel_mech.Cm0
        Vm0 = channel_mech.Vm0
        BilayerSonophore.__init__(self, geom, params, Fdrive, Cm0, Cm0 * Vm0 * 1e-3)

        logger.info('US solver initialization with %s channel mechanism', channel_mech.name)


    def eqHH(self, t, y, channel_mech, Cm):
        """ Compute the derivatives of the n-ODE HH system variables,
            based on a value of membrane capacitance.


            :param t: specific instant in time (s)
            :param y: vector of HH system variables at time t
            :param channel_mech: channels mechanism object
            :param Cm: membrane capacitance (F/m2)
            :return: vector of HH system derivatives at time t
        """

        # Split input vector explicitly
        Qm, *states = y

        # Compute membrane potential
        Vm = Qm / Cm * 1e3  # mV

        # Compute derivatives
        dQm = - channel_mech.currNet(Vm, states) * 1e-3  # A/m2
        dstates = channel_mech.derStates(Vm, states)

        # Return derivatives vector
        return [dQm, *dstates]


    def eqHHeff(self, t, y, channel_mech, A, interpolators):
        """ Compute the derivatives of the n-ODE effective HH system variables,
            based on 2-dimensional linear interpolation of "effective" coefficients
            that summarize the system's behaviour over an acoustic cycle.

            :param t: specific instant in time (s)
            :param y: vector of HH system variables at time t
            :param channel_mech: channels mechanism object
            :param A: acoustic drive amplitude (Pa)
            :param channels: Channel object to compute a specific electrical membrane dynamics
            :param interpolators: dictionary of 2-dimensional linear interpolators
                of "effective" coefficients over the 2D amplitude x charge input domain.
            :return: vector of effective system derivatives at time t
        """

        # Split input vector explicitly
        Qm, *states = y

        # Compute charge and channel states variation
        Vm = interpolators['V'](A, Qm)  # mV
        dQmdt = - channel_mech.currNet(Vm, states) * 1e-3
        dstates = channel_mech.derStatesEff(A, Qm, states, interpolators)

        # Return derivatives vector
        return [dQmdt, *dstates]


    def eqFull(self, t, y, channel_mech, Adrive, Fdrive, phi):
        """ Compute the derivatives of the (n+3) ODE full NBLS system variables.

            :param t: specific instant in time (s)
            :param y: vector of state variables
            :param channel_mech: channels mechanism object
            :param Adrive: acoustic drive amplitude (Pa)
            :param Fdrive: acoustic drive frequency (Hz)
            :param phi: acoustic drive phase (rad)
            :return: vector of derivatives
        """

        # Compute derivatives of mechanical and electrical systems
        dydt_mech = self.eqMech(t, y[:3], Adrive, Fdrive, y[3], phi)
        dydt_elec = self.eqHH(t, y[3:], channel_mech, self.Capct(y[1]))

        # return concatenated output
        return dydt_mech + dydt_elec


    def getEffCoeffs(self, channel_mech, Fdrive, Adrive, Qm, phi=np.pi):
        """ Compute "effective" coefficients of the HH system for a specific combination
            of stimulus frequency, stimulus amplitude and charge density.

            A short mechanical simulation is run while imposing the specific charge density,
            until periodic stabilization. The HH coefficients are then averaged over the last
            acoustic cycle to yield "effective" coefficients.

            :param channel_mech: channels mechanism object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param Qm: imposed charge density (C/m2)
            :param phi: acoustic drive phase (rad)
            :return: tuple with the effective potential, rates, and gas content coefficients
        """

        # Run simulation and retrieve deflection and gas content vectors from last cycle
        (_, y, _) = self.runMech(Fdrive, Adrive, Qm, phi)
        (Z, ng) = y
        Z_last = Z[-NPC_FULL:]  # m

        # Compute membrane potential vector
        Vm = np.array([Qm / self.Capct(ZZ) * 1e3 for ZZ in Z_last])  # mV

        # Compute average cycle value for membrane potential and rate constants
        Vm_eff = np.mean(Vm)  # mV
        rates_eff = channel_mech.getEffRates(Vm)

        # Take final cycle value for gas content
        ng_eff = ng[-1]  # mole

        return (Vm_eff, rates_eff, ng_eff)


    def createLookup(self, channel_mech, Fdrive, amps, charges, phi=np.pi):
        """ Run simulations of the mechanical system for a multiple combinations of
            imposed charge densities and acoustic amplitudes, compute effective coefficients
            and store them as 2D arrays in a lookup file.

            :param channel_mech: channels mechanism object
            :param Fdrive: acoustic drive frequency (Hz)
            :param amps: array of acoustic drive amplitudes (Pa)
            :param charges: array of charge densities (C/m2)
            :param phi: acoustic drive phase (rad)
        """

        # Check validity of stimulation parameters
        assert Fdrive > 0, 'Driving frequency must be strictly positive'
        assert np.amin(amps) >= 0, 'Acoustic pressure amplitudes must be positive'

        logger.info('Creating lookup table for f = %.2f kHz', Fdrive * 1e-3)

        # Initialize 3D array to store effective coefficients
        nA = amps.size
        nQ = charges.size
        Vm = np.empty((nA, nQ))
        ng = np.empty((nA, nQ))
        nrates = len(channel_mech.coeff_names)
        rates = np.empty((nA, nQ, nrates))

        # Loop through all (A, Q) combinations
        isim = 0
        for i in range(nA):
            for j in range(nQ):
                isim += 1
                # Run short simulation and store effective coefficients
                logger.info('sim %u/%u (A = %.2f kPa, Q = %.2f nC/cm2)',
                            isim, nA * nQ, amps[i] * 1e-3, charges[j] * 1e5)
                (Vm[i, j], rates[i, j, :], ng[i, j]) = self.getEffCoeffs(channel_mech, Fdrive,
                                                                         amps[i], charges[j], phi)

        # Convert coefficients array into dictionary with specific names
        lookup_dict = {channel_mech.coeff_names[k]: rates[:, :, k] for k in range(nrates)}
        lookup_dict['V'] = Vm  # mV
        lookup_dict['ng'] = ng  # mole

        # Add input amplitude and charge arrays to dictionary
        lookup_dict['A'] = amps  # Pa
        lookup_dict['Q'] = charges  # C/m2

        # Save dictionary in lookup file
        lookup_file = '{}_lookups_a{:.1f}nm_f{:.1f}kHz.pkl'.format(channel_mech.name,
                                                                   self.a * 1e9,
                                                                   Fdrive * 1e-3)
        logger.info('Saving effective coefficients arrays in lookup file: "%s"', lookup_file)
        lookup_filepath = '{0}/{1}/{2}'.format(getLookupDir(), channel_mech.name, lookup_file)
        with open(lookup_filepath, 'wb') as fh:
            pickle.dump(lookup_dict, fh)


    def runClassic(self, channel_mech, Fdrive, Adrive, tstim, toffset, PRF, DF, phi=np.pi):
        """ Compute solutions of the system for a specific set of
            US stimulation parameters, using a classic integration scheme.

            The first iteration uses the quasi-steady simplification to compute
            the initiation of motion from a flat leaflet configuration. Afterwards,
            the ODE system is solved iteratively until completion.

            :param channel_mech: channels mechanism object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DF: pulse duty factor (-)
            :param phi: acoustic drive phase (rad)
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        """

        # Raise warnings as error
        warnings.filterwarnings('error')

        # Initialize system solver
        solver_full = integrate.ode(self.eqFull)
        solver_full.set_integrator('lsoda', nsteps=SOLVER_NSTEPS, ixpr=True)

        # Determine system time step
        Tdrive = 1 / Fdrive
        dt = Tdrive / NPC_FULL

        # Determine proportion of tstim in total integration
        stim_prop = tstim / (tstim + toffset)

        # if CW stimulus: divide integration during stimulus into 100 intervals
        if DF == 1.0:
            PRF = 100 / tstim

        # Compute vector sizes
        npulses = int(np.round(PRF * tstim))
        Tpulse_on = DF / PRF
        Tpulse_off = (1 - DF) / PRF
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
        states = np.array([1, 1])
        t = np.array([0., dt])
        y_membrane = np.array([[0., (Z1 - Z0) / dt], [Z0, Z1], [ng0, ng0], [Qm0, Qm0]])
        y_channels = np.tile(channel_mech.states0, (2, 1)).T
        y = np.vstack((y_membrane, y_channels))
        nvar = y.shape[0]

        # Initialize pulse time and states vectors
        t_pulse0 = np.linspace(0, Tpulse_on + Tpulse_off, n_pulse_on + n_pulse_off)
        states_pulse = np.concatenate((np.ones(n_pulse_on), np.zeros(n_pulse_off)))

        # Loop through all pulse (ON and OFF) intervals
        for i in range(npulses):

            # logger.debug('pulse %u/%u', i + 1, npulses)
            printPct(100 * stim_prop * (i + 1) / npulses, 1)

            # Construct and initialize arrays
            t_pulse = t_pulse0 + t[-1]
            y_pulse = np.empty((nvar, n_pulse_on + n_pulse_off))
            y_pulse[:, 0] = y[:, -1]

            # Initialize iterator
            k = 0

            # Integrate ON system
            solver_full.set_f_params(channel_mech, Adrive, Fdrive, phi)
            solver_full.set_initial_value(y_pulse[:, k], t_pulse[k])
            while solver_full.successful() and k < n_pulse_on - 1:
                k += 1
                solver_full.integrate(t_pulse[k])
                y_pulse[:, k] = solver_full.y

            # Integrate OFF system
            solver_full.set_f_params(channel_mech, 0.0, 0.0, 0.0)
            solver_full.set_initial_value(y_pulse[:, k], t_pulse[k])
            while solver_full.successful() and k < n_pulse_on + n_pulse_off - 1:
                k += 1
                solver_full.integrate(t_pulse[k])
                y_pulse[:, k] = solver_full.y

            # Append pulse arrays to global arrays
            states = np.concatenate([states, states_pulse[1:]])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)

        # Integrate offset interval
        t_off = np.linspace(0, toffset, n_off) + t[-1]
        states_off = np.zeros(n_off)
        y_off = np.empty((nvar, n_off))
        y_off[:, 0] = y[:, -1]
        solver_full.set_initial_value(y_off[:, 0], t_off[0])
        solver_full.set_f_params(channel_mech, 0.0, 0.0, 0.0)
        k = 0
        while solver_full.successful() and k < n_off - 1:
            k += 1
            solver_full.integrate(t_off[k])
            y_off[:, k] = solver_full.y

        # Concatenate offset arrays to global arrays
        states = np.concatenate([states, states_off[1:]])
        t = np.concatenate([t, t_off[1:]])
        y = np.concatenate([y, y_off[:, 1:]], axis=1)

        # Downsample arrays in time-domain to reduce overall size
        t = t[::CLASSIC_DS_FACTOR]
        y = y[:, ::CLASSIC_DS_FACTOR]
        states = states[::CLASSIC_DS_FACTOR]

        # return output variables
        return (t, y[1:, :], states)


    def runEffective(self, channel_mech, Fdrive, Adrive, tstim, toffset, PRF, DF, dt=DT_EFF):
        """ Compute solutions of the system for a specific set of
            US stimulation parameters, using charge-predicted "effective"
            coefficients to solve the HH equations at each step.

            :param channel_mech: channels mechanism object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DF: pulse duty factor (-)
            :param dt: integration time step (s)
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        """

        # Raise warnings as error
        warnings.filterwarnings('error')

        # Check lookup file existence
        lookup_file = '{}_lookups_a{:.1f}nm_f{:.1f}kHz.pkl'.format(channel_mech.name,
                                                                   self.a * 1e9,
                                                                   Fdrive * 1e-3)
        lookup_path = '{}/{}/{}'.format(getLookupDir(), channel_mech.name, lookup_file)
        assert os.path.isfile(lookup_path), 'No lookup file for this stimulation frequency'

        # Load coefficients
        with open(lookup_path, 'rb') as fh:
            coeffs = pickle.load(fh)

        # Check that pressure amplitude is within lookup range
        Amax = np.amax(coeffs['A']) + 1e-9  # adding margin to compensate for eventual round error
        assert Adrive <= Amax, 'Amplitude must be within [0, {:.1f}] kPa'.format(Amax * 1e-3)

        # Initialize interpolators
        interpolators = {cn: interp2d(coeffs['A'], coeffs['Q'], np.transpose(coeffs[cn]))
                         for cn in channel_mech.coeff_names}
        interpolators['V'] = interp2d(coeffs['A'], coeffs['Q'], np.transpose(coeffs['V']))
        interpolators['ng'] = interp2d(coeffs['A'], coeffs['Q'], np.transpose(coeffs['ng']))

        # Initialize system solvers
        solver_on = integrate.ode(self.eqHHeff)
        solver_on.set_integrator('lsoda', nsteps=SOLVER_NSTEPS)
        solver_on.set_f_params(channel_mech, Adrive, interpolators)
        solver_off = integrate.ode(self.eqHH)
        solver_off.set_integrator('lsoda', nsteps=SOLVER_NSTEPS)

        # if CW stimulus: change PRF to have exactly one integration interval during stimulus
        if DF == 1.0:
            PRF = 1 / tstim

        # Compute vector sizes
        npulses = int(np.round(PRF * tstim))
        Tpulse_on = DF / PRF
        Tpulse_off = (1 - DF) / PRF
        n_pulse_on = int(np.round(Tpulse_on / dt)) + 1
        n_pulse_off = int(np.round(Tpulse_off / dt))
        n_off = int(np.round(toffset / dt))

        # Initialize global arrays
        states = np.array([1])
        t = np.array([0.0])
        y = np.atleast_2d(np.insert(channel_mech.states0, 0, self.Qm0)).T
        nvar = y.shape[0]
        Zeff = np.array([0.0])
        ngeff = np.array([self.ng0])

        # Initializing accurate pulse time vector
        t_pulse_on = np.linspace(0, Tpulse_on, n_pulse_on)
        t_pulse_off = np.linspace(dt, Tpulse_off, n_pulse_off) + Tpulse_on
        t_pulse0 = np.concatenate([t_pulse_on, t_pulse_off])
        states_pulse = np.concatenate((np.ones(n_pulse_on), np.zeros(n_pulse_off)))

        # Loop through all pulse (ON and OFF) intervals
        for i in range(npulses):

            # Construct and initialize arrays
            t_pulse = t_pulse0 + t[-1]
            y_pulse = np.empty((nvar, n_pulse_on + n_pulse_off))
            ngeff_pulse = np.empty(n_pulse_on + n_pulse_off)
            Zeff_pulse = np.empty(n_pulse_on + n_pulse_off)
            y_pulse[:, 0] = y[:, -1]
            ngeff_pulse[0] = ngeff[-1]
            Zeff_pulse[0] = Zeff[-1]

            # Initialize iterator
            k = 0

            # Integrate ON system
            solver_on.set_initial_value(y_pulse[:, k], t_pulse[k])
            while solver_on.successful() and k < n_pulse_on - 1:
                k += 1
                solver_on.integrate(t_pulse[k])
                y_pulse[:, k] = solver_on.y
                ngeff_pulse[k] = interpolators['ng'](Adrive, y_pulse[0, k])  # mole
                Zeff_pulse[k] = self.balancedefQS(ngeff_pulse[k], y_pulse[0, k])  # m

            # Integrate OFF system
            solver_off.set_initial_value(y_pulse[:, k], t_pulse[k])
            solver_off.set_f_params(channel_mech, self.Capct(Zeff_pulse[k]))
            while solver_off.successful() and k < n_pulse_on + n_pulse_off - 1:
                k += 1
                solver_off.integrate(t_pulse[k])
                y_pulse[:, k] = solver_off.y
                ngeff_pulse[k] = interpolators['ng'](0.0, y_pulse[0, k])  # mole
                Zeff_pulse[k] = self.balancedefQS(ngeff_pulse[k], y_pulse[0, k])  # m
                solver_off.set_f_params(channel_mech, self.Capct(Zeff_pulse[k]))

            # Append pulse arrays to global arrays
            states = np.concatenate([states[:-1], states_pulse])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)
            Zeff = np.concatenate([Zeff, Zeff_pulse[1:]])
            ngeff = np.concatenate([ngeff, ngeff_pulse[1:]])

        # Integrate offset interval
        t_off = np.linspace(0, toffset, n_off) + t[-1]
        states_off = np.zeros(n_off)
        y_off = np.empty((nvar, n_off))
        ngeff_off = np.empty(n_off)
        Zeff_off = np.empty(n_off)

        y_off[:, 0] = y[:, -1]
        ngeff_off[0] = ngeff[-1]
        Zeff_off[0] = Zeff[-1]
        solver_off.set_initial_value(y_off[:, 0], t_off[0])
        solver_off.set_f_params(channel_mech, self.Capct(Zeff_pulse[k]))
        k = 0
        while solver_off.successful() and k < n_off - 1:
            k += 1
            solver_off.integrate(t_off[k])
            y_off[:, k] = solver_off.y
            ngeff_off[k] = interpolators['ng'](0.0, y_off[0, k])  # mole
            Zeff_off[k] = self.balancedefQS(ngeff_off[k], y_off[0, k])  # m
            solver_off.set_f_params(channel_mech, self.Capct(Zeff_off[k]))

        # Concatenate offset arrays to global arrays
        states = np.concatenate([states, states_off[1:]])
        t = np.concatenate([t, t_off[1:]])
        y = np.concatenate([y, y_off[:, 1:]], axis=1)
        Zeff = np.concatenate([Zeff, Zeff_off[1:]])
        ngeff = np.concatenate([ngeff, ngeff_off[1:]])

        # Add Zeff and ngeff to solution matrix
        y = np.vstack([Zeff, ngeff, y])

        # return output variables
        return (t, y, states)


    def runHybrid(self, channel_mech, Fdrive, Adrive, tstim, toffset, phi=np.pi):
        """ Compute solutions of the system for a specific set of
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

            :param channel_mech: channels mechanism object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param phi: acoustic drive phase (rad)
            :return: 3-tuple with the time profile, the solution matrix and a state vector

            .. warning:: This method cannot handle pulsed stimuli
        """

        # Raise warnings as error
        warnings.filterwarnings('error')

        # Initialize full and HH systems solvers
        solver_full = integrate.ode(self.eqFull)
        solver_full.set_f_params(channel_mech, Adrive, Fdrive, phi)
        solver_full.set_integrator('lsoda', nsteps=SOLVER_NSTEPS)
        solver_hh = integrate.ode(self.eqHH)
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
        states = np.array([1, 1])
        t = np.array([0., dt_full])
        y_membrane = np.array([[0., (Z1 - Z0) / dt_full], [Z0, Z1], [ng0, ng0], [Qm0, Qm0]])
        y_channels = np.tile(channel_mech.states0, (2, 1)).T
        y = np.vstack((y_membrane, y_channels))
        nvar = y.shape[0]

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
                    solver_full.set_f_params(channel_mech, 0.0, 0.0, 0.0)
                t_full = t_full_cycle + t[-1] + dt_full
                y_full = np.empty((nvar, NPC_FULL))
                y0_full = y[:, -1]
                solver_full.set_initial_value(y0_full, t[-1])
                k = 0
                try:  # try to integrate and catch errors/warnings
                    while solver_full.successful() and k <= NPC_FULL - 1:
                        solver_full.integrate(t_full[k])
                        y_full[:, k] = solver_full.y
                        assert (y_full[1, k] > -0.5 * self.Delta), 'Deflection out of range'
                        k += 1
                except (Warning, AssertionError) as inst:
                    sim_error = True
                    logger.error('Full system integration error at step %u', k)
                    print(inst)

                # Compare Z and ng signals over the last 2 acoustic periods
                if j > 0 and rmse(Z_last, y_full[1, :]) < Z_ERR_MAX \
                   and rmse(ng_last, y_full[2, :]) < NG_ERR_MAX:
                    periodic_conv = True

                # Update last vectors for next comparison
                Z_last = y_full[1, :]
                ng_last = y_full[2, :]

                # Concatenate time and solutions to global vectors
                states = np.concatenate([states, np.ones(NPC_FULL)], axis=0)
                t = np.concatenate([t, t_full], axis=0)
                y = np.concatenate([y, y_full], axis=1)

                # Increment loop index
                j += 1

            # Retrieve last period of the 3 mechanical variables to propagate in HH system
            t_last = t[-npc_pred:]
            mech_last = y[0:3, -npc_pred:]

            # print('convergence after {} cycles'.format(j))

            # Downsample signals to specified HH system time step
            (_, mech_pred) = DownSample(t_last, mech_last, NPC_HH)

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
            while (not sim_error and t[-1] < tlim
                   and (np.abs(dQ) < DQ_UPDATE or dt_interval < DT_UPDATE)):
                t_hh = t_hh_cycle + t[-1] + dt_hh
                y_hh = np.empty((nvar - 3, NPC_HH))
                y0_hh = y[3:, -1]
                solver_hh.set_initial_value(y0_hh, t[-1])
                k = 0
                try:  # try to integrate and catch errors/warnings
                    while solver_hh.successful() and k <= NPC_HH - 1:
                        solver_hh.set_f_params(channel_mech, self.Capct(mech_pred[1, k]))
                        solver_hh.integrate(t_hh[k])
                        y_hh[:, k] = solver_hh.y
                        k += 1
                except (Warning, AssertionError) as inst:
                    sim_error = True
                    logger.error('HH system integration error at step %u', k)
                    print(inst)

                # Concatenate time and solutions to global vectors
                states = np.concatenate([states, np.zeros(NPC_HH)], axis=0)
                t = np.concatenate([t, t_hh], axis=0)
                y = np.concatenate([y, np.concatenate([mech_pred, y_hh], axis=0)], axis=1)

                # Compute charge variation from interval beginning
                dQ = y[3, -1] - Q0
                dt_interval = t[-1] - t0_interval

                # Increment loop index
                j += 1

            # Print progress
            printPct(100 * (t[-1] / (tstim + toffset)), 1)

            irep += 1

        # Return output
        return (t, y[1:, :], states)


    def runSim(self, channel_mech, Fdrive, Adrive, tstim, toffset, PRF, DF=1.0, sim_type='effective'):
        """ Run simulation of the system for a specific set of
            US stimulation parameters.

            :param channel_mech: channels mechanism object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DF: pulse duty factor (-)
            :param sim_type: selected integration method
            :return: 3-tuple with the time profile, the solution matrix and a state vector
        """

        # Check validity of stimulation parameters
        assert Fdrive > 0, 'Driving frequency must be strictly positive'
        assert Adrive > 0, 'Acoustic pressure amplitude must be strictly positive'
        assert tstim > 0, 'Stimulus duration must be strictly positive'
        assert toffset >= 0, 'Stimulus offset must be positive or null'
        assert PRF >= 1 / tstim, 'Pulse repetition interval must be smaller than stimulus duration'
        assert PRF < Fdrive, 'PRF must be smaller than driving frequency'
        assert DF > 0 and DF <= 1, 'Duty cycle must be within [0; 1)'
        sim_types = ('classic, effective, hybrid')
        assert sim_type in sim_types, 'Allowed simulation types are {}'.format(sim_types)

        # Call appropriate simulation function
        if sim_type == 'classic':
            return self.runClassic(channel_mech, Fdrive, Adrive, tstim, toffset, PRF, DF)
        elif sim_type == 'effective':
            return self.runEffective(channel_mech, Fdrive, Adrive, tstim, toffset, PRF, DF)
        elif sim_type == 'hybrid':
            assert DF == 1.0, 'Hybrid method can only handle continuous wave stimuli'
            return self.runHybrid(channel_mech, Fdrive, Adrive, tstim, toffset)


    def titrateAmp(self, channel_mech, Fdrive, Arange, tstim, toffset,
                   PRF=1.5e3, DF=1.0, sim_type='effective'):
        """ Use a dichotomic search to determine the threshold acoustic amplitude
            needed to obtain neural excitation, for specific stimulation parameters.

            This function is called recursively until an accurate threshold is found.

            :param channel_mech: channels mechanism object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Arange: bounds of the acoustic amplitude searching interval (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DF: pulse duty factor (-)
            :param sim_type: selected integration method
            :return: 5-tuple with the determined amplitude threshold, time profile,
                     solution matrix, state vector and response latency
        """

        # Check amplitude interval
        assert Arange[0] < Arange[1], 'Amplitude bounds must be (lower_bound, upper_bound)'

        # Define current amplitude
        Adrive = (Arange[0] + Arange[1]) / 2

        # Run simulation
        (t, y, states) = self.runSim(channel_mech, Fdrive, Adrive, tstim, toffset,
                                     PRF, DF, sim_type)

        # Detect spikes
        n_spikes, latency, _ = detectSpikes(t, y[2, :], SPIKE_MIN_QAMP, SPIKE_MIN_DT)
        logger.info('%.2f kPa ---> %u spike%s detected', Adrive * 1e-3, n_spikes,
                    "s" if n_spikes > 1 else "")

        # If accurate threshold is found, return simulation results
        if (Arange[1] - Arange[0]) <= TITRATION_DA_THR and n_spikes == 1:
            return (Adrive, t, y, states, latency)

        # Otherwise, refine titration interval and iterate recursively
        else:
            if n_spikes == 0:
                new_Arange = (Adrive, Arange[1])
            else:
                new_Arange = (Arange[0], Adrive)
            return self.titrateAmp(channel_mech, Fdrive, new_Arange, tstim, toffset,
                                   PRF, DF, sim_type)


    def titrateDur(self, channel_mech, Fdrive, Adrive, trange, toffset,
                   PRF=1.5e3, DF=1.0, sim_type='effective'):
        """ Use a dichotomic search to determine the threshold stimulus duration
            needed to obtain neural excitation, for specific stimulation parameters.

            This function is called recursively until an accurate threshold is found.

            :param channel_mech: channels mechanism object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param trange: bounds of the stimulus duration (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DF: pulse duty factor (-)
            :param sim_type: selected integration method
            :return: 5-tuple with the determined duration threshold, time profile,
                     solution matrix, state vector and response latency
        """

        # Check duration interval
        assert trange[0] < trange[1], 'Duration bounds must be (lower_bound, upper_bound)'

        # Define current duration
        tstim = (trange[0] + trange[1]) / 2

        # Run simulation
        (t, y, states) = self.runSim(channel_mech, Fdrive, Adrive, tstim, toffset,
                                     PRF, DF, sim_type)

        # Detect spikes
        n_spikes, latency, _ = detectSpikes(t, y[2, :], SPIKE_MIN_QAMP, SPIKE_MIN_DT)
        logger.info('%.2f ms ---> %u spike%s detected', tstim * 1e3, n_spikes,
                    "s" if n_spikes > 1 else "")

        # If accurate threshold is found, return simulation results
        if (trange[1] - trange[0]) <= TITRATION_DT_THR and n_spikes == 1:
            return (tstim, t, y, states, latency)

        # Otherwise, refine titration interval and iterate recursively
        else:
            if n_spikes == 0:
                new_trange = (tstim, trange[1])
            else:
                new_trange = (trange[0], tstim)
            return self.titrateDur(channel_mech, Fdrive, Adrive, new_trange, toffset,
                                   PRF, DF, sim_type)


    def titrate(self, channel_mech, Fdrive, x, toffset, PRF, DF, titr_type, sim_type='effective'):
        if titr_type == 'amplitude':
            return self.titrateAmp(channel_mech, Fdrive, (0.0, 2 * TITRATION_AMAX), x,
                                   toffset, PRF, DF, sim_type)
        elif titr_type == 'duration':
            return self.titrateDur(channel_mech, Fdrive, x, (0.0, 2 * TITRATION_TMAX),
                                   toffset, PRF, DF, sim_type)

