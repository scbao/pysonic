#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-05-08 14:20:38

import time
import os
import warnings
import pickle
import logging
import progressbar as pb
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp2d

from ..bls import BilayerSonophore
from ..utils import *
from ..constants import *
from ..neurons import BaseMech

# Get package logger
logger = logging.getLogger('PointNICE')


class SolverUS(BilayerSonophore):
    """ This class extends the BilayerSonophore class by adding a biophysical
        Hodgkin-Huxley model on top of the mechanical BLS model. """

    def __init__(self, diameter, neuron, Fdrive, embedding_depth=0.0):
        """ Constructor of the class.

            :param diameter: in-plane diameter of the sonophore structure within the membrane (m)
            :param neuron: neuron object
            :param Fdrive: frequency of acoustic perturbation (Hz)
            :param embedding_depth: depth of the embedding tissue around the membrane (m)
        """

        # Check validity of input parameters
        if not isinstance(neuron, BaseMech):
            raise InputError('Invalid neuron type: "{}" (must inherit from BaseMech class)'
                             .format(neuron.name))
        if not isinstance(Fdrive, float):
            raise InputError('Invalid US driving frequency (must be float typed)')
        if Fdrive < 0:
            raise InputError('Invalid US driving frequency: {} kHz (must be positive or null)'
                             .format(Fdrive * 1e-3))

        # TODO: check parameters dictionary (float type, mandatory members)

        # Initialize BLS object
        Cm0 = neuron.Cm0
        Vm0 = neuron.Vm0
        BilayerSonophore.__init__(self, diameter, Fdrive, Cm0, Cm0 * Vm0 * 1e-3, embedding_depth)

        logger.debug('US solver initialization with %s neuron', neuron.name)


    def eqHH(self, y, t, neuron, Cm):
        """ Compute the derivatives of the n-ODE HH system variables,
            based on a value of membrane capacitance.


            :param y: vector of HH system variables at time t
            :param t: specific instant in time (s)
            :param neuron: neuron object
            :param Cm: membrane capacitance (F/m2)
            :return: vector of HH system derivatives at time t
        """

        # Split input vector explicitly
        Qm, *states = y

        # Compute membrane potential
        Vm = Qm / Cm * 1e3  # mV

        # Compute derivatives
        dQm = - neuron.currNet(Vm, states) * 1e-3  # A/m2
        dstates = neuron.derStates(Vm, states)

        # Return derivatives vector
        return [dQm, *dstates]

    def eqHH2(self, t, y, neuron, Cm):
        return self.eqHH(y, t, neuron, Cm)


    def eqFull(self, y, t, neuron, Adrive, Fdrive, phi):
        """ Compute the derivatives of the (n+3) ODE full NBLS system variables.

            :param y: vector of state variables
            :param t: specific instant in time (s)
            :param neuron: neuron object
            :param Adrive: acoustic drive amplitude (Pa)
            :param Fdrive: acoustic drive frequency (Hz)
            :param phi: acoustic drive phase (rad)
            :return: vector of derivatives
        """

        # Compute derivatives of mechanical and electrical systems
        dydt_mech = self.eqMech(y[:3], t, Adrive, Fdrive, y[3], phi)
        dydt_elec = self.eqHH(y[3:], t, neuron, self.Capct(y[1]))

        # return concatenated output
        return dydt_mech + dydt_elec

    def eqFull2(self, t, y, neuron, Adrive, Fdrive, phi):
        return self.eqFull(y, t, neuron, Adrive, Fdrive, phi)


    def eqHHeff(self, t, y, neuron, interp_data):
        """ Compute the derivatives of the n-ODE effective HH system variables,
            based on 1-dimensional linear interpolation of "effective" coefficients
            that summarize the system's behaviour over an acoustic cycle.

            :param t: specific instant in time (s)
            :param y: vector of HH system variables at time t
            :param neuron: neuron object
            :param interp_data: dictionary of 1D data points of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :return: vector of effective system derivatives at time t
        """

        # Split input vector explicitly
        Qm, *states = y

        # Compute charge and channel states variation
        Vm = np.interp(Qm, interp_data['Q'], interp_data['V'])  # mV
        dQmdt = - neuron.currNet(Vm, states) * 1e-3
        dstates = neuron.derStatesEff(Qm, states, interp_data)

        # Return derivatives vector
        return [dQmdt, *dstates]


    def getEffCoeffs(self, neuron, Fdrive, Adrive, Qm, phi=np.pi):
        """ Compute "effective" coefficients of the HH system for a specific combination
            of stimulus frequency, stimulus amplitude and charge density.

            A short mechanical simulation is run while imposing the specific charge density,
            until periodic stabilization. The HH coefficients are then averaged over the last
            acoustic cycle to yield "effective" coefficients.

            :param neuron: neuron object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param Qm: imposed charge density (C/m2)
            :param phi: acoustic drive phase (rad)
            :return: tuple with the effective potential, gas content and channel rates
        """

        # Run simulation and retrieve deflection and gas content vectors from last cycle
        (_, y, _) = super(SolverUS, self).run(Fdrive, Adrive, Qm, phi)
        (Z, ng) = y
        Z_last = Z[-NPC_FULL:]  # m

        # Compute membrane potential vector
        Vm = Qm / self.v_Capct(Z_last) * 1e3  # mV

        # Compute average cycle value for membrane potential and rate constants
        Vm_eff = np.mean(Vm)  # mV
        rates_eff = neuron.getEffRates(Vm)

        # Take final cycle value for gas content
        ng_eff = ng[-1]  # mole

        return (Vm_eff, ng_eff, *rates_eff)


    def createLookup(self, neuron, freqs, amps, phi=np.pi):
        """ Run simulations of the mechanical system for a multiple combinations of
            imposed charge densities and acoustic amplitudes, compute effective coefficients
            and store them as 2D arrays in a lookup file.

            :param neuron: neuron object
            :param freqs: array of acoustic drive frequencies (Hz)
            :param amps: array of acoustic drive amplitudes (Pa)
            :param phi: acoustic drive phase (rad)
        """

        # Check if lookup file already exists
        lookup_file = '{}_lookups_a{:.1f}nm.pkl'.format(neuron.name, self.a * 1e9)
        lookup_filepath = '{0}/{1}'.format(getLookupDir(), lookup_file)
        if os.path.isfile(lookup_filepath):
            logger.warning('"%s" file already exists and will be overwritten. ' +
                           'Continue? (y/n)', lookup_file)
            user_str = input()
            if user_str not in ['y', 'Y']:
                return -1

        # Check validity of input parameters
        if not isinstance(neuron, BaseMech):
            raise InputError('Invalid neuron type: "{}" (must inherit from BaseMech class)'
                             .format(neuron.name))
        if not isinstance(freqs, np.ndarray):
            if isinstance(freqs, list):
                if not all(isinstance(x, float) for x in freqs):
                    raise InputError('Invalid frequencies (must all be float typed)')
                freqs = np.array(freqs)
            else:
                raise InputError('Invalid frequencies (must be provided as list or numpy array)')
        if not isinstance(amps, np.ndarray):
            if isinstance(amps, list):
                if not all(isinstance(x, float) for x in amps):
                    raise InputError('Invalid amplitudes (must all be float typed)')
                amps = np.array(amps)
            else:
                raise InputError('Invalid amplitudes (must be provided as list or numpy array)')

        nf = freqs.size
        nA = amps.size
        if nf == 0:
            raise InputError('Empty frequencies array')
        if nA == 0:
            raise InputError('Empty amplitudes array')
        if freqs.min() <= 0:
            raise InputError('Invalid US driving frequencies (must all be strictly positive)')
        if amps.min() < 0:
            raise InputError('Invalid US pressure amplitudes (must all be positive or null)')

        logger.info('Creating lookup table for %s neuron', neuron.name)

        # Create neuron-specific charge vector
        charges = np.arange(neuron.Qbounds[0], neuron.Qbounds[1] + 1e-5, 1e-5)  # C/m2

        # Initialize lookup dictionary of 3D array to store effective coefficients
        nQ = charges.size
        coeffs_names = ['V', 'ng', *neuron.coeff_names]
        ncoeffs = len(coeffs_names)
        lookup_dict = {cn: np.empty((nf, nA, nQ)) for cn in coeffs_names}

        # Loop through all (f, A, Q) combinations
        nsims = nf * nA * nQ
        isim = 0
        log_str = 'short simulation %u/%u (f = %.2f kHz, A = %.2f kPa, Q = %.2f nC/cm2)'
        for i in range(nf):
            for j in range(nA):
                for k in range(nQ):
                    isim += 1
                    # Run short simulation and store effective coefficients
                    logger.info(log_str, isim, nsims, freqs[i] * 1e-3, amps[j] * 1e-3,
                                charges[k] * 1e5)
                    try:
                        sim_coeffs = self.getEffCoeffs(neuron, freqs[i], amps[j], charges[k], phi)
                        for icoeff in range(ncoeffs):
                            lookup_dict[coeffs_names[icoeff]][i, j, k] = sim_coeffs[icoeff]
                    except (Warning, AssertionError) as inst:
                        logger.warning('Integration error: %s. Continue batch? (y/n)', extra={inst})
                        user_str = input()
                        if user_str not in ['y', 'Y']:
                            return -1

        # Add input frequency, amplitude and charge arrays to lookup dictionary
        lookup_dict['f'] = freqs  # Hz
        lookup_dict['A'] = amps  # Pa
        lookup_dict['Q'] = charges  # C/m2

        # Save dictionary in lookup file
        logger.info('Saving %s neuron lookup table in file: "%s"', neuron.name, lookup_file)
        with open(lookup_filepath, 'wb') as fh:
            pickle.dump(lookup_dict, fh)

        return 1


    def __runClassic(self, neuron, Fdrive, Adrive, tstim, toffset, PRF, DC, phi=np.pi):
        """ Compute solutions of the system for a specific set of
            US stimulation parameters, using a classic integration scheme.

            The first iteration uses the quasi-steady simplification to compute
            the initiation of motion from a flat leaflet configuration. Afterwards,
            the ODE system is solved iteratively until completion.

            :param neuron: neuron object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param phi: acoustic drive phase (rad)
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        """

        # Raise warnings as error
        warnings.filterwarnings('error')

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
        states = np.array([1, 1])
        t = np.array([0., dt])
        y_membrane = np.array([[0., (Z1 - Z0) / dt], [Z0, Z1], [ng0, ng0], [Qm0, Qm0]])
        y_channels = np.tile(neuron.states0, (2, 1)).T
        y = np.vstack((y_membrane, y_channels))
        nvar = y.shape[0]

        # Initialize pulse time and states vectors
        t_pulse0 = np.linspace(0, Tpulse_on + Tpulse_off, n_pulse_on + n_pulse_off)
        states_pulse = np.concatenate((np.ones(n_pulse_on), np.zeros(n_pulse_off)))

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
            y_pulse[:, :n_pulse_on] = integrate.odeint(self.eqFull, y[:, -1], t_pulse[:n_pulse_on],
                                                       args=(neuron, Adrive, Fdrive, phi)).T

            # Integrate OFF system
            if n_pulse_off > 0:
                y_pulse[:, n_pulse_on:] = integrate.odeint(self.eqFull, y_pulse[:, n_pulse_on - 1],
                                                           t_pulse[n_pulse_on:],
                                                           args=(neuron, 0.0, 0.0, 0.0)).T

            # Append pulse arrays to global arrays
            states = np.concatenate([states, states_pulse[1:]])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)

            # Update progress bar
            if logger.getEffectiveLevel() <= logging.INFO:
                pbar.update(i)

        # Integrate offset interval
        if n_off > 0:
            t_off = np.linspace(0, toffset, n_off) + t[-1]
            states_off = np.zeros(n_off)
            y_off = integrate.odeint(self.eqFull, y[:, -1], t_off, args=(neuron, 0.0, 0.0, 0.0)).T

            # Concatenate offset arrays to global arrays
            states = np.concatenate([states, states_off[1:]])
            t = np.concatenate([t, t_off[1:]])
            y = np.concatenate([y, y_off[:, 1:]], axis=1)

        # Terminate progress bar
        if logger.getEffectiveLevel() <= logging.INFO:
            pbar.finish()

        # Downsample arrays in time-domain accordgin to target temporal resolution
        ds_factor = int(np.round(CLASSIC_TARGET_DT / dt))
        if ds_factor > 1:
            Fs = 1 / (dt * ds_factor)
            logger.info('Downsampling output arrays by factor %u (Fs = %.2f MHz)',
                        ds_factor, Fs * 1e-6)
            t = t[::ds_factor]
            y = y[:, ::ds_factor]
            states = states[::ds_factor]

        # Compute membrane potential vector (in mV)
        Vm = y[3, :] / self.v_Capct(y[1, :]) * 1e3  # mV

        # Return output variables with Vm
        # return (t, y[1:, :], states)
        return (t, np.vstack([y[1:4, :], Vm, y[4:, :]]), states)


    def __runEffective(self, neuron, Fdrive, Adrive, tstim, toffset, PRF, DC, dt=DT_EFF):
        """ Compute solutions of the system for a specific set of
            US stimulation parameters, using charge-predicted "effective"
            coefficients to solve the HH equations at each step.

            :param neuron: neuron object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param dt: integration time step (s)
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        """

        # Raise warnings as error
        warnings.filterwarnings('error')

        # Check lookup file existence
        lookup_file = '{}_lookups_a{:.1f}nm.pkl'.format(neuron.name, self.a * 1e9)
        lookup_path = '{}/{}'.format(getLookupDir(), lookup_file)
        if not os.path.isfile(lookup_path):
            raise InputError('Missing lookup file: "{}"'.format(lookup_file))

        # Load lookups dictionary
        with open(lookup_path, 'rb') as fh:
            lookups3D = pickle.load(fh)

        # Retrieve 1D inputs from lookups dictionary
        freqs = lookups3D.pop('f')
        amps = lookups3D.pop('A')
        charges = lookups3D.pop('Q')

        # Check that stimulation parameters are within lookup range
        margin = 1e-9  # adding margin to compensate for eventual round error
        frange = (freqs.min() - margin, freqs.max() + margin)
        Arange = (amps.min() - margin, amps.max() + margin)

        if Fdrive < frange[0] or Fdrive > frange[1]:
            raise InputError(('Invalid frequency: {:.2f} kHz (must be within ' +
                              '{:.1f} kHz - {:.1f} MHz lookup interval)')
                             .format(Fdrive * 1e-3, frange[0] * 1e-3, frange[1] * 1e-6))
        if Adrive < Arange[0] or Adrive > Arange[1]:
            raise InputError(('Invalid amplitude: {:.2f} kPa (must be within ' +
                              '{:.1f} - {:.1f} kPa lookup interval)')
                             .format(Adrive * 1e-3, Arange[0] * 1e-3, Arange[1] * 1e-3))

        # Interpolate 3D lookups at US frequency
        lookups2D = itrpLookupsFreq(lookups3D, freqs, Fdrive)

        # Interpolate 2D lookups at US amplitude (along with "ng" at zero amplitude)
        lookups1D = {key: np.squeeze(interp2d(amps, charges, lookups2D[key].T)(Adrive, charges))
                     for key in lookups2D.keys()}
        lookups1D['ng0'] = np.squeeze(interp2d(amps, charges, lookups2D['ng'].T)(0.0, charges))

        # Add reference charge vector to 1D lookup dictionary
        lookups1D['Q'] = charges

        # Initialize system solvers
        solver_on = integrate.ode(self.eqHHeff)
        solver_on.set_integrator('lsoda', nsteps=SOLVER_NSTEPS)
        solver_on.set_f_params(neuron, lookups1D)
        solver_off = integrate.ode(self.eqHH2)
        solver_off.set_integrator('lsoda', nsteps=SOLVER_NSTEPS)

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

        # Compute ofset size
        n_off = int(np.round(toffset / dt))

        # Initialize global arrays
        states = np.array([1])
        t = np.array([0.0])
        y = np.atleast_2d(np.insert(neuron.states0, 0, self.Qm0)).T
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
                ngeff_pulse[k] = np.interp(y_pulse[0, k], lookups1D['Q'], lookups1D['ng'])  # mole
                Zeff_pulse[k] = self.balancedefQS(ngeff_pulse[k], y_pulse[0, k])  # m

            # Integrate OFF system
            if n_pulse_off > 0:
                solver_off.set_initial_value(y_pulse[:, k], t_pulse[k])
                solver_off.set_f_params(neuron, self.Capct(Zeff_pulse[k]))
                while solver_off.successful() and k < n_pulse_on + n_pulse_off - 1:
                    k += 1
                    solver_off.integrate(t_pulse[k])
                    y_pulse[:, k] = solver_off.y
                    ngeff_pulse[k] = np.interp(y_pulse[0, k], lookups1D['Q'], lookups1D['ng0'])  # mole
                    Zeff_pulse[k] = self.balancedefQS(ngeff_pulse[k], y_pulse[0, k])  # m
                    solver_off.set_f_params(neuron, self.Capct(Zeff_pulse[k]))

            # Append pulse arrays to global arrays
            states = np.concatenate([states[:-1], states_pulse])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)
            Zeff = np.concatenate([Zeff, Zeff_pulse[1:]])
            ngeff = np.concatenate([ngeff, ngeff_pulse[1:]])

        # Integrate offset interval
        if n_off > 0:
            t_off = np.linspace(0, toffset, n_off) + t[-1]
            states_off = np.zeros(n_off)
            y_off = np.empty((nvar, n_off))
            ngeff_off = np.empty(n_off)
            Zeff_off = np.empty(n_off)

            y_off[:, 0] = y[:, -1]
            ngeff_off[0] = ngeff[-1]
            Zeff_off[0] = Zeff[-1]
            solver_off.set_initial_value(y_off[:, 0], t_off[0])
            solver_off.set_f_params(neuron, self.Capct(Zeff_pulse[-1]))
            k = 0
            while solver_off.successful() and k < n_off - 1:
                k += 1
                solver_off.integrate(t_off[k])
                y_off[:, k] = solver_off.y
                ngeff_off[k] = np.interp(y_off[0, k], lookups1D['Q'], lookups1D['ng0'])  # mole
                Zeff_off[k] = self.balancedefQS(ngeff_off[k], y_off[0, k])  # m
                solver_off.set_f_params(neuron, self.Capct(Zeff_off[k]))

            # Concatenate offset arrays to global arrays
            states = np.concatenate([states, states_off[1:]])
            t = np.concatenate([t, t_off[1:]])
            y = np.concatenate([y, y_off[:, 1:]], axis=1)
            Zeff = np.concatenate([Zeff, Zeff_off[1:]])
            ngeff = np.concatenate([ngeff, ngeff_off[1:]])

        # Compute membrane potential vector (in mV)
        Vm = np.zeros(states.size)
        Vm[states == 0] = y[0, states == 0] / self.v_Capct(Zeff[states == 0]) * 1e3  # mV
        Vm[states == 1] = np.interp(y[0, states == 1], lookups1D['Q'], lookups1D['V'])  # mV

        # Add Zeff, ngeff and Vm to solution matrix
        y = np.vstack([Zeff, ngeff, y[0, :], Vm, y[1:, :]])

        # return output variables
        return (t, y, states)


    def __runHybrid(self, neuron, Fdrive, Adrive, tstim, toffset, phi=np.pi):
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

            :param neuron: neuron object
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
        solver_full = integrate.ode(self.eqFull2)
        solver_full.set_f_params(neuron, Adrive, Fdrive, phi)
        solver_full.set_integrator('lsoda', nsteps=SOLVER_NSTEPS)
        solver_hh = integrate.ode(self.eqHH2)
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
        y_channels = np.tile(neuron.states0, (2, 1)).T
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
                    solver_full.set_f_params(neuron, 0.0, 0.0, 0.0)
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
                states = np.concatenate([states, np.ones(NPC_FULL)], axis=0)
                t = np.concatenate([t, t_full], axis=0)
                y = np.concatenate([y, y_full], axis=1)

                # Increment loop index
                j += 1

            # Retrieve last period of the 3 mechanical variables to propagate in HH system
            t_last = t[-npc_pred:]
            mech_last = y[0:3, -npc_pred:]

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
            while (not sim_error and t[-1] < tlim and
                   (np.abs(dQ) < DQ_UPDATE or dt_interval < DT_UPDATE)):
                t_hh = t_hh_cycle + t[-1] + dt_hh
                y_hh = np.empty((nvar - 3, NPC_HH))
                y0_hh = y[3:, -1]
                solver_hh.set_initial_value(y0_hh, t[-1])
                k = 0
                while solver_hh.successful() and k <= NPC_HH - 1:
                    solver_hh.set_f_params(neuron, self.Capct(mech_pred[1, k]))
                    solver_hh.integrate(t_hh[k])
                    y_hh[:, k] = solver_hh.y
                    k += 1

                # Concatenate time and solutions to global vectors
                states = np.concatenate([states, np.zeros(NPC_HH)], axis=0)
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
        # return (t, y[1:, :], states)
        return (t, np.vstack([y[1:4, :], Vm, y[4:, :]]), states)


    def run(self, neuron, Fdrive, Adrive, tstim, toffset, PRF=None, DC=1.0,
            sim_type='effective'):
        """ Run simulation of the system for a specific set of
            US stimulation parameters.

            :param neuron: neuron object
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param sim_type: selected integration method
            :return: 3-tuple with the time profile, the solution matrix and a state vector
        """

        # Check validity of simulation type
        if sim_type not in ('classic', 'effective', 'hybrid'):
            raise InputError('Invalid integration method: "{}"'.format(sim_type))

        # Check validity of stimulation parameters
        if not isinstance(neuron, BaseMech):
            raise InputError('Invalid neuron type: "{}" (must inherit from BaseMech class)'
                             .format(neuron.name))
        if not all(isinstance(param, float) for param in [Fdrive, Adrive, tstim, toffset, DC]):
            raise InputError('Invalid stimulation parameters (must be float typed)')
        if Fdrive <= 0:
            raise InputError('Invalid US driving frequency: {} kHz (must be strictly positive)'
                             .format(Fdrive * 1e-3))
        if Adrive < 0:
            raise InputError('Invalid US pressure amplitude: {} kPa (must be positive or null)'
                             .format(Adrive * 1e-3))
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
                raise InputError('Invalid PRF: {} Hz (PR interval exceeds stimulus duration'
                                 .format(PRF))
            if PRF >= Fdrive:
                raise InputError('Invalid PRF: {} Hz (must be smaller than driving frequency)'
                                 .format(PRF))

        # Call appropriate simulation function
        if sim_type == 'classic':
            return self.__runClassic(neuron, Fdrive, Adrive, tstim, toffset, PRF, DC)
        elif sim_type == 'effective':
            return self.__runEffective(neuron, Fdrive, Adrive, tstim, toffset, PRF, DC)
        elif sim_type == 'hybrid':
            if DC < 1.0:
                raise InputError('Pulsed protocol incompatible with hybrid integration method')
            return self.__runHybrid(neuron, Fdrive, Adrive, tstim, toffset)


    def findRheobaseAmps(self, neuron, Fdrive, DCs, Vthr):
        ''' Find the rheobase amplitudes (i.e. threshold acoustic amplitudes of infinite duration
            that would result in excitation) of a specific neuron for various stimulation duty cycles.

            :param neuron: neuron object
            :param Fdrive: acoustic drive frequency (Hz)
            :param DCs: duty cycles vector (-)
            :param Vthr: threshold membrane potential above which the neuron necessarily fires (mV)
            :return: rheobase amplitudes vector (Pa)
        '''

        # Check lookup file existence
        lookup_file = '{}_lookups_a{:.1f}nm.pkl'.format(neuron.name, self.a * 1e9)
        lookup_path = '{}/{}'.format(getLookupDir(), lookup_file)
        if not os.path.isfile(lookup_path):
            raise InputError('Missing lookup file: "{}"'.format(lookup_file))

        # Load lookups dictionary
        with open(lookup_path, 'rb') as fh:
            lookups3D = pickle.load(fh)

        # Retrieve 1D inputs from lookups dictionary
        freqs = lookups3D.pop('f')
        amps = lookups3D.pop('A')
        charges = lookups3D.pop('Q')

        # Check that stimulation parameters are within lookup range
        margin = 1e-9  # adding margin to compensate for eventual round error
        frange = (freqs.min() - margin, freqs.max() + margin)

        if Fdrive < frange[0] or Fdrive > frange[1]:
            raise InputError(('Invalid frequency: {:.2f} kHz (must be within ' +
                              '{:.1f} kHz - {:.1f} MHz lookup interval)')
                             .format(Fdrive * 1e-3, frange[0] * 1e-3, frange[1] * 1e-6))

        # Interpolate 3D lookpus at given frequency and threshold charge
        lookups2D = itrpLookupsFreq(lookups3D, freqs, Fdrive)
        Qthr = neuron.Cm0 * Vthr * 1e-3  # C/m2
        lookups1D = {key: np.squeeze(interp2d(amps, charges, lookups2D[key].T)(amps, Qthr))
                     for key in lookups2D.keys()}

        # Remove unnecessary items ot get ON rates and effective potential at threshold charge
        rates_on = lookups1D
        rates_on.pop('ng')
        Vm_on = rates_on.pop('V')

        # Compute neuron OFF rates at threshold potential
        rates_off = neuron.getRates(Vthr)

        # Compute rheobase amplitudes
        rheboase_amps = np.empty(DCs.size)
        for i, DC in enumerate(DCs):
            sstates_pulse = np.empty((len(neuron.states_names), amps.size))
            for j, x in enumerate(neuron.states_names):
                # If channel state, compute pulse-average steady-state values
                if x in neuron.getGates():
                    x = x.lower()
                    alpha_str, beta_str = ['{}{}'.format(s, x) for s in ['alpha', 'beta']]
                    alphax_pulse = rates_on[alpha_str] * DC + rates_off[alpha_str] * (1 - DC)
                    betax_pulse = rates_on[beta_str] * DC + rates_off[beta_str] * (1 - DC)
                    sstates_pulse[j, :] = alphax_pulse / (alphax_pulse + betax_pulse)
                # Otherwise assume the state has reached a steady-state value for Vthr
                else:
                    sstates_pulse[j, :] = np.ones(amps.size) * neuron.steadyStates(Vthr)[j]

            # Compute ON and OFF net currents along the amplitude space
            iNet_on = neuron.currNet(Vm_on, sstates_pulse)
            iNet_off = neuron.currNet(Vthr, sstates_pulse)
            iNet_avg = iNet_on * DC + iNet_off * (1 - DC)

            # Find the threshold amplitude that cancels the approximated pulse average net current
            rheboase_amps[i] = np.interp(0, -iNet_avg, amps, left=0., right=np.nan)

        inan = np.where(np.isnan(rheboase_amps))[0]
        if inan.size > 0:
            if inan.size == rheboase_amps.size:
                logger.error('No rheobase amplitudes within [%s - %sPa] for the provided duty cycles',
                             *si_format((amps.min(), amps.max())))
            else:
                minDC = DCs[inan.max() + 1]
                logger.warning('No rheobase amplitudes within [%s - %sPa] below %.1f%% duty cycle',
                               *si_format((amps.min(), amps.max())), minDC * 1e2)

        return rheboase_amps
