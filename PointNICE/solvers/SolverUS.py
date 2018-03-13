#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-13 15:08:52

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
        assert isinstance(neuron, BaseMech), ('neuron mechanism must be inherited '
                                              'from the BaseMech class')
        assert Fdrive >= 0., 'Driving frequency must be positive'
        # TODO: check parameters dictionary (float type, mandatory members)

        # Initialize BLS object
        Cm0 = neuron.Cm0
        Vm0 = neuron.Vm0
        BilayerSonophore.__init__(self, diameter, Fdrive, Cm0, Cm0 * Vm0 * 1e-3, embedding_depth)

        logger.debug('US solver initialization with %s neuron', neuron.name)


    def eqHH(self, t, y, neuron, Cm):
        """ Compute the derivatives of the n-ODE HH system variables,
            based on a value of membrane capacitance.


            :param t: specific instant in time (s)
            :param y: vector of HH system variables at time t
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


    def eqFull(self, t, y, neuron, Adrive, Fdrive, phi):
        """ Compute the derivatives of the (n+3) ODE full NBLS system variables.

            :param t: specific instant in time (s)
            :param y: vector of state variables
            :param neuron: neuron object
            :param Adrive: acoustic drive amplitude (Pa)
            :param Fdrive: acoustic drive frequency (Hz)
            :param phi: acoustic drive phase (rad)
            :return: vector of derivatives
        """

        # Compute derivatives of mechanical and electrical systems
        dydt_mech = self.eqMech(t, y[:3], Adrive, Fdrive, y[3], phi)
        dydt_elec = self.eqHH(t, y[3:], neuron, self.Capct(y[1]))

        # return concatenated output
        return dydt_mech + dydt_elec


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
        (_, y, _) = self.runMech(Fdrive, Adrive, Qm, phi)
        (Z, ng) = y
        Z_last = Z[-NPC_FULL:]  # m

        # Compute membrane potential vector
        Vm = np.array([Qm / self.Capct(ZZ) * 1e3 for ZZ in Z_last])  # mV

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
        assert not os.path.isfile(lookup_filepath), '"{}" file already exists'.format(lookup_file)

        # Check validity of stimulation parameters
        assert freqs.min() > 0, 'Driving frequencies must be strictly positive'
        assert amps.min() >= 0, 'Acoustic pressure amplitudes must be positive'

        logger.info('Creating lookup table for %s neuron', neuron.name)

        # Create neuron-specific charge vector
        charges = np.arange(neuron.Qbounds[0], neuron.Qbounds[1] + 1e-5, 1e-5)  # C/m2

        # Initialize lookup dictionary of 3D array to store effective coefficients
        nf = freqs.size
        nA = amps.size
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
                    sim_coeffs = self.getEffCoeffs(neuron, freqs[i], amps[j], charges[k], phi)
                    for icoeff in range(ncoeffs):
                        lookup_dict[coeffs_names[icoeff]][i, j, k] = sim_coeffs[icoeff]


        # Add input frequency, amplitude and charge arrays to lookup dictionary
        lookup_dict['f'] = freqs  # Hz
        lookup_dict['A'] = amps  # Pa
        lookup_dict['Q'] = charges  # C/m2

        # Save dictionary in lookup file
        logger.info('Saving %s neuron lookup table in file: "%s"', neuron.name, lookup_file)
        with open(lookup_filepath, 'wb') as fh:
            pickle.dump(lookup_dict, fh)


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

        # Initialize system solver
        solver_full = integrate.ode(self.eqFull)
        solver_full.set_integrator('lsoda', nsteps=SOLVER_NSTEPS, ixpr=True)

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
        if logger.getEffectiveLevel() == logging.DEBUG:
            widgets = ['Running: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
            pbar = pb.ProgressBar(widgets=widgets,
                                  max_value=int(npulses * (toffset + tstim) / tstim))
            pbar.start()

        # Loop through all pulse (ON and OFF) intervals
        for i in range(npulses):

            # Construct and initialize arrays
            t_pulse = t_pulse0 + t[-1]
            y_pulse = np.empty((nvar, n_pulse_on + n_pulse_off))
            y_pulse[:, 0] = y[:, -1]

            # Initialize iterator
            k = 0

            # Integrate ON system
            solver_full.set_f_params(neuron, Adrive, Fdrive, phi)
            solver_full.set_initial_value(y_pulse[:, k], t_pulse[k])
            while solver_full.successful() and k < n_pulse_on - 1:
                k += 1
                solver_full.integrate(t_pulse[k])
                y_pulse[:, k] = solver_full.y

            # Integrate OFF system
            solver_full.set_f_params(neuron, 0.0, 0.0, 0.0)
            solver_full.set_initial_value(y_pulse[:, k], t_pulse[k])
            while solver_full.successful() and k < n_pulse_on + n_pulse_off - 1:
                k += 1
                solver_full.integrate(t_pulse[k])
                y_pulse[:, k] = solver_full.y

            # Append pulse arrays to global arrays
            states = np.concatenate([states, states_pulse[1:]])
            t = np.concatenate([t, t_pulse[1:]])
            y = np.concatenate([y, y_pulse[:, 1:]], axis=1)

            # Update progress bar
            if logger.getEffectiveLevel() == logging.DEBUG:
                pbar.update(i)

        # Integrate offset interval
        if n_off > 0:
            t_off = np.linspace(0, toffset, n_off) + t[-1]
            states_off = np.zeros(n_off)
            y_off = np.empty((nvar, n_off))
            y_off[:, 0] = y[:, -1]
            solver_full.set_initial_value(y_off[:, 0], t_off[0])
            solver_full.set_f_params(neuron, 0.0, 0.0, 0.0)
            k = 0
            while solver_full.successful() and k < n_off - 1:
                k += 1
                solver_full.integrate(t_off[k])
                y_off[:, k] = solver_full.y

            # Concatenate offset arrays to global arrays
            states = np.concatenate([states, states_off[1:]])
            t = np.concatenate([t, t_off[1:]])
            y = np.concatenate([y, y_off[:, 1:]], axis=1)

        # Terminate progress bar
        if logger.getEffectiveLevel() == logging.DEBUG:
            pbar.finish()

        # Downsample arrays in time-domain to reduce overall size
        t = t[::CLASSIC_DS_FACTOR]
        y = y[:, ::CLASSIC_DS_FACTOR]
        states = states[::CLASSIC_DS_FACTOR]

        # Return output variables
        return (t, y[1:, :], states)


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
        assert os.path.isfile(lookup_path), ('No lookup file available for {} '
                                             'neuron type').format(neuron.name)

        # Load coefficients
        with open(lookup_path, 'rb') as fh:
            lookup_dict = pickle.load(fh)


        # Retrieve 1D inputs from lookup dictionary
        freqs = lookup_dict['f']
        amps = lookup_dict['A']
        charges = lookup_dict['Q']

        # Check that stimulation parameters are within lookup range
        margin = 1e-9  # adding margin to compensate for eventual round error
        frange = (freqs.min() - margin, freqs.max() + margin)
        Arange = (amps.min() - margin, amps.max() + margin)
        assert frange[0] <= Fdrive <= frange[1], \
            'Fdrive must be within [{:.1f}, {:.1f}] kHz'.format(*[f * 1e-3 for f in frange])
        assert Arange[0] <= Adrive <= Arange[1], \
            'Adrive must be within [{:.1f}, {:.1f}] kPa'.format(*[A * 1e-3 for A in Arange])

        # Define interpolation datasets to be projected
        coeffs_list = ['V', 'ng', *neuron.coeff_names]

        # If Fdrive in lookup frequencies, simply project (A, Q) interpolation dataset
        # at Fdrive index onto 1D charge-based interpolation dataset
        if Fdrive in freqs:
            iFdrive = np.searchsorted(freqs, Fdrive)
            logger.debug('Using lookups directly at %.2f kHz', freqs[iFdrive] * 1e-3)
            coeffs1d = {}
            for cn in coeffs_list:
                coeff2d = np.squeeze(lookup_dict[cn][iFdrive, :, :])
                itrp = interp2d(amps, charges, coeff2d.T)
                coeffs1d[cn] = itrp(Adrive, charges)
                if cn == 'ng':
                    coeffs1d['ng0'] = itrp(0.0, charges)

        # Otherwise, project 2 (A, Q) interpolation datasets at Fdrive bounding values
        # indexes in lookup frequencies onto two 1D charge-based interpolation datasets, and
        # interpolate between them afterwards
        else:
            ilb = np.searchsorted(freqs, Fdrive) - 1
            logger.debug('Interpolating lookups between %.2f kHz and %.2f kHz',
                         freqs[ilb] * 1e-3, freqs[ilb + 1] * 1e-3)
            coeffs1d = {}
            for cn in coeffs_list:
                coeffs1d_bounds = []
                ng0_bounds = []
                for iFdrive in [ilb, ilb + 1]:
                    coeff2d = np.squeeze(lookup_dict[cn][iFdrive, :, :])
                    itrp = interp2d(amps, charges, coeff2d.T)
                    coeffs1d_bounds.append(itrp(Adrive, charges))
                    if cn == 'ng':
                        ng0_bounds.append(itrp(0.0, charges))
                coeffs1d_bounds = np.squeeze(np.array([coeffs1d_bounds]))
                itrp = interp2d(freqs[ilb:ilb + 2], charges, coeffs1d_bounds.T)
                coeffs1d[cn] = itrp(Fdrive, charges)
                if cn == 'ng':
                    ng0_bounds = np.squeeze(np.array([ng0_bounds]))
                    itrp = interp2d(freqs[ilb:ilb + 2], charges, ng0_bounds.T)
                    coeffs1d['ng0'] = itrp(Fdrive, charges)

        # Squeeze interpolated vectors extra dimensions and add input charges vector
        coeffs1d = {key: np.squeeze(value) for key, value in coeffs1d.items()}
        coeffs1d['Q'] = charges

        # Initialize system solvers
        solver_on = integrate.ode(self.eqHHeff)
        solver_on.set_integrator('lsoda', nsteps=SOLVER_NSTEPS)
        solver_on.set_f_params(neuron, coeffs1d)
        solver_off = integrate.ode(self.eqHH)
        solver_off.set_integrator('lsoda', nsteps=SOLVER_NSTEPS)

        # if CW stimulus: change PRF to have exactly one integration interval during stimulus
        if DC == 1.0:
            PRF = 1 / tstim

        # Compute vector sizes
        npulses = int(np.round(PRF * tstim))
        Tpulse_on = DC / PRF
        Tpulse_off = (1 - DC) / PRF
        n_pulse_on = int(np.round(Tpulse_on / dt)) + 1
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
                ngeff_pulse[k] = np.interp(y_pulse[0, k], coeffs1d['Q'], coeffs1d['ng'])  # mole
                Zeff_pulse[k] = self.balancedefQS(ngeff_pulse[k], y_pulse[0, k])  # m

            # Integrate OFF system
            solver_off.set_initial_value(y_pulse[:, k], t_pulse[k])
            solver_off.set_f_params(neuron, self.Capct(Zeff_pulse[k]))
            while solver_off.successful() and k < n_pulse_on + n_pulse_off - 1:
                k += 1
                solver_off.integrate(t_pulse[k])
                y_pulse[:, k] = solver_off.y
                ngeff_pulse[k] = np.interp(y_pulse[0, k], coeffs1d['Q'], coeffs1d['ng0'])  # mole
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
            solver_off.set_f_params(neuron, self.Capct(Zeff_pulse[k]))
            k = 0
            while solver_off.successful() and k < n_off - 1:
                k += 1
                solver_off.integrate(t_off[k])
                y_off[:, k] = solver_off.y
                ngeff_off[k] = np.interp(y_off[0, k], coeffs1d['Q'], coeffs1d['ng0'])  # mole
                Zeff_off[k] = self.balancedefQS(ngeff_off[k], y_off[0, k])  # m
                solver_off.set_f_params(neuron, self.Capct(Zeff_off[k]))

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
        solver_full = integrate.ode(self.eqFull)
        solver_full.set_f_params(neuron, Adrive, Fdrive, phi)
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
                try:  # try to integrate and catch errors/warnings
                    while solver_full.successful() and k <= NPC_FULL - 1:
                        solver_full.integrate(t_full[k])
                        y_full[:, k] = solver_full.y
                        k += 1
                except (Warning, AssertionError) as inst:
                    sim_error = True
                    logger.error('Full system integration error at step %u', k)
                    logger.error(inst)

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
            while (not sim_error and t[-1] < tlim
                   and (np.abs(dQ) < DQ_UPDATE or dt_interval < DT_UPDATE)):
                t_hh = t_hh_cycle + t[-1] + dt_hh
                y_hh = np.empty((nvar - 3, NPC_HH))
                y0_hh = y[3:, -1]
                solver_hh.set_initial_value(y0_hh, t[-1])
                k = 0
                try:  # try to integrate and catch errors/warnings
                    while solver_hh.successful() and k <= NPC_HH - 1:
                        solver_hh.set_f_params(neuron, self.Capct(mech_pred[1, k]))
                        solver_hh.integrate(t_hh[k])
                        y_hh[:, k] = solver_hh.y
                        k += 1
                except (Warning, AssertionError) as inst:
                    sim_error = True
                    logger.error('HH system integration error at step %u', k)
                    logger.error(inst)

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

        # Return output
        return (t, y[1:, :], states)


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

        # # Check validity of simulation type
        sim_types = ('classic, effective, hybrid')
        assert sim_type in sim_types, 'Allowed simulation types are {}'.format(sim_types)

        # Check validity of stimulation parameters
        assert isinstance(neuron, BaseMech), ('neuron mechanism must be inherited '
                                              'from the BaseMech class')
        for param in [Fdrive, Adrive, tstim, toffset, DC]:
            assert isinstance(param, float), 'stimulation parameters must be float typed'
        assert Fdrive > 0, 'Driving frequency must be strictly positive'
        assert Adrive >= 0, 'Acoustic pressure amplitude must be positive'
        assert tstim > 0, 'Stimulus duration must be strictly positive'
        assert toffset >= 0, 'Stimulus offset must be positive or null'
        assert DC > 0 and DC <= 1, 'Duty cycle must be within [0; 1)'
        if DC < 1.0:
            assert isinstance(PRF, float), 'if provided, the PRF parameter must be float typed'
            assert PRF is not None, 'PRF must be provided when using duty cycles smaller than 1'
            assert PRF >= 1 / tstim, 'PR interval must be smaller than stimulus duration'
            assert PRF < Fdrive, 'PRF must be smaller than driving frequency'

        # Call appropriate simulation function
        if sim_type == 'classic':
            return self.__runClassic(neuron, Fdrive, Adrive, tstim, toffset, PRF, DC)
        elif sim_type == 'effective':
            return self.__runEffective(neuron, Fdrive, Adrive, tstim, toffset, PRF, DC)
        elif sim_type == 'hybrid':
            assert DC == 1.0, 'Hybrid method can only handle continuous wave stimuli'
            return self.__runHybrid(neuron, Fdrive, Adrive, tstim, toffset)

