#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-31 16:52:41

from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from .simulators import PWSimulator, HybridSimulator
from .bls import BilayerSonophore
from .pneuron import PointNeuron
from ..batches import createQueue
from ..utils import *
from ..constants import *
from ..postpro import getFixedPoints


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

    def filecode(self, Fdrive, Adrive, tstim, toffset, PRF, DC, method):
        return 'ASTIM_{}_{}_{:.0f}nm_{:.0f}kHz_{:.2f}kPa_{:.0f}ms_{}{}'.format(
            self.neuron.name, 'CW' if DC == 1 else 'PW', self.a * 1e9,
            Fdrive * 1e-3, Adrive * 1e-3, tstim * 1e3,
            'PRF{:.2f}Hz_DC{:.2f}%_'.format(PRF, DC * 1e2) if DC < 1. else '', method)

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

    def effDerivatives(self, y, t, lkp):
        ''' Compute the derivatives of the n-ODE effective HH system variables,
            based on 1-dimensional linear interpolation of "effective" coefficients
            that summarize the system's behaviour over an acoustic cycle.

            :param y: vector of HH system variables at time t
            :param t: specific instant in time (s)
            :param lkp: dictionary of 1D data points of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :return: vector of effective system derivatives at time t
        '''

        # Split input vector explicitly
        Qm, *states = y

        # Compute charge and channel states variation
        Vmeff = self.neuron.interpVmeff(Qm, lkp)
        dQmdt = - self.neuron.iNet(Vmeff, states) * 1e-3
        dstates = self.neuron.derEffStates(Qm, states, lkp)

        # Return derivatives vector
        return [dQmdt, *[dstates[k] for k in self.neuron.states]]

    def interpEffVariable(self, key, Qm, stim, lkp_on, lkp_off):
        ''' Interpolate Q-dependent effective variable along solution.

            :param key: lookup variable key
            :param Qm: charge density solution vector
            :param stim: stimulation state solution vector
            :param lkp_on: lookups for ON states
            :param lkp_off: lookups for OFF states
            :return: interpolated effective variable vector
        '''
        x = np.zeros(stim.size)
        x[stim == 0] = np.interp(
            Qm[stim == 0], lkp_on['Q'], lkp_on[key], left=np.nan, right=np.nan)
        x[stim == 1] = np.interp(
            Qm[stim == 1], lkp_off['Q'], lkp_off[key], left=np.nan, right=np.nan)
        return x

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
            :return: 2-tuple with the output dataframe and computation time.
        '''

        # Determine time step
        dt = 1 / (NPC_FULL * Fdrive)

        # Compute non-zero deflection value for a small perturbation (solving quasi-steady equation)
        Pac = self.Pacoustic(dt, Adrive, Fdrive, phi)
        Z0 = self.balancedefQS(self.ng0, self.Qm0, Pac)

        # Set initial conditions
        steady_states = self.neuron.steadyStates(self.neuron.Vm0)
        y0 = np.concatenate((
            [0., Z0, self.ng0, self.Qm0],
            [steady_states[k] for k in self.neuron.states]))

        # Initialize simulator and compute solution
        logger.debug('Computing detailed solution')
        simulator = PWSimulator(
            lambda y, t: self.fullDerivatives(y, t, Adrive, Fdrive, phi),
            lambda y, t: self.fullDerivatives(y, t, 0., 0., 0.))
        (t, y, stim), tcomp = simulator(
            y0, dt, tstim, toffset, PRF, DC,
            print_progress=logger.getEffectiveLevel() <= logging.INFO,
            target_dt=CLASSIC_TARGET_DT,
            monitor_time=True)
        logger.debug('completed in %ss', si_format(tcomp, 1))

        # Store output in dataframe
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Z': y[:, 1],
            'ng': y[:, 2],
            'Qm': y[:, 3]
        })
        data['Vm'] = data['Qm'].values / self.v_Capct(data['Z'].values) * 1e3  # mV
        for i in range(len(self.neuron.states)):
            data[self.neuron.states[i]] = y[:, i + 4]

        # Return dataframe and computation time
        return data, tcomp

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

        # Set initial conditions
        steady_states = self.neuron.steadyStates(self.neuron.Vm0)
        y0 = np.insert(
            np.array([steady_states[k] for k in self.neuron.states]),
            0, self.Qm0)

        # Initialize simulator and compute solution
        logger.debug('Computing effective solution')
        simulator = PWSimulator(
            lambda y, t: self.effDerivatives(y, t, lookups_on),
            lambda y, t: self.effDerivatives(y, t, lookups_off))
        (t, y, stim), tcomp = simulator(y0, dt, tstim, toffset, PRF, DC, monitor_time=True)
        logger.debug('completed in %ss', si_format(tcomp, 1))

        # Store output in dataframe
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Qm': y[:, 0]
        })
        for key in ['ng', 'V']:
            data[key] = self.interpEffVariable(
                key, data['Qm'].values, stim, lookups_on, lookups_off)
        data['Z'] = np.array([self.balancedefQS(ng, Qm) for ng, Qm in zip(
            data['ng'].values, data['Qm'].values)])  # m
        data['Vm'] = data['Qm'].values / self.v_Capct(data['Z'].values) * 1e3  # mV
        for i in range(len(self.neuron.states)):
            data[self.neuron.states[i]] = y[:, i + 1]

        # Return dataframe and computation time
        return data, tcomp

    def runHybrid(self, Fdrive, Adrive, tstim, toffset, PRF, DC, phi=np.pi):
        ''' Compute solutions of the system for a specific set of
            US stimulation parameters, using a hybrid integration scheme.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param phi: acoustic drive phase (rad)
            :return: 3-tuple with the time profile, the solution matrix and a state vector

        '''

        # Determine time step
        dt_dense = 1 / (NPC_FULL * Fdrive)
        dt_sparse = 1 / (NPC_HH * Fdrive)

        # Compute non-zero deflection value for a small perturbation (solving quasi-steady equation)
        Pac = self.Pacoustic(dt_dense, Adrive, Fdrive, phi)
        Z0 = self.balancedefQS(self.ng0, self.Qm0, Pac)

        # Set initial conditions
        steady_states = self.neuron.steadyStates(self.neuron.Vm0)
        y0 = np.concatenate((
            [0., Z0, self.ng0, self.Qm0],
            [steady_states[k] for k in self.neuron.states],
        ))
        is_dense_var = np.array([True] * 3 + [False] * (len(self.neuron.states) + 1))

        # Initialize simulator and compute solution
        logger.debug('Computing hybrid solution')
        simulator = HybridSimulator(
            lambda y, t: self.fullDerivatives(y, t, Adrive, Fdrive, phi),
            lambda y, t: self.fullDerivatives(y, t, 0., 0., 0.),
            lambda t, y, Cm: self.neuron.Qderivatives(y, t, Cm),
            lambda yref: self.Capct(yref[1]),
            is_dense_var,
            ivars_to_check=[1, 2])
        (t, y, stim), tcomp = simulator(
            y0, dt_dense, dt_sparse, Fdrive, tstim, toffset, PRF, DC,
            monitor_time=True)
        logger.debug('completed in %ss', si_format(tcomp, 1))

        # Store output in dataframe
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Z': y[:, 1],
            'ng': y[:, 2],
            'Qm': y[:, 3]
        })
        data['Vm'] = data['Qm'].values / self.v_Capct(data['Z'].values) * 1e3  # mV
        for i in range(len(self.neuron.states)):
            data[self.neuron.states[i]] = y[:, i + 4]

        # Return dataframe and computation time
        return data, tcomp

    def simulate(self, Fdrive, Adrive, tstim, toffset, PRF=None, DC=1.0, method='sonic'):
        ''' Simulate the electro-mechanical model for a specific set of US stimulation parameters,
            and return output data in a dataframe.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param method: selected integration method
            :return: 2-tuple with the output dataframe and computation time.
        '''

        logger.info(
            '%s: %s @ f = %sHz, %st = %ss (%ss offset)%s',
            self,
            'titration' if Adrive is None else 'simulation',
            si_format(Fdrive, 0, space=' '),
            'A = {}Pa, '.format(si_format(Adrive, 2, space=' ')) if Adrive is not None else '',
            *si_format([tstim, toffset], 1, space=' '),
            (', PRF = {}Hz, DC = {:.2f}%'.format(
                si_format(PRF, 2, space=' '), DC * 1e2) if DC < 1.0 else ''))

        # TODO: If no amplitude provided, perform titration
        if Adrive is None:
            Adrive = self.titrate(Fdrive, tstim, toffset, PRF, DC, method=method)
            if np.isnan(Adrive):
                logger.error('Could not find threshold excitation amplitude')
                return None

        # Check validity of stimulation parameters
        BilayerSonophore.checkInputs(self, Fdrive, Adrive, 0.0, 0.0)
        self.neuron.checkInputs(Adrive, tstim, toffset, PRF, DC)

        # Call appropriate simulation function
        try:
            simfunc = {
                'full': self.runFull,
                'hybrid': self.runHybrid,
                'sonic': self.runSONIC
            }[method]
        except KeyError:
            raise ValueError('Invalid integration method: "{}"'.format(method))
        data, tcomp = simfunc(Fdrive, Adrive, tstim, toffset, PRF, DC)

        # Log number of detected spikes
        nspikes = self.neuron.getNSpikes(data)
        logger.debug('{} spike{} detected'.format(nspikes, plural(nspikes)))

        # Return dataframe and computation time
        return data, tcomp

    def meta(self, Fdrive, Adrive, tstim, toffset, PRF, DC, method):
        ''' Return information about object and simulation parameters.

            :param Fdrive: US frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle (-)
            :param method: integration method
            :return: meta-data dictionary
        '''
        return {
            'neuron': self.neuron.name,
            'a': self.a,
            'd': self.d,
            'Fdrive': Fdrive,
            'Adrive': Adrive,
            'tstim': tstim,
            'toffset': toffset,
            'PRF': PRF,
            'DC': DC,
            'method': method
        }

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

        # Determine output function
        if self.neuron.isTitratable():
            xfunc = self.isExcited
        else:
            xfunc = self.isSilenced

        # Titrate
        return titrate(xfunc, (Fdrive, tstim, toffset, PRF, DC, method),
                       Arange, TITRATION_ASTIM_DA_MAX)

    def createQueue(self, freqs, amps, durations, offsets, PRFs, DCs, method):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps, while avoiding repetition of CW protocols for a given PRF sweep.

            :param freqs: list (or 1D-array) of US frequencies
            :param amps: list (or 1D-array) of acoustic amplitudes
            :param durations: list (or 1D-array) of stimulus durations
            :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
            :param PRFs: list (or 1D-array) of pulse-repetition frequencies
            :param DCs: list (or 1D-array) of duty cycle values
            :params method: integration method
            :return: list of parameters (list) for each simulation
        '''
        if amps is None:
            amps = [np.nan]
        DCs = np.array(DCs)
        queue = []
        if 1.0 in DCs:
            queue += createQueue(freqs, amps, durations, offsets, min(PRFs), 1.0)
        if np.any(DCs != 1.0):
            queue += createQueue(freqs, amps, durations, offsets, PRFs, DCs[DCs != 1.0])
        for item in queue:
            if np.isnan(item[1]):
                item[1] = None
            item.append(method)
        return queue

    def quasiSteadyStates(self, Fdrive, amps=None, charges=None, DCs=1.0, squeeze_output=False):
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

        # Get DC-averaged lookups interpolated at the appropriate amplitudes and charges
        amps, charges, lookups = getLookupsDCavg(
            self.neuron.name, self.a, Fdrive, amps, charges, DCs)

        # Compute QSS states using these lookups
        nA, nQ, nDC = lookups['V'].shape
        QSS = {k: np.empty((nA, nQ, nDC)) for k in self.neuron.states}
        for iA in range(nA):
            for iDC in range(nDC):
                QSS_1D = self.neuron.quasiSteadyStates(
                    {k: v[iA, :, iDC] for k, v in lookups.items()})
                for k in QSS.keys():
                    QSS[k][iA, :, iDC] = QSS_1D[k]

        # Compress outputs if needed
        if squeeze_output:
            QSS = {k: v.squeeze() for k, v in QSS.items()}
            lookups = {k: v.squeeze() for k, v in lookups.items()}

        # Return reference inputs and outputs
        return amps, charges, lookups, QSS


    def quasiSteadyStateiNet(self, Qm, Fdrive, Adrive, DC):
        ''' Compute quasi-steady state net membrane current for a given combination
            of US parameters and a given membrane charge density.

            :param Qm: membrane charge density (C/m2)
            :param Fdrive: US frequency (Hz)
            :param Adrive: US amplitude (Pa)
            :param DC: duty cycle (-)
            :return: net membrane current (mA/m2)
        '''
        _, _, lookups, QSS = self.quasiSteadyStates(
            Fdrive, amps=Adrive, charges=Qm, DCs=DC, squeeze_output=True)
        return self.neuron.iNet(lookups['V'], np.array(list(QSS.values())))  # mA/m2


    def evaluateStability(self, Qm0, states0, lkp):
        ''' Integrate the effective differential system from a given starting point,
            until clear convergence or clear divergence is found.

            :param Qm0: initial membrane charge density (C/m2)
            :param states0: dictionary of initial states values
            :param lkp: dictionary of 1D data points of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :return: boolean indicating convergence state
        '''

        # Initialize y0 vector
        t0 = 0.
        y0 = np.array([Qm0] + list(states0.values()))

        # Initializing empty list to record evolution of charge deviation
        n = int(QSS_HISTORY_INTERVAL // QSS_INTEGRATION_INTERVAL)  # size of history
        dQ = []

        # As long as there is no clear charge convergence or divergence
        conv, div = False, False
        tf, yf = t0, y0
        while not conv and not div:

            # Integrate system for small interval and retrieve final charge deviation
            t0, y0 = tf, yf
            sol = solve_ivp(
                lambda t, y: self.effDerivatives(y, t, lkp),
                [t0, t0 + QSS_INTEGRATION_INTERVAL], y0,
                method='LSODA'
            )
            tf, yf = sol.t[-1], sol.y[:, -1]
            dQ.append(yf[0] - Qm0)

            # logger.debug('{:.0f} ms: dQ = {:.5f} nC/cm2, avg dQ = {:.5f} nC/cm2'.format(
            #     tf * 1e3, dQ[-1] * 1e5, np.mean(dQ[-n:]) * 1e5))

            # If last charge deviation is too large -> divergence
            if np.abs(dQ[-1]) > QSS_Q_DIV_THR:
                div = True

            # If last charge deviation or average deviation in recent history
            # is small enough -> convergence
            for x in [dQ[-1], np.mean(dQ[-n:])]:
                if np.abs(x) < QSS_Q_CONV_THR:
                    conv = True

            # If max integration duration is been reached -> error
            if tf > QSS_MAX_INTEGRATION_DURATION:
                raise ValueError('too many iterations')

        logger.debug('{}vergence after {:.0f} ms: dQ = {:.5f} nC/cm2'.format(
            {True: 'con', False: 'di'}[conv], tf * 1e3, dQ[-1] * 1e5))

        return conv


    def quasiSteadyStateFixedPoints(self, Fdrive, Adrive, DC, lkp, dQdt):
        ''' Compute QSS fixed points along the charge dimension for a given combination
            of US parameters, and determine their stability.

            :param Fdrive: US frequency (Hz)
            :param Adrive: US amplitude (Pa)
            :param DC: duty cycle (-)
            :param lkp: lookup dictionary for effective variables along charge dimension
            :param dQdt: charge derivative profile along charge dimension
            :return: 2-tuple with values of stable and unstable fixed points
        '''

        logger.debug('A = {:.2f} kPa, DC = {:.0f}%'.format(Adrive * 1e-3, DC * 1e2))

        # Extract stable and unstable fixed points from QSS charge variation profile
        dfunc = lambda Qm: - self.quasiSteadyStateiNet(Qm, Fdrive, Adrive, DC)
        SFP_candidates = getFixedPoints(lkp['Q'], dQdt, filter='stable', der_func=dfunc).tolist()
        UFPs = getFixedPoints(lkp['Q'], dQdt, filter='unstable', der_func=dfunc).tolist()
        SFPs = []

        pltvars = self.getPltVars()

        # For each candidate SFP
        for i, Qm in enumerate(SFP_candidates):

            logger.debug('Q-SFP = {:.2f} nC/cm2'.format(Qm * 1e5))

            # Re-compute QSS
            *_, QSS_FP = self.quasiSteadyStates(Fdrive, amps=Adrive, charges=Qm, DCs=DC,
                                                squeeze_output=True)

            # Simulate from unperturbed QSS and evaluate stability
            if not self.evaluateStability(Qm, QSS_FP, lkp):
                logger.warning('diverging system at ({:.2f} kPa, {:.2f} nC/cm2)'.format(
                    Adrive * 1e-3, Qm * 1e5))
                UFPs.append(Qm)
            else:
                # For each state
                unstable_states = []
                for x in self.neuron.states:
                    pltvar = pltvars[x]
                    unit_str = pltvar.get('unit', '')
                    factor = pltvar.get('factor', 1)
                    is_stable_direction = []
                    for sign in [-1, +1]:
                        # Perturb state with small offset
                        QSS_perturbed = deepcopy(QSS_FP)
                        QSS_perturbed[x] *= (1 + sign * QSS_REL_OFFSET)

                        # If gating state, bound within [0., 1.]
                        if self.neuron.isVoltageGated(x):
                            QSS_perturbed[x] = np.clip(QSS_perturbed[x], 0., 1.)

                        logger.debug('{}: {:.5f} -> {:.5f} {}'.format(
                            x, QSS_FP[x] * factor, QSS_perturbed[x] * factor, unit_str))

                        # Simulate from perturbed QSS and evaluate stability
                        is_stable_direction.append(
                            self.evaluateStability(Qm, QSS_perturbed, lkp))

                    # Check if system shows stability upon x-state perturbation
                    # in both directions
                    if not np.all(is_stable_direction):
                        unstable_states.append(x)

                # Classify fixed point as stable only if all states show stability
                is_stable_FP = len(unstable_states) == 0
                {True: SFPs, False: UFPs}[is_stable_FP].append(Qm)
                logger.info('{}stable fixed-point at ({:.2f} kPa, {:.2f} nC/cm2){}'.format(
                    '' if is_stable_FP else 'un', Adrive * 1e-3, Qm * 1e5,
                    '' if is_stable_FP else ', caused by {} states'.format(unstable_states)))

            return SFPs, UFPs


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

    def computeEffVars(self, Fdrive, Adrive, Qm, fs):
        ''' Compute "effective" coefficients of the HH system for a specific
            combination of stimulus frequency, stimulus amplitude and charge density.

            A short mechanical simulation is run while imposing the specific charge density,
            until periodic stabilization. The HH coefficients are then averaged over the last
            acoustic cycle to yield "effective" coefficients.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param Qm: imposed charge density (C/m2)
            :param fs: list of sonophore membrane coverage fractions
            :return: list with computation time and a list of dictionaries of effective variables
        '''

        # Run simulation and retrieve deflection and gas content vectors from last cycle
        data, tcomp = BilayerSonophore.simulate(self, Fdrive, Adrive, Qm)
        Z_last = data.loc[-NPC_FULL:, 'Z'].values  # m
        Cm_last = self.v_Capct(Z_last)  # F/m2

        # For each coverage fraction
        effvars = []
        for x in fs:
            # Compute membrane capacitance and membrane potential vectors
            Cm = x * Cm_last + (1 - x) * self.Cm0  # F/m2
            Vm = Qm / Cm * 1e3  # mV

            # Compute average cycle value for membrane potential and rate constants
            effvars.append({'V': np.mean(Vm)})
            effvars[-1].update(self.neuron.computeEffRates(Vm))

        # Log process
        log = '{}: lookups @ {}Hz, {}Pa, {:.2f} nC/cm2'.format(
            self, *si_format([Fdrive, Adrive], precision=1, space=' '), Qm * 1e5)
        if len(fs) > 1:
            log += ', fs = {:.0f} - {:.0f}%'.format(fs.min() * 1e2, fs.max() * 1e2)
        log += ', tcomp = {:.3f} s'.format(tcomp)
        logger.info(log)

        # Return effective coefficients
        return [tcomp, effvars]
