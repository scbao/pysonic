#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-29 16:16:19
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-30 19:33:45

from copy import deepcopy
import time
import logging
import pickle
import progressbar as pb
import numpy as np
import pandas as pd
from scipy.integrate import ode, odeint, solve_ivp
from scipy.interpolate import interp1d

from .simulators import PWSimulator, HybridSimulator
from .bls import BilayerSonophore
from .pneuron import PointNeuron
from ..utils import *
from ..constants import *
from ..postpro import findPeaks, getFixedPoints


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

    def filecode(self, Fdrive, Adrive, tstim, PRF, DC, method):
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
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
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
            [steady_states[k] for k in self.neuron.states],
        ))

        # Initialize simulator and compute solution
        logger.debug('Computing detailed solution')
        simulator = PWSimulator(
            lambda y, t: self.fullDerivatives(y, t, Adrive, Fdrive, phi),
            lambda y, t: self.fullDerivatives(y, t, 0., 0., 0.),
        )
        t, y, stim = simulator.compute(
            y0, dt, tstim, toffset, PRF, DC,
            print_progress=logger.getEffectiveLevel() <= logging.INFO,
            target_dt=CLASSIC_TARGET_DT
        )

        # Compute membrane potential vector (in mV)
        Qm = y[:, 3]
        Z = y[:, 1]
        Vm = Qm / self.v_Capct(Z) * 1e3  # mV

        # Add Vm to solution matrix
        y = np.hstack((
            y[:, 1:4],
            np.array([Vm]).T,
            y[:, 4:]
        ))

        # Return output variables
        return t, y.T, stim

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
            0, self.Qm0
        )

        # Initialize simulator and compute solution
        logger.debug('Computing effective solution')
        simulator = PWSimulator(
            lambda y, t: self.effDerivatives(y, t, lookups_on),
            lambda y, t: self.effDerivatives(y, t, lookups_off)
        )
        t, y, stim = simulator.compute(y0, dt, tstim, toffset, PRF, DC)
        Qm = y[:, 0]

        # Compute effective gas content and membrane potential vectors
        ng, Vm = [
            self.interpEffVariable(key, Qm, stim, lookups_on, lookups_off)
            for key in ['ng', 'V']
        ]

        # Compute quasi-steady deflection vector
        Z = np.array([self.balancedefQS(x1, x2) for x1, x2 in zip(ng, Qm)])  # m

        # Add Z, ng and Vm to solution matrix
        y = np.hstack((
            np.array([Z, ng, Qm, Vm]).T,
            y[:, 1:]
        ))

        # Return output variables
        return t, y.T, stim


    def runHybrid(self, Fdrive, Adrive, tstim, toffset, PRF, DC, phi=np.pi):
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
            is_dense_var=is_dense_var,
            ivars_to_check=[1, 2]
        )
        t, y, stim = simulator.compute(
            y0, dt_dense, dt_sparse, Fdrive, tstim, toffset, PRF, DC,
            # print_progress=logger.getEffectiveLevel() <= logging.INFO
        )

        # Compute membrane potential vector (in mV)
        Qm = y[:, 3]
        Z = y[:, 1]
        Vm = Qm / self.v_Capct(Z) * 1e3  # mV

        # Add Vm to solution matrix
        y = np.hstack((
            y[:, 1:4],
            np.array([Vm]).T,
            y[:, 4:]
        ))

        # Return output variables
        return t, y.T, stim


    def checkInputs(self, Fdrive, Adrive, tstim, toffset, PRF, DC, method):
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


    def simulate(self, Fdrive, Adrive, tstim, toffset, PRF=None, DC=1.0, method='sonic'):
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
        self.checkInputs(Fdrive, Adrive, tstim, toffset, PRF, DC, method)

        # Call appropriate simulation function
        if method == 'full':
            return self.runFull(Fdrive, Adrive, tstim, toffset, PRF, DC)
        elif method == 'sonic':
            return self.runSONIC(Fdrive, Adrive, tstim, toffset, PRF, DC)
        elif method == 'hybrid':
            # if DC < 1.0:
            #     raise ValueError('Pulsed protocol incompatible with hybrid integration method')
            return self.runHybrid(Fdrive, Adrive, tstim, toffset, PRF, DC)


    def isExcited(self, Adrive, Fdrive, tstim, toffset, PRF, DC, method, return_val=False):
        ''' Run a simulation and determine if neuron is excited.

            :param Adrive: acoustic amplitude (Pa)
            :param Fdrive: US frequency (Hz)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :return: boolean stating whether neuron is excited or not
        '''
        t, y, _ = self.simulate(Fdrive, Adrive, tstim, toffset, PRF, DC, method=method)
        dt = t[1] - t[0]
        ipeaks, *_ = findPeaks(y[2, :], SPIKE_MIN_QAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                               SPIKE_MIN_QPROM)
        nspikes = ipeaks.size
        logger.debug('A = %sPa ---> %s spike%s detected',
                     si_format(Adrive, 2, space=' '),
                     nspikes, "s" if nspikes > 1 else "")
        cond = nspikes > 0
        if return_val:
            return {True: nspikes, False: np.nan}[cond]
        else:
            return cond


    def isSilenced(self, Adrive, Fdrive, tstim, toffset, PRF, DC, method, return_val=False):
        ''' Run a simulation and determine if neuron is silenced.

            :param Adrive: acoustic amplitude (Pa)
            :param Fdrive: US frequency (Hz)
            :param tstim: duration of US stimulation (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :return: boolean stating whether neuron is silenced or not
        '''
        if tstim <= TMIN_STABILIZATION:
            raise ValueError(
                'stimulus duration must be greater than {:.0f} ms'.format(TMIN_STABILIZATION * 1e3))

        # Simulate model without offset
        t, y, _ = self.simulate(Fdrive, Adrive, tstim, 0., PRF, DC, method=method)

        # Extract charge signal posterior to observation window
        Qm = y[2, t > TMIN_STABILIZATION]

        # Compute variation range
        Qm_range = np.ptp(Qm)
        logger.debug('A = %sPa ---> %.2f nC/cm2 variation range over the last %.0f ms',
                     si_format(Adrive, 2, space=' '), Qm_range * 1e5, TMIN_STABILIZATION * 1e3)

        cond = np.ptp(Qm) < QSS_Q_DIV_THR
        if return_val:
            return {True: Qm[-1], False: np.nan}[cond]
        else:
            return cond


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

    def run(self, Fdrive, tstim, toffset, PRF=None, DC=1.0, Adrive=None, method='sonic'):
        ''' Run a simulation of the full electro-mechanical system for a given neuron type
            with specific parameters, and return output data and metadata.

            :param Fdrive: US frequency (Hz)
            :param tstim: stimulus duration (s)
            :param toffset: stimulus offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: stimulus duty cycle (-)
            :param Adrive: acoustic pressure amplitude (Pa)
            :param method: integration method
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

        # If no amplitude provided, perform titration
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
        logger.debug('completed in %ss', si_format(tcomp, 1))

        # Store dataframe and metadata
        data = pd.DataFrame({
            't': t,
            'stimstate': stimstate,
            'Z': Z,
            'ng': ng,
            'Qm': Qm,
            'Vm': Vm
        })
        for j in range(len(self.neuron.states)):
            data[self.neuron.states[j]] = channels[j]
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

        # Log number of detected spikes
        self.neuron.logNSpikes(data)

        return data, meta

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
        data, meta = self.run(Fdrive, tstim, toffset, PRF, DC, Adrive, method)
        simcode = self.filecode(Fdrive, Adrive, tstim, PRF, DC, method)
        outpath = '{}/{}.pkl'.format(outdir, simcode)
        with open(outpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': data}, fh)
        logger.debug('simulation data exported to "%s"', outpath)
        return outpath


    # def runIfNone(self, outdir, Fdrive, Adrive, tstim, toffset, PRF=None, DC=1.0, Adrive=None,
    #               method='sonic'):
    #     ''' Run a simulation of the full electro-mechanical system for a given neuron type
    #         with specific parameters, and save the results in a PKL file,
    #         only if file not present.

    #         :param outdir: full path to output directory
    #         :param Fdrive: US frequency (Hz)
    #         :param tstim: stimulus duration (s)
    #         :param toffset: stimulus offset (s)
    #         :param PRF: pulse repetition frequency (Hz)
    #         :param DC: stimulus duty cycle (-)
    #         :param Adrive: acoustic pressure amplitude (Pa)
    #         :param method: integration method
    #     '''
    #     fname = self.filecode(Fdrive, Adrive, tstim, PRF, DC, method)
    #     fpath = os.path.join(outdir, fname)
    #     if not os.path.isfile(fpath):
    #         logger.warning('"{}"" file not found'.format(fname))
    #         self.runAndSave(outdir=Fdrive, tstim, toffset, PRF, DC, Adrive, method)
    #     return loadData(fpath)


    def getStabPoints():
        # Simulate model without offset
        t, y, _ = self.simulate(Fdrive, Adrive, tstim, 0., PRF, DC, method=method)

        # Extract charge signal posterior to observation window
        Qm = y[2, t > TMIN_STABILIZATION]

        # Compute variation range
        Qm_range = np.ptp(Qm)
        logger.debug('A = %sPa ---> %.2f nC/cm2 variation range over the last %.0f ms',
                     si_format(Adrive, 2, space=' '), Qm_range * 1e5, TMIN_STABILIZATION * 1e3)

        cond = np.ptp(Qm) < QSS_Q_DIV_THR
        if return_val:
            return {True: Qm[-1], False: np.nan}[cond]
        else:
            return cond


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

        tstart = time.time()

        # Run simulation and retrieve deflection and gas content vectors from last cycle
        _, [Z, ng], _ = BilayerSonophore.simulate(self, Fdrive, Adrive, Qm)
        Z_last = Z[-NPC_FULL:]  # m
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

        tcomp = time.time() - tstart

        # Log process
        log = '{}: lookups @ {}Hz, {}Pa, {:.2f} nC/cm2'.format(
            self, *si_format([Fdrive, Adrive], precision=1, space=' '), Qm * 1e5)
        if len(fs) > 1:
            log += ', fs = {:.0f} - {:.0f}%'.format(fs.min() * 1e2, fs.max() * 1e2)
        log += ', tcomp = {:.3f} s'.format(tcomp)
        logger.info(log)

        # Return effective coefficients
        return [tcomp, effvars]
