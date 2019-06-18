# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2016-09-29 16:16:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-18 16:09:10

from copy import deepcopy
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

from .simulators import PWSimulator, HybridSimulator, PeriodicSimulator
from .bls import BilayerSonophore
from .pneuron import PointNeuron
from .model import Model
from .batches import createQueue
from ..neurons import getLookups2D, getLookupsDCavg
from ..utils import *
from ..constants import *
from ..postpro import getFixedPoints


class NeuronalBilayerSonophore(BilayerSonophore):
    ''' This class inherits from the BilayerSonophore class and receives an PointNeuron instance
        at initialization, to define the electro-mechanical NICE model and its SONIC variant. '''

    tscale = 'ms'  # relevant temporal scale of the model
    simkey = 'ASTIM'  # keyword used to characterize simulations made with this model

    def __init__(self, a, pneuron, Fdrive=None, embedding_depth=0.0):
        ''' Constructor of the class.

            :param a: in-plane radius of the sonophore structure within the membrane (m)
            :param pneuron: point-neuron model
            :param Fdrive: frequency of acoustic perturbation (Hz)
            :param embedding_depth: depth of the embedding tissue around the membrane (m)
        '''
        # Check validity of input parameters
        if not isinstance(pneuron, PointNeuron):
            raise ValueError('Invalid neuron type: "{}" (must inherit from PointNeuron class)'
                             .format(pneuron.name))
        self.pneuron = pneuron

        # Initialize BilayerSonophore parent object
        BilayerSonophore.__init__(self, a, pneuron.Cm0, pneuron.Qm0, embedding_depth)

    def __repr__(self):
        s = '{}({:.1f} nm, {}'.format(self.__class__.__name__, self.a * 1e9, self.pneuron)
        if self.d > 0.:
            s += ', d={}m'.format(si_format(self.d, precision=1, space=' '))
        return s + ')'

    def params(self):
        params = super().params()
        params.update(self.pneuron.params())
        return params

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        pltvars = super().getPltVars(wrapleft, wrapright)
        pltvars.update(self.pneuron.getPltVars(wrapleft, wrapright))
        return pltvars

    def getPltScheme(self):
        return self.pneuron.getPltScheme()

    def filecode(self, *args):
        return Model.filecode(self, *args)

    def inputVars(self):
        # Get parent input vars and supress irrelevant entries
        bls_vars = super().inputVars()
        pneuron_vars = self.pneuron.inputVars()
        del bls_vars['Qm']
        del pneuron_vars['Astim']

        # Fill in current input vars in appropriate order
        inputvars = bls_vars
        inputvars.update(pneuron_vars)
        inputvars['method'] = None
        return inputvars

    def filecodes(self, Fdrive, Adrive, tstim, toffset, PRF, DC, method='sonic'):
        # Get parent codes and supress irrelevant entries
        bls_codes = super().filecodes(Fdrive, Adrive, 0.0)
        pneuron_codes = self.pneuron.filecodes(0.0, tstim, toffset, PRF, DC)
        for x in [bls_codes, pneuron_codes]:
            del x['simkey']
        del bls_codes['Qm']
        del pneuron_codes['Astim']

        # Fill in current codes in appropriate order
        codes = {
            'simkey': self.simkey,
            'neuron': pneuron_codes.pop('neuron'),
            'nature': pneuron_codes.pop('nature')
        }
        codes.update(bls_codes)
        codes.update(pneuron_codes)
        codes['method'] = method
        return codes

    def fullDerivatives(self, t, y, Adrive, Fdrive, phi):
        ''' Compute the derivatives of the (n+3) ODE full NBLS system variables.

            :param t: specific instant in time (s)
            :param y: vector of state variables
            :param Adrive: acoustic drive amplitude (Pa)
            :param Fdrive: acoustic drive frequency (Hz)
            :param phi: acoustic drive phase (rad)
            :return: vector of derivatives
        '''
        dydt_mech = BilayerSonophore.derivatives(self, t, y[:3], Adrive, Fdrive, y[3], phi)
        dydt_elec = self.pneuron.Qderivatives(t, y[3:], self.Capct(y[1]))
        return dydt_mech + dydt_elec

    def effDerivatives(self, t, y, lkp):
        ''' Compute the derivatives of the n-ODE effective HH system variables,
            based on 1-dimensional linear interpolation of "effective" coefficients
            that summarize the system's behaviour over an acoustic cycle.

            :param t: specific instant in time (s)
            :param y: vector of HH system variables at time t
            :param lkp: dictionary of 1D data points of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :return: vector of effective system derivatives at time t
        '''
        Qm, *states = y
        states_dict = dict(zip(self.states, states))
        rates = self.pneuron.interpEffRates(Qm, lkp)
        Vmeff = self.pneuron.interpVmeff(Qm, lkp)
        dQmdt = - self.pneuron.iNet(Vmeff, states_dict) * 1e-3
        dstates = self.pneuron.derEffStates(Vmeff, states_dict, rates)
        return [dQmdt, *[dstates[k] for k in self.pneuron.states]]

    def interpEffVariable(self, key, Qm, stim, lkps1D):
        ''' Interpolate Q-dependent effective variable along solution.

            :param key: lookup variable key
            :param Qm: charge density solution vector
            :param stim: stimulation state solution vector
            :param lkps1D: dictionary of lookups for ON and OFF states
            :return: interpolated effective variable vector
        '''
        x = np.zeros(stim.size)
        x[stim == 0] = np.interp(
            Qm[stim == 0], lkps1D['OFF']['Q'], lkps1D['OFF'][key], left=np.nan, right=np.nan)
        x[stim == 1] = np.interp(
            Qm[stim == 1], lkps1D['ON']['Q'], lkps1D['ON'][key], left=np.nan, right=np.nan)
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
        dt = 1 / (NPC_DENSE * Fdrive)

        # Compute non-zero deflection value for a small perturbation (solving quasi-steady equation)
        Pac = self.Pacoustic(dt, Adrive, Fdrive, phi)
        Z0 = self.balancedefQS(self.ng0, self.Qm0, Pac)

        # Set initial conditions
        steady_states = self.pneuron.steadyStates(self.pneuron.Vm0)
        y0 = np.concatenate((
            [0., Z0, self.ng0, self.Qm0],
            [steady_states[k] for k in self.pneuron.states]))

        # Initialize simulator and compute solution
        logger.debug('Computing detailed solution')
        simulator = PWSimulator(
            lambda t, y: self.fullDerivatives(t, y, Adrive, Fdrive, phi),
            lambda t, y: self.fullDerivatives(t, y, 0., 0., 0.))
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
        for i in range(len(self.pneuron.states)):
            data[self.pneuron.states[i]] = y[:, i + 4]

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
        # Determine time steps
        dt_dense, dt_sparse = [1. / (n * Fdrive) for n in [NPC_DENSE, NPC_SPARSE]]

        # Compute non-zero deflection value for a small perturbation (solving quasi-steady equation)
        Pac = self.Pacoustic(dt_dense, Adrive, Fdrive, phi)
        Z0 = self.balancedefQS(self.ng0, self.Qm0, Pac)

        # Set initial conditions
        steady_states = self.pneuron.steadyStates(self.pneuron.Vm0)
        y0 = np.concatenate((
            [0., Z0, self.ng0, self.Qm0],
            [steady_states[k] for k in self.pneuron.states],
        ))
        is_dense_var = np.array([True] * 3 + [False] * (len(self.pneuron.states) + 1))

        # Initialize simulator and compute solution
        logger.debug('Computing hybrid solution')
        simulator = HybridSimulator(
            lambda t, y: self.fullDerivatives(t, y, Adrive, Fdrive, phi),
            lambda t, y: self.fullDerivatives(t, y, 0., 0., 0.),
            lambda t, y, Cm: self.pneuron.Qderivatives(t, y, Cm),
            lambda yref: self.Capct(yref[1]),
            is_dense_var,
            ivars_to_check=[1, 2])
        (t, y, stim), tcomp = simulator(
            y0, dt_dense, dt_sparse, 1. / Fdrive, tstim, toffset, PRF, DC,
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
        for i in range(len(self.pneuron.states)):
            data[self.pneuron.states[i]] = y[:, i + 4]

        # Return dataframe and computation time
        return data, tcomp

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
        Z_last = data.loc[-NPC_DENSE:, 'Z'].values  # m
        Cm_last = self.v_Capct(Z_last)  # F/m2

        # For each coverage fraction
        effvars = []
        for x in fs:
            # Compute membrane capacitance and membrane potential vectors
            Cm = x * Cm_last + (1 - x) * self.Cm0  # F/m2
            Vm = Qm / Cm * 1e3  # mV

            # Compute average cycle value for membrane potential and rate constants
            effvars.append({'V': np.mean(Vm)})
            effvars[-1].update(self.pneuron.computeEffRates(Vm))

        # Log process
        log = '{}: lookups @ {}Hz, {}Pa, {:.2f} nC/cm2'.format(
            self, *si_format([Fdrive, Adrive], precision=1, space=' '), Qm * 1e5)
        if len(fs) > 1:
            log += ', fs = {:.0f} - {:.0f}%'.format(fs.min() * 1e2, fs.max() * 1e2)
        log += ', tcomp = {:.3f} s'.format(tcomp)
        logger.info(log)

        # Return effective coefficients
        return [tcomp, effvars]

    # def buildEffDerivatives(self, pneuron):
    #     states_names = pneuron.states
    #     eff_dstates = {}
    #     for k in states_names:
    #         der_func_str = 'der{}'.format(k.upper())
    #         if hasattr(pneuron, der_func_str):
    #             der_func = getattr(pneuron, der_func_str)
    #             args_str = inspect.getargspec(cfunc)[0][1:]
    #             args = [states.index(a) for a in args_str]
    #             eff_dstates[k] = der_func(states*cargs))
    #         else:
    #             eff_dstates[k] = lambda x, Qm, lkp: effDerivative(x, Qm, lkp, k)

    #     def effDerivatives(Qm, states, lkp):
    #         return eff_dstates

    #     return effDerivatives

    def runSONIC(self, Fdrive, Adrive, tstim, toffset, PRF, DC):
        ''' Compute solutions of the system for a specific set of
            US stimulation parameters, using charge-predicted "effective"
            coefficients to solve the HH equations at each step.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        '''
        # Load appropriate 2D lookups
        Aref, Qref, lkps2D, _ = getLookups2D(self.pneuron.name, a=self.a, Fdrive=Fdrive)

        # Check that acoustic amplitude is within lookup range
        Adrive = isWithin('amplitude', Adrive, (Aref.min(), Aref.max()))

        # Interpolate 2D lookups at zero and US amplitude
        logger.debug('Interpolating lookups at A = %.2f kPa and A = 0', Adrive * 1e-3)
        lkps1D = {state: {key: interp1d(Aref, y2D, axis=0)(val) for key, y2D in lkps2D.items()}
                  for state, val in {'ON': Adrive, 'OFF': 0.}.items()}

        # Add reference charge vector to 1D lookup dictionaries
        for state in lkps1D.keys():
            lkps1D[state]['Q'] = Qref

        # Set initial conditions
        steady_states = self.pneuron.steadyStates(self.pneuron.Vm0)
        y0 = np.insert(np.array([steady_states[k] for k in self.pneuron.states]), 0, self.Qm0)

        # Initialize simulator and compute solution
        logger.debug('Computing effective solution')
        simulator = PWSimulator(
            lambda t, y: self.effDerivatives(t, y, lkps1D['ON']),
            lambda t, y: self.effDerivatives(t, y, lkps1D['OFF']))
        (t, y, stim), tcomp = simulator(
            y0, DT_EFFECTIVE, tstim, toffset, PRF, DC, monitor_time=True)
        logger.debug('completed in %ss', si_format(tcomp, 1))

        # Store output in dataframe
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Qm': y[:, 0]
        })
        data['Vm'] = self.interpEffVariable('V', data['Qm'].values, stim, lkps1D)
        for key in ['Z', 'ng']:
            data[key] = np.full(t.size, np.nan)
        for i in range(len(self.pneuron.states)):
            data[self.pneuron.states[i]] = y[:, i + 1]

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
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
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

    def simulate(self, Fdrive, Adrive, tstim, toffset, PRF=100., DC=1.0, method='sonic'):
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
            '%s: simulation @ f = %sHz, A = %sPa, t = %ss (%ss offset)%s',
            self, si_format(Fdrive, 0, space=' '), si_format(Adrive, 2, space=' '),
            *si_format([tstim, toffset], 1, space=' '),
            (', PRF = {}Hz, DC = {:.2f}%'.format(
                si_format(PRF, 2, space=' '), DC * 1e2) if DC < 1.0 else ''))

        # Check validity of stimulation parameters
        BilayerSonophore.checkInputs(self, Fdrive, Adrive, 0.0, 0.0)
        self.pneuron.checkInputs(Adrive, tstim, toffset, PRF, DC)

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
        nspikes = self.pneuron.getNSpikes(data)
        logger.debug('{} spike{} detected'.format(nspikes, plural(nspikes)))

        # Return dataframe and computation time
        return data, tcomp

    @logCache(os.path.join(os.path.split(__file__)[0], 'astim_titrations.log'))
    def titrate(self, Fdrive, tstim, toffset, PRF=100., DC=1., method='sonic',
                xfunc=None, Arange=None):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given frequency, duration, PRF and duty cycle.

            :param Fdrive: US frequency (Hz)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param method: integration method
            :param xfunc: function determining whether condition is reached from simulation output
            :param Arange: search interval for Adrive, iteratively refined
            :return: determined threshold amplitude (Pa)
        '''
        # Default output function
        if xfunc is None:
            xfunc = self.pneuron.titrationFunc

        # Default amplitude interval
        if Arange is None:
            Arange = [0., getLookups2D(self.pneuron.name, a=self.a, Fdrive=Fdrive)[0].max()]

        return binarySearch(
            lambda x: xfunc(self.simulate(*x)[0]),
            [Fdrive, tstim, toffset, PRF, DC, method], 1, Arange, THRESHOLD_CONV_RANGE_ASTIM
        )

    def simQueue(self, freqs, amps, durations, offsets, PRFs, DCs, methods, outputdir=None):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps, while avoiding repetition of CW protocols for a given PRF sweep.

            :param freqs: list (or 1D-array) of US frequencies
            :param amps: list (or 1D-array) of acoustic amplitudes
            :param durations: list (or 1D-array) of stimulus durations
            :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
            :param PRFs: list (or 1D-array) of pulse-repetition frequencies
            :param DCs: list (or 1D-array) of duty cycle values
            :params methods: integration methods
            :return: list of parameters (list) for each simulation
        '''
        method_ids = list(range(len(methods)))
        if amps is None:
            amps = [np.nan]
        DCs = np.array(DCs)
        queue = []
        if 1.0 in DCs:
            queue += createQueue(freqs, amps, durations, offsets, min(PRFs), 1.0, method_ids)
        if np.any(DCs != 1.0):
            queue += createQueue(freqs, amps, durations, offsets, PRFs, DCs[DCs != 1.0], method_ids)
        for item in queue:
            if np.isnan(item[1]):
                item[1] = None
            item[-1] = methods[int(item[-1])]
        if outputdir is not None:
            for item in queue:
                item.insert(0, outputdir)
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
            self.pneuron.name, self.a, Fdrive, amps, charges, DCs)

        # Compute QSS states using these lookups
        nA, nQ, nDC = lookups['V'].shape
        QSS = {k: np.empty((nA, nQ, nDC)) for k in self.pneuron.states}
        for iA in range(nA):
            for iDC in range(nDC):
                QSS_1D = self.pneuron.quasiSteadyStates(
                    {k: v[iA, :, iDC] for k, v in lookups.items()})
                for k in QSS.keys():
                    QSS[k][iA, :, iDC] = QSS_1D[k]

        # Compress outputs if needed
        if squeeze_output:
            QSS = {k: v.squeeze() for k, v in QSS.items()}
            lookups = {k: v.squeeze() for k, v in lookups.items()}

        # Return reference inputs and outputs
        return amps, charges, lookups, QSS

    def iNetQSS(self, Qm, Fdrive, Adrive, DC):
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
        return self.pneuron.iNet(lookups['V'], np.array(list(QSS.values())))  # mA/m2

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

        # Initialize simulator and compute solution
        # simulator = PeriodicSimulator(
        #     lambda t, y: self.effDerivatives(t, y, lkp),
        #     ivars_to_check=[0])
        # simulator.stopfunc = simulator.stopFuncTmp
        # simulator.refs = [Qm0]
        # simulator.conv_thr = [QSS_Q_CONV_THR]
        # simulator.div_thr = [QSS_Q_DIV_THR]
        # simulator.t_history = QSS_HISTORY_INTERVAL
        # t, y, stim = simulator.compute(
        #     y0,
        #     QSS_INTEGRATION_INTERVAL,
        #     QSS_INTEGRATION_INTERVAL,
        #     nmax=int(QSS_MAX_INTEGRATION_DURATION // QSS_INTEGRATION_INTERVAL)
        # )
        # conv = simulator.isAsymptoticallyStable(t, y, QSS_INTEGRATION_INTERVAL) != -1
        # tf = t[-1]
        # dQ = [y[-1, 0]]

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
                lambda t, y: self.effDerivatives(t, y, lkp),
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
                logger.warning('too many iterations (dQ = {:.5f} nC/cm2)'.format(dQ[-1] * 1e5))
                conv = True

        logger.debug('{}vergence after {:.0f} ms: dQ = {:.5f} nC/cm2'.format(
            {True: 'con', False: 'di'}[conv], tf * 1e3, dQ[-1] * 1e5))

        return conv

    def fixedPointsQSS(self, Fdrive, Adrive, DC, lkp, dQdt):
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
        def dfunc(Qm):
            return - self.iNetQSS(Qm, Fdrive, Adrive, DC)
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
                for x in self.pneuron.states:
                    pltvar = pltvars[x]
                    unit_str = pltvar.get('unit', '')
                    factor = pltvar.get('factor', 1)
                    is_stable_direction = []
                    for sign in [-1, +1]:
                        # Perturb state with small offset
                        QSS_perturbed = deepcopy(QSS_FP)
                        QSS_perturbed[x] *= (1 + sign * QSS_REL_OFFSET)

                        # If gating state, bound within [0., 1.]
                        if self.pneuron.isVoltageGated(x):
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

    def isStableQSS(self, Fdrive, Adrive, DC):
            _, Qref, lookups, QSS = self.quasiSteadyStates(
                Fdrive, amps=Adrive, DCs=DC, squeeze_output=True)
            lookups['Q'] = Qref
            dQdt = -self.pneuron.iNet(
                lookups['V'], np.array([QSS[k] for k in self.pneuron.states]))  # mA/m2
            SFPs, _ = self.fixedPointsQSS(Fdrive, Adrive, DC, lookups, dQdt)
            return len(SFPs) > 0

    def titrateQSS(self, Fdrive, DC=1., Arange=None):

        # Default amplitude interval
        if Arange is None:
            Arange = [0., getLookups2D(self.pneuron.name, a=self.a, Fdrive=Fdrive)[0].max()]

        # Titration function
        def xfunc(x):
            if self.pneuron.name == 'STN':
                return self.isStableQSS(*x)
            else:
                return not self.isStableQSS(*x)

        return binarySearch(
            xfunc,
            [Fdrive, DC], 1, Arange, THRESHOLD_CONV_RANGE_ASTIM)
