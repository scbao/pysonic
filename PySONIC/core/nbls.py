# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2016-09-29 16:16:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-02-03 14:22:29

import time
from copy import deepcopy
import logging
import numpy as np
import pandas as pd

from .simulators import PWSimulator, HybridSimulator, PeriodicSimulator
from .bls import BilayerSonophore
from .pneuron import PointNeuron
from .model import Model
from .sources import *
from .protocols import TimeProtocol, PulsedProtocol, createPulsedProtocols
from ..utils import *
from ..threshold import threshold
from ..constants import *
from ..postpro import getFixedPoints
from .lookups import EffectiveVariablesLookup
from ..neurons import getPointNeuron


NEURONS_LOOKUP_DIR = os.path.abspath(os.path.split(__file__)[0] + "/../neurons/")


class NeuronalBilayerSonophore(BilayerSonophore):
    ''' This class inherits from the BilayerSonophore class and receives an PointNeuron instance
        at initialization, to define the electro-mechanical NICE model and its SONIC variant. '''

    tscale = 'ms'  # relevant temporal scale of the model
    simkey = 'ASTIM'  # keyword used to characterize simulations made with this model
    titration_obj = 'US_source'  # name of the object containing the titration variable
    titration_var = 'Adrive'  # name of the titration variable

    def __init__(self, a, pneuron, embedding_depth=0.0):
        ''' Constructor of the class.

            :param a: in-plane radius of the sonophore structure within the membrane (m)
            :param pneuron: point-neuron model
            :param embedding_depth: depth of the embedding tissue around the membrane (m)
        '''
        # Check validity of input parameters
        if not isinstance(pneuron, PointNeuron):
            raise ValueError(f'Invalid neuron type: "{pneuron.name}" (must inherit from PointNeuron class)')
        self.pneuron = pneuron

        # Initialize BilayerSonophore parent object
        super().__init__(a, pneuron.Cm0, pneuron.Qm0, embedding_depth=embedding_depth)

    def __repr__(self):
        s = f'{self.__class__.__name__}({self.a * 1e9:.1f} nm, {self.pneuron}'
        if self.d > 0.:
            s += f', d={si_format(self.d, precision=1)}m'
        return f'{s})'

    @classmethod
    def initFromMeta(cls, meta):
        return cls(meta['a'], getPointNeuron(meta['neuron']), embedding_depth=meta['d'])

    def params(self):
        return {**super().params(), **self.pneuron.params()}

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        return {**super().getPltVars(wrapleft, wrapright),
                **self.pneuron.getPltVars(wrapleft, wrapright)}

    @property
    def pltScheme(self):
        return self.pneuron.pltScheme

    def filecode(self, *args):
        return Model.filecode(self, *args)

    @property
    @staticmethod
    def inputs():
        # Get parent input vars and supress irrelevant entries
        bls_vars = BilayerSonophore.inputs
        pneuron_vars = PointNeuron.inputs
        del bls_vars['Qm']
        del pneuron_vars['Astim']

        # Fill in current input vars in appropriate order
        inputvars = bls_vars
        inputvars.update(pneuron_vars)
        inputvars['fs'] = {
            'desc': 'sonophore membrane coverage fraction',
            'label': 'f_s',
            'unit': '\%',
            'factor': 1e2,
            'precision': 0
        }
        inputvars['method'] = None
        return inputvars

    def filecodes(self, US_source, pp, fs, method, qss_vars):
        codes = {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'nature': pp.nature,
            'a': f'{self.a * 1e9:.0f}nm',
            **US_source.filecodes,
            **pp.filecodes,
        }
        codes['fs'] = f'fs{fs * 1e2:.0f}%' if fs < 1 else None
        codes['method'] = method
        codes['qss_vars'] = qss_vars
        return codes

    @staticmethod
    def interpOnOffVariable(key, Qm, stim, lkp):
        ''' Interpolate Q-dependent effective variable along ON and OFF periods of a solution.

            :param key: lookup variable key
            :param Qm: charge density solution vector
            :param stim: stimulation state solution vector
            :param lkp: dictionary of lookups for ON and OFF states
            :return: interpolated effective variable vector
        '''
        x = np.zeros(stim.size)
        x[stim == 0] = lkp['OFF'].interpVar1D(Qm[stim == 0], key)
        x[stim == 1] = lkp['ON'].interpVar1D(Qm[stim == 1], key)
        return x

    @staticmethod
    def spatialAverage(fs, x, x0):
        ''' fs-modulated spatial averaging. '''
        return fs * x + (1 - fs) * x0

    @timer
    def computeEffVars(self, US_source, Qm, fs):
        ''' Compute "effective" coefficients of the HH system for a specific
            acoustic stimulus and charge density.

            A short mechanical simulation is run while imposing the specific charge density,
            until periodic stabilization. The HH coefficients are then averaged over the last
            acoustic cycle to yield "effective" coefficients.

            :param US_source: acoustic source object
            :param Qm: imposed charge density (C/m2)
            :param fs: list of sonophore membrane coverage fractions
            :return: list with computation time and a list of dictionaries of effective variables
        '''
        # Run simulation and retrieve deflection and gas content vectors from last cycle
        data = BilayerSonophore.simCycles(self, US_source, Qm)
        Z_last = data.loc[-NPC_DENSE:, 'Z'].values  # m
        Cm_last = self.v_capacitance(Z_last)  # F/m2

        # For each coverage fraction
        effvars = []
        for x in fs:
            # Compute membrane capacitance and membrane potential vectors
            Cm = self.spatialAverage(x, Cm_last, self.Cm0)  # F/m2
            Vm = Qm / Cm * 1e3  # mV

            # Compute average cycle value for membrane potential and rate constants
            effvars.append({**{'V': np.mean(Vm)}, **self.pneuron.getEffRates(Vm)})

        # Log process
        log = f'{self}: lookups @ {US_source.desc}, {Qm * 1e5:.2f} nC/cm2'
        if len(fs) > 1:
            log += f', fs = {fs.min() * 1e2:.0f} - {fs.max() * 1e2:.0f}%'
        logger.info(log)

        # Return effective coefficients
        return effvars

    def getLookupFileName(self, a=None, Fdrive=None, Adrive=None, fs=False):
        fname = f'{self.pneuron.name}_lookups'
        if a is not None:
            fname += f'_{a * 1e9:.0f}nm'
        if Fdrive is not None:
            fname += f'_{Fdrive * 1e-3:.0f}kHz'
        if Adrive is not None:
            fname += f'_{Adrive * 1e-3:.0f}kPa'
        if fs is True:
            fname += '_fs'
        return f'{fname}.pkl'

    def getLookupFilePath(self, *args, **kwargs):
        return os.path.join(NEURONS_LOOKUP_DIR, self.getLookupFileName(*args, **kwargs))

    def getLookup(self, *args, **kwargs):
        keep_tcomp = kwargs.pop('keep_tcomp', False)
        lookup_path = self.getLookupFilePath(*args, **kwargs)
        lkp = EffectiveVariablesLookup.fromPickle(lookup_path)
        if not keep_tcomp:
            del lkp.tables['tcomp']
        return lkp

    def getLookup2D(self, Fdrive, fs):
        kwargs = {'a': self.a, 'Fdrive': Fdrive, 'fs': True} if fs < 1. else {}
        lkp2d = self.getLookup(**kwargs).projectN({'a': self.a, 'f': Fdrive, 'fs': fs})
        return lkp2d

    def fullDerivatives(self, t, y, US_source, fs):
        ''' Compute the full system derivatives.

            :param t: specific instant in time (s)
            :param y: vector of state variables
            :param US_source: acoustic source object
            :param fs: sonophore membrane coverage fraction (-)
            :return: vector of derivatives
        '''
        dydt_mech = BilayerSonophore.derivatives(
            self, t, y[:3], US_source, y[3])
        dydt_elec = self.pneuron.derivatives(
            t, y[3:], Cm=self.spatialAverage(fs, self.capacitance(y[1]), self.Cm0))
        return dydt_mech + dydt_elec

    def effDerivatives(self, t, y, lkp1d, qss_vars):
        ''' Compute the derivatives of the n-ODE effective system variables,
            based on 1-dimensional linear interpolation of "effective" coefficients
            that summarize the system's behaviour over an acoustic cycle.

            :param t: specific instant in time (s)
            :param y: vector of HH system variables at time t
            :param lkp: dictionary of 1D data points of "effective" coefficients
             over the charge domain, for specific frequency and amplitude values.
            :param qss_vars: list of QSS variables
            :return: vector of effective system derivatives at time t
        '''
        # Unpack values and interpolate lookup at current charge density
        Qm, *states = y
        lkp0d = lkp1d.interpolate1D(Qm)

        # Compute states dictionary from differential and QSS variables
        states_dict = {}
        i = 0
        for k in self.pneuron.statesNames():
            if k in qss_vars:
                states_dict[k] = self.pneuron.quasiSteadyStates()[k](lkp0d)
            else:
                states_dict[k] = states[i]
                i += 1

        # Compute charge density derivative
        dQmdt = - self.pneuron.iNet(lkp0d['V'], states_dict) * 1e-3

        # Compute states derivative vector only for differential variable
        dstates = []
        for k in self.pneuron.statesNames():
            if k not in qss_vars:
                dstates.append(self.pneuron.derEffStates()[k](lkp0d, states_dict))

        return [dQmdt, *dstates]

    def __simFull(self, US_source, pp, fs):
        # Determine time step
        dt = US_source.dt

        # Compute initial non-zero deflection
        Z = self.computeInitialDeflection(US_source, self.Qm0, dt)

        # Set initial conditions
        ss0 = self.pneuron.getSteadyStates(self.pneuron.Vm0)
        y0 = np.concatenate(([0., 0., self.ng0, self.Qm0], ss0))
        y1 = np.concatenate(([0., Z, self.ng0, self.Qm0], ss0))

        US_source_OFF = US_source.copy()
        US_source_OFF.Adrive = 0

        # Initialize simulator and compute solution
        logger.debug('Computing detailed solution')
        simulator = PWSimulator(
            lambda t, y: self.fullDerivatives(t, y, US_source, fs),
            lambda t, y: self.fullDerivatives(t, y, US_source_OFF, fs))
        t, y, stim = simulator(
            y1, dt, pp,
            target_dt=CLASSIC_TARGET_DT,
            print_progress=logger.getEffectiveLevel() <= logging.INFO,
            monitor_func=None)
            # monitor_func=lambda t, y: f't = {t * 1e3:.5f} ms, Qm = {y[3] * 1e5:.2f} nC/cm2')

        # Prepend initial conditions (prior to stimulation)
        t, y, stim = simulator.prependSolution(t, y, stim, y0=y0)

        # Store output in dataframe and return
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Z': y[:, 1],
            'ng': y[:, 2],
            'Qm': y[:, 3]
        })
        data['Vm'] = data['Qm'].values / self.spatialAverage(
            fs, self.v_capacitance(data['Z'].values), self.Cm0) * 1e3  # mV
        for i in range(len(self.pneuron.states)):
            data[self.pneuron.statesNames()[i]] = y[:, i + 4]
        return data

    def __simHybrid(self, US_source, pp, fs):
        # Determine time steps
        dt_dense, dt_sparse = [US_source.dt, US_source.dt_sparse]

        # Compute initial non-zero deflection
        Z = self.computeInitialDeflection(US_source, self.Qm0, dt_dense)

        # Set initial conditions
        ss0 = self.pneuron.getSteadyStates(self.pneuron.Vm0)
        y0 = np.concatenate(([0., 0., self.ng0, self.Qm0], ss0))
        y1 = np.concatenate(([0., Z, self.ng0, self.Qm0], ss0))

        US_source_OFF = US_source.copy()
        US_source_OFF.Adrive = 0

        # Initialize simulator and compute solution
        is_dense_var = np.array([True] * 3 + [False] * (len(self.pneuron.states) + 1))
        logger.debug('Computing hybrid solution')
        simulator = HybridSimulator(
            lambda t, y: self.fullDerivatives(t, y, US_source, fs),
            lambda t, y: self.fullDerivatives(t, y, US_source_OFF, fs),
            lambda t, y, Cm: self.pneuron.derivatives(
                t, y, Cm=self.spatialAverage(fs, Cm, self.Cm0)),
            lambda yref: self.capacitance(yref[1]),
            is_dense_var,
            ivars_to_check=[1, 2])
        t, y, stim = simulator(y1, dt_dense, dt_sparse, US_source.periodicity, pp)

        # Prepend initial conditions (prior to stimulation)
        t, y, stim = simulator.prependSolution(t, y, stim, y0=y0)

        # Store output in dataframe and return
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Z': y[:, 1],
            'ng': y[:, 2],
            'Qm': y[:, 3]
        })
        data['Vm'] = data['Qm'].values / self.spatialAverage(
            fs, self.v_capacitance(data['Z'].values), self.Cm0) * 1e3  # mV
        for i in range(len(self.pneuron.states)):
            data[self.pneuron.statesNames()[i]] = y[:, i + 4]
        return data

    def __simSonic(self, US_source, pp, fs, qss_vars=None, pavg=False):
        # Extract US parameters
        Fdrive, Adrive = US_source.Fdrive, US_source.Adrive

        # Load appropriate 2D lookups
        lkp2d = self.getLookup2D(Fdrive, fs)

        # Interpolate 2D lookups at zero and US amplitude
        logger.debug('Interpolating lookups at A = %.2f kPa and A = 0', Adrive * 1e-3)
        lkps1d = {'ON': lkp2d.project('A', Adrive), 'OFF': lkp2d.project('A', 0.)}

        # Adapt lookups and pulsing protocol if pulse-average mode is selected
        if pavg:
            lkps1d['ON'] = lkps1d['ON'] * pp.DC + lkps1d['OFF'] * (1 - pp.DC)
            tstim = (int(pp.tstim * pp.PRF) - 1 + pp.DC) / pp.PRF
            toffset = pp.tstim + pp.toffset - tstim
            tp = TimeProtocol(tstim, toffset)

        # # Determine QSS and differential variables
        if qss_vars is None:
            qss_vars = []
        diff_vars = [item for item in self.pneuron.statesNames() if item not in qss_vars]

        # Create 1D lookup of QSS variables with reference charge vector
        QSS_1D_lkp = {
            key: EffectiveVariablesLookup(
                lkps1d['ON'].refs,
                {k: self.pneuron.quasiSteadyStates()[k](val) for k in qss_vars})
            for key, val in lkps1d.items()}

        # Set initial conditions
        sstates = [self.pneuron.steadyStates()[k](self.pneuron.Vm0) for k in diff_vars]
        y0 = np.array([self.Qm0, *sstates])

        # Initialize simulator and compute solution
        logger.debug('Computing effective solution')
        simulator = PWSimulator(
            lambda t, y: self.effDerivatives(t, y, lkps1d['ON'], qss_vars),
            lambda t, y: self.effDerivatives(t, y, lkps1d['OFF'], qss_vars))
        t, y, stim = simulator(y0, self.pneuron.chooseTimeStep(), pp)

        # Prepend initial conditions (prior to stimulation)
        t, y, stim = simulator.prependSolution(t, y, stim)

        # Store output vectors in dataframe: time, stim state, charge, potential
        # and other differential variables
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Qm': y[:, 0]
        })
        data['Vm'] = self.interpOnOffVariable('V', data['Qm'].values, stim, lkps1d)
        for key in ['Z', 'ng']:
            data[key] = np.full(t.size, np.nan)
        for i, k in enumerate(diff_vars):
            data[k] = y[:, i + 1]

        # Interpolate QSS variables along charge vector and store them in dataframe
        for k in qss_vars:
            data[k] = self.interpOnOffVariable(k, data['Qm'].values, stim, QSS_1D_lkp)

        return data

    def intMethods(self):
        ''' Listing of model integration methods. '''
        return {
                'full': self.__simFull,
                'hybrid': self.__simHybrid,
                'sonic': self.__simSonic
        }

    @classmethod
    @Model.checkOutputDir
    def simQueue(cls, freqs, amps, durations, offsets, PRFs, DCs, fs, methods, qss_vars, **kwargs):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps, while avoiding repetition of CW protocols for a given PRF sweep.

            :param freqs: list (or 1D-array) of US frequencies
            :param amps: list (or 1D-array) of acoustic amplitudes
            :param durations: list (or 1D-array) of stimulus durations
            :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
            :param PRFs: list (or 1D-array) of pulse-repetition frequencies
            :param DCs: list (or 1D-array) of duty cycle values
            :param fs: sonophore membrane coverage fractions (-)
            :params methods: integration methods
            :param qss_vars: QSS variables
            :return: list of parameters (list) for each simulation
        '''
        if ('full' in methods or 'hybrid' in methods) and kwargs['outputdir'] is None:
            logger.warning('Running cumbersome simulation(s) without file saving')

        if amps is None:
            amps = [None]
        sources = createSources(freqs, amps)
        protocols = createPulsedProtocols(durations, offsets, PRFs, DCs)
        queue = []
        for s in sources:
            for pp in protocols:
                for cov in fs:
                    for method in methods:
                        queue.append([s, pp, cov, method, qss_vars])
        return queue

    def checkInputs(self, US_source, pp, fs, method, qss_vars):
        if not isinstance(US_source, AcousticSource) and not isinstance(US_source, AcousticSourceArray):
            raise TypeError(f'Invalid "US_source" parameter (must be an "AcousticSource" object)')
        if not isinstance(pp, PulsedProtocol):
            raise TypeError('Invalid pulsed protocol (must be "PulsedProtocol" instance)')
        if not isinstance(fs, float):
            raise TypeError(f'Invalid "fs" parameter (must be float typed)')
        if qss_vars is not None:
            if not isIterable(qss_vars) or not isinstance(qss_vars[0], str):
                raise ValueError('Invalid QSS variables: must be None or an iterable of strings')
            sn = self.pneuron.statesNames()
            for item in qss_vars:
                if item not in sn:
                    raise ValueError(f'Invalid QSS variable: {item} (must be in {sn}')
        if method not in list(self.intMethods().keys()):
            raise ValueError(f'Invalid integration method: "{method}"')

    @Model.logNSpikes
    @Model.checkTitrate
    @Model.addMeta
    @Model.logDesc
    @Model.checkSimParams
    def simulate(self, US_source, pp, fs=1., method='sonic', qss_vars=None):
        ''' Simulate the electro-mechanical model for a specific set of US stimulation parameters,
            and return output data in a dataframe.

            :param US_source: acoustic source object
            :param pp: pulse protocol object
            :param fs: sonophore membrane coverage fraction (-)
            :param method: selected integration method
            :return: output dataframe
        '''
        # Set the tissue elastic modulus
        self.setTissueModulus(US_source)

        # Call appropriate simulation function and return
        simfunc = self.intMethods()[method]
        simargs = [US_source, pp, fs]
        if method == 'sonic':
            simargs.append(qss_vars)
        return simfunc(*simargs)

    def meta(self, US_source, pp, fs, method, qss_vars):
        return {
            'simkey': self.simkey,
            'neuron': self.pneuron.name,
            'a': self.a,
            'd': self.d,
            'US_source': US_source,
            'pp': pp,
            'fs': fs,
            'method': method,
            'qss_vars': qss_vars
        }

    def desc(self, meta):
        s = f'{self}: {meta["method"]} simulation @ {meta["US_source"].desc}, {meta["pp"].desc}'
        if meta['fs'] < 1.0:
            s += f', fs = {(meta["fs"] * 1e2):.2f}%'
        if 'qss_vars' in meta and meta['qss_vars'] is not None:
                s += f" - QSS ({', '.join(meta['qss_vars'])})"
        return s

    @staticmethod
    def getNSpikes(data):
        return PointNeuron.getNSpikes(data)

    @logCache(os.path.join(os.path.split(__file__)[0], 'astim_titrations.log'))
    def titrate(self, Fdrive, pp, fs=1., method='sonic', qss_vars=None, xfunc=None, Arange=None):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given frequency and pulsed protocol.

            :param Fdrive: US frequency (Hz)
            :param pp: pulse protocol object
            :param fs: sonophore membrane coverage fraction (-)
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
            Arange = [0., self.getLookup().refs['A'].max()]

        return threshold(
            lambda x: xfunc(self.simulate(
                AcousticSource(Fdrive, x), pp, fs=fs, method=method, qss_vars=qss_vars)[0]),
            Arange, x0=ASTIM_AMP_INITIAL, eps_thr=ASTIM_ABS_CONV_THR, rel_eps_thr=1e0, precheck=True)

    def getQuasiSteadyStates(self, Fdrive, amps=None, charges=None, DC=1.0, squeeze_output=False):
        ''' Compute the quasi-steady state values of the neuron's gating variables
            for a combination of US amplitudes, charge densities,
            at a specific US frequency and duty cycle.

            :param Fdrive: US frequency (Hz)
            :param amps: US amplitudes (Pa)
            :param charges: membrane charge densities (C/m2)
            :param DC: duty cycle
            :return: 4-tuple with reference values of US amplitude and charge density,
                as well as interpolated Vmeff and QSS gating variables
        '''
        # Get DC-averaged lookups interpolated at the appropriate amplitudes and charges
        lkp = self.getLookup().projectDC(amps=amps, DC=DC).projectN({'a': self.a, 'f': Fdrive})
        if charges is not None:
            lkp = lkp.project('Q', charges)

        # Specify dimensions with A as the first axis
        A_axis = lkp.getAxisIndex('A')
        lkp.move('A', 0)
        nA = lkp.dims()[0]

        # Compute QSS states using these lookups
        QSS = EffectiveVariablesLookup(
            lkp.refs,
            {k: v(lkp) for k, v in self.pneuron.quasiSteadyStates().items()})

        # Compress outputs if needed
        if squeeze_output:
            QSS = QSS.squeeze()
            lkp = lkp.squeeze()

        return lkp, QSS

    def iNetQSS(self, Qm, Fdrive, Adrive, DC):
        ''' Compute quasi-steady state net membrane current for a given combination
            of US parameters and a given membrane charge density.

            :param Qm: membrane charge density (C/m2)
            :param Fdrive: US frequency (Hz)
            :param Adrive: US amplitude (Pa)
            :param DC: duty cycle (-)
            :return: net membrane current (mA/m2)
        '''
        lkp, QSS = self.getQuasiSteadyStates(
            Fdrive, amps=Adrive, charges=Qm, DC=DC, squeeze_output=True)
        return self.pneuron.iNet(lkp['V'], QSS)  # mA/m2


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
        pltvars = self.getPltVars()
        logger.debug(f'A = {Adrive * 1e-3:.2f} kPa, DC = {DC * 1e2:.0f}%')

        # Extract fixed points from QSS charge variation profile
        def dfunc(Qm):
            return - self.iNetQSS(Qm, Fdrive, Adrive, DC)
        fixed_points = getFixedPoints(
            lkp.refs['Q'], dQdt, filter='both', der_func=dfunc).tolist()
        dfunc = lambda x: np.array(self.effDerivatives(_, x, lkp))

        # classified_fixed_points = {'stable': [], 'unstable': [], 'saddle': []}
        classified_fixed_points = []

        np.set_printoptions(precision=2)

        # For each fixed point
        for i, Qm in enumerate(fixed_points):

            # Re-compute QSS at fixed point
            *_, QSS = self.getQuasiSteadyStates(Fdrive, amps=Adrive, charges=Qm, DC=DC,
                                                squeeze_output=True)

            # Classify fixed point stability by numerically evaluating its Jacobian and
            # computing its eigenvalues
            x = np.array([Qm, *QSS.tables.values()])
            eigvals, key = classifyFixedPoint(x, dfunc)
            # classified_fixed_points[key].append(Qm)

            classified_fixed_points.append((x, eigvals, key))
            # eigenvalues.append(eigvals)
            logger.debug(f'{key} point @ Q = {(Qm * 1e5):.1f} nC/cm2')

        # eigenvalues = np.array(eigenvalues).T
        # print(eigenvalues.shape)

        return classified_fixed_points

    def isStableQSS(self, Fdrive, Adrive, DC):
            lookups, QSS = self.getQuasiSteadyStates(
                Fdrive, amps=Adrive, DCs=DC, squeeze_output=True)
            dQdt = -self.pneuron.iNet(lookups['V'], QSS.tables)  # mA/m2
            classified_fixed_points = self.fixedPointsQSS(Fdrive, Adrive, DC, lookups, dQdt)
            return len(classified_fixed_points['stable']) > 0



class DrivenNeuronalBilayerSonophore(NeuronalBilayerSonophore):

    simkey = 'DASTIM'  # keyword used to characterize simulations made with this model

    def __init__(self, Idrive, *args, **kwargs):
        self.Idrive = Idrive
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return super().__repr__()[:-1] + f', Idrive = {self.Idrive:.2f} mA/m2)'

    @classmethod
    def initFromMeta(cls, meta):
        return cls(meta['Idrive'], meta['a'], getPointNeuron(meta['neuron']), embedding_depth=meta['d'])

    def params(self):
        return {**{'Idrive': self.Idrive}, **super().params()}

    @property
    @staticmethod
    def inputs():
        inputvars = NeuronalBilayerSonophore.inputs
        inputvars['Idrive'] = {
            'desc': 'driving current density',
            'label': 'I_{drive}',
            'unit': 'mA/m2',
            'factor': 1e0,
            'precision': 0
        }
        return inputvars

    def filecodes(self, *args):
        codes = super().filecodes(*args)
        codes['Idrive'] = f'Idrive{self.Idrive:.1f}mAm2'
        return codes

    def fullDerivatives(self, *args):
        dydt = super().fullDerivatives(*args)
        dydt[3] += self.Idrive * 1e-3
        return dydt

    def effDerivatives(self, *args):
        dQmdt, *dstates = super().effDerivatives(*args)
        dQmdt += self.Idrive * 1e-3
        return [dQmdt, *dstates]

    def meta(self, US_source, pp, fs, method, qss_vars):
        d = super().meta(US_source, pp, fs, method, qss_vars)
        d['Idrive'] = self.Idrive
        return d