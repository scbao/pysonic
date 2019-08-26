# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2016-09-29 16:16:19
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-26 14:54:58

from copy import deepcopy
import logging
import numpy as np
import numdifftools as nd
import pandas as pd
from scipy import linalg

from .simulators import PWSimulator, HybridSimulator, PeriodicSimulator
from .bls import BilayerSonophore
from .pneuron import PointNeuron
from .model import Model
from .batches import Batch
from ..utils import *
from ..constants import *
from ..postpro import getFixedPoints
from .lookups import SmartLookup, SmartDict


NEURONS_LOOKUP_DIR = os.path.abspath(os.path.split(__file__)[0] + "/../neurons/")


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
        BilayerSonophore.__init__(self, a, pneuron.Cm0, pneuron.Qm0(), embedding_depth)

    def __repr__(self):
        s = '{}({:.1f} nm, {}'.format(self.__class__.__name__, self.a * 1e9, self.pneuron)
        if self.d > 0.:
            s += ', d={}m'.format(si_format(self.d, precision=1, space=' '))
        return s + ')'

    def params(self):
        return {**super().params(), **self.pneuron.params()}

    def getPltVars(self, wrapleft='df["', wrapright='"]'):
        return {**super().getPltVars(wrapleft, wrapright),
                **self.pneuron.getPltVars(wrapleft, wrapright)}

    def getPltScheme(self):
        return self.pneuron.getPltScheme()

    def filecode(self, *args):
        return Model.filecode(self, *args)

    @staticmethod
    def inputs():
        # Get parent input vars and supress irrelevant entries
        bls_vars = BilayerSonophore.inputs()
        pneuron_vars = PointNeuron.inputs()
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

    def filecodes(self, Fdrive, Adrive, tstim, toffset, PRF, DC, fs, method):
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
        codes['fs'] = 'fs{:.0f}%'.format(fs * 1e2) if fs < 1 else None
        codes['method'] = method
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
        x[stim == 0] = lkp['OFF'].interpVar(Qm[stim == 0], 'Q', key)
        x[stim == 1] = lkp['ON'].interpVar(Qm[stim == 1], 'Q', key)
        return x

    @staticmethod
    def spatialAverage(fs, x, x0):
        ''' fs-modulated spatial averaging. '''
        return fs * x + (1 - x) * x0

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
        data, meta = BilayerSonophore.simulate(self, Fdrive, Adrive, Qm)
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
        log = '{}: lookups @ {}Hz, {}Pa, {:.2f} nC/cm2'.format(
            self, *si_format([Fdrive, Adrive], precision=1, space=' '), Qm * 1e5)
        if len(fs) > 1:
            log += ', fs = {:.0f} - {:.0f}%'.format(fs.min() * 1e2, fs.max() * 1e2)
        log += ', tcomp = {:.3f} s'.format(meta['tcomp'])
        logger.info(log)

        # Return effective coefficients
        return [meta['tcomp'], effvars]

    def getLookupFileName(self, a=None, Fdrive=None, Adrive=None, fs=False):
        fname = '{}_lookups'.format(self.pneuron.name)
        if a is not None:
            fname += '_{:.0f}nm'.format(a * 1e9)
        if Fdrive is not None:
            fname += '_{:.0f}kHz'.format(Fdrive * 1e-3)
        if Adrive is not None:
            fname += '_{:.0f}kPa'.format(Adrive * 1e-3)
        if fs is True:
            fname += '_fs'
        return '{}.pkl'.format(fname)

    def getLookupFilePath(self, *args, **kwargs):
        return os.path.join(NEURONS_LOOKUP_DIR, self.getLookupFileName(*args, **kwargs))

    def getLookup(self, *args, **kwargs):
        lookup_path = self.getLookupFilePath(*args, **kwargs)
        if not os.path.isfile(lookup_path):
            raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))
        with open(lookup_path, 'rb') as fh:
            frame = pickle.load(fh)
        if 'ng' in frame['lookup']:
            del frame['lookup']['ng']
        refs = frame['input']

        # Move fs to last reference dimension
        keys = list(refs.keys())
        if 'fs' in keys and keys.index('fs') < len(keys) - 1:
            del keys[keys.index('fs')]
            keys.append('fs')
            refs = {k: refs[k] for k in keys}

        return SmartLookup(refs, frame['lookup'])

    def getLookup2D(self, Fdrive, fs):
        if fs < 1:
            lkp2d = self.getLookup(a=self.a, Fdrive=Fdrive, fs=True).project('fs', fs)
        else:
            lkp2d = self.getLookup().projectN({'a': self.a, 'f': Fdrive})
        return lkp2d

    def fullDerivatives(self, t, y, Fdrive, Adrive, phi, fs):
        ''' Compute the full system derivatives.

            :param t: specific instant in time (s)
            :param y: vector of state variables
            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param phi: acoustic drive phase (rad)
            :param fs: sonophore membrane coevrage fraction (-)
            :return: vector of derivatives
        '''
        dydt_mech = BilayerSonophore.derivatives(
            self, t, y[:3], Fdrive, Adrive, y[3], phi)
        dydt_elec = self.pneuron.derivatives(
            t, y[3:], Cm=self.spatialAverage(fs, self.capacitance(y[1]), self.Cm0))
        return dydt_mech + dydt_elec

    def effDerivatives(self, t, y, lkp1d):
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
        states_dict = dict(zip(self.pneuron.statesNames(), states))
        lkp0d = lkp1d.interpolate1D('Q', Qm)
        dQmdt = - self.pneuron.iNet(lkp0d['V'], states_dict) * 1e-3
        return [dQmdt, *self.pneuron.getDerEffStates(lkp0d, states_dict)]

    def __simFull(self, Fdrive, Adrive, tstim, toffset, PRF, DC, fs, phi=np.pi):
        # Determine time step
        dt = 1 / (NPC_DENSE * Fdrive)

        # Compute initial non-zero deflection
        Z = self.computeInitialDeflection(Adrive, Fdrive, phi, self.Qm0, dt)

        # Set initial conditions
        ss0 = self.pneuron.getSteadyStates(self.pneuron.Vm0)
        y0 = np.concatenate(([0., 0., self.ng0, self.Qm0], ss0))
        y1 = np.concatenate(([0., Z, self.ng0, self.Qm0], ss0))

        # Initialize simulator and compute solution
        logger.debug('Computing detailed solution')
        simulator = PWSimulator(
            lambda t, y: self.fullDerivatives(t, y, Fdrive, Adrive, phi, fs),
            lambda t, y: self.fullDerivatives(t, y, 0., 0., 0., fs))
        t, y, stim = simulator(
            y1, dt, tstim, toffset, PRF, DC,
            print_progress=logger.getEffectiveLevel() <= logging.INFO,
            target_dt=CLASSIC_TARGET_DT)

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

    def __simHybrid(self, Fdrive, Adrive, tstim, toffset, PRF, DC, fs, phi=np.pi):
        # Determine time steps
        dt_dense, dt_sparse = [1. / (n * Fdrive) for n in [NPC_DENSE, NPC_SPARSE]]

        # Compute initial non-zero deflection
        Z = self.computeInitialDeflection(Adrive, Fdrive, phi, self.Qm0, dt_dense)

        # Set initial conditions
        ss0 = self.pneuron.getSteadyStates(self.pneuron.Vm0)
        y0 = np.concatenate(([0., 0., self.ng0, self.Qm0], ss0))
        y1 = np.concatenate(([0., Z, self.ng0, self.Qm0], ss0))

        # Initialize simulator and compute solution
        is_dense_var = np.array([True] * 3 + [False] * (len(self.pneuron.states) + 1))
        logger.debug('Computing hybrid solution')
        simulator = HybridSimulator(
            lambda t, y: self.fullDerivatives(t, y, Fdrive, Adrive, phi, fs),
            lambda t, y: self.fullDerivatives(t, y, 0., 0., 0., fs),
            lambda t, y, Cm: self.pneuron.derivatives(
                t, y, Cm=self.spatialAverage(fs, Cm, self.Cm0)),
            lambda yref: self.capacitance(yref[1]),
            is_dense_var,
            ivars_to_check=[1, 2])
        t, y, stim = simulator(y1, dt_dense, dt_sparse, 1. / Fdrive, tstim, toffset, PRF, DC)

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

    def __simSonic(self, Fdrive, Adrive, tstim, toffset, PRF, DC, fs):
        # Load appropriate 2D lookups
        lkp2d = self.getLookup2D(Fdrive, fs)

        # Interpolate 2D lookups at zero and US amplitude
        logger.debug('Interpolating lookups at A = %.2f kPa and A = 0', Adrive * 1e-3)
        lkps1d = {'ON': lkp2d.project('A', Adrive), 'OFF': lkp2d.project('A', 0.)}

        # Set initial conditions
        y0 = np.concatenate(([self.Qm0], self.pneuron.getSteadyStates(self.pneuron.Vm0)))

        # Initialize simulator and compute solution
        logger.debug('Computing effective solution')
        simulator = PWSimulator(
            lambda t, y: self.effDerivatives(t, y, lkps1d['ON']),
            lambda t, y: self.effDerivatives(t, y, lkps1d['OFF']))
        t, y, stim = simulator(y0, self.pneuron.chooseTimeStep(), tstim, toffset, PRF, DC)

        # Prepend initial conditions (prior to stimulation)
        t, y, stim = simulator.prependSolution(t, y, stim)

        # Store output in dataframe and return
        data = pd.DataFrame({
            't': t,
            'stimstate': stim,
            'Qm': y[:, 0]
        })
        data['Vm'] = self.interpOnOffVariable('V', data['Qm'].values, stim, lkps1d)
        for key in ['Z', 'ng']:
            data[key] = np.full(t.size, np.nan)
        for i in range(len(self.pneuron.states)):
            data[self.pneuron.statesNames()[i]] = y[:, i + 1]
        return data

    @classmethod
    def simQueue(cls, freqs, amps, durations, offsets, PRFs, DCs, fs, methods, outputdir=None):
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
            :return: list of parameters (list) for each simulation
        '''
        method_ids = list(range(len(methods)))
        if ('full' in methods or 'hybrid' in methods) and outputdir is None:
            logger.warning('Running cumbersome simulation(s) without file saving')
        if amps is None:
            amps = [np.nan]
        DCs = np.array(DCs)
        queue = []
        if 1.0 in DCs:
            queue += Batch.createQueue(
                freqs, amps, durations, offsets, min(PRFs), 1.0, fs, method_ids)
        if np.any(DCs != 1.0):
            queue += Batch.createQueue(
                freqs, amps, durations, offsets, PRFs, DCs[DCs != 1.0], fs, method_ids)
        for item in queue:
            if np.isnan(item[1]):
                item[1] = None
            item[-1] = methods[int(item[-1])]
        return cls.checkOutputDir(queue, outputdir)

    @Model.logNSpikes
    @Model.checkTitrate('Adrive')
    @Model.addMeta
    def simulate(self, Fdrive, Adrive, tstim, toffset, PRF=100., DC=1., fs=1., method='sonic'):
        ''' Simulate the electro-mechanical model for a specific set of US stimulation parameters,
            and return output data in a dataframe.

            :param Fdrive: acoustic drive frequency (Hz)
            :param Adrive: acoustic drive amplitude (Pa)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param fs: sonophore membrane coverage fraction (-)
            :param method: selected integration method
            :return: 2-tuple with the output dataframe and computation time.
        '''
        logger.info(
            '%s: simulation @ f = %sHz, A = %sPa, t = %ss (%ss offset)%s%s',
            self, si_format(Fdrive, 0, space=' '), si_format(Adrive, 2, space=' '),
            *si_format([tstim, toffset], 1, space=' '),
            (', PRF = {}Hz, DC = {:.2f}%'.format(
                si_format(PRF, 2, space=' '), DC * 1e2) if DC < 1.0 else ''),
            ', fs = {:.2f}%'.format(fs * 1e2) if fs < 1.0 else '')

        # Check validity of stimulation parameters
        BilayerSonophore.checkInputs(Fdrive, Adrive, 0.0, 0.0)
        PointNeuron.checkInputs(Adrive, tstim, toffset, PRF, DC)

        # Call appropriate simulation function and return
        try:
            simfunc = {
                'full': self.__simFull,
                'hybrid': self.__simHybrid,
                'sonic': self.__simSonic
            }[method]
        except KeyError:
            raise ValueError('Invalid integration method: "{}"'.format(method))
        return simfunc(Fdrive, Adrive, tstim, toffset, PRF, DC, fs)

    def meta(self, Fdrive, Adrive, tstim, toffset, PRF, DC, fs, method):
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
            'fs': fs,
            'method': method
        }

    @staticmethod
    def getNSpikes(data):
        return PointNeuron.getNSpikes(data)

    @logCache(os.path.join(os.path.split(__file__)[0], 'astim_titrations.log'))
    def titrate(self, Fdrive, tstim, toffset, PRF=100., DC=1., fs=1., method='sonic',
                xfunc=None, Arange=None):
        ''' Use a binary search to determine the threshold amplitude needed to obtain
            neural excitation for a given frequency, duration, PRF and duty cycle.

            :param Fdrive: US frequency (Hz)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
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

        return binarySearch(
            lambda x: xfunc(self.simulate(*x)[0]),
            [Fdrive, tstim, toffset, PRF, DC, fs, method], 1, Arange, THRESHOLD_CONV_RANGE_ASTIM
        )

    def getQuasiSteadyStates(self, Fdrive, amps=None, charges=None, DCs=1.0, squeeze_output=False):
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
        lkp = self.getLookup().projectDCs(amps=amps, DCs=DCs).projectN({'a': self.a, 'f': Fdrive})
        if charges is not None:
            lkp = lkp.project('Q', charges)

        # Specify dimensions with A and DC as the first two axes
        A_axis = lkp.getAxisIndex('A')
        lkp.move('A', 0)
        lkp.move('DC', 1)
        nA, nDC = lkp.dims()[:2]

        # Compute QSS states using these lookups
        QSS = {k: np.empty(lkp.dims()) for k in self.pneuron.statesNames()}
        for iA in range(nA):
            for iDC in range(nDC):
                lkp1d = SmartDict({k: v[iA, iDC] for k, v in lkp.items()})
                QSS_1D = {k: v(lkp1d) for k, v in self.pneuron.quasiSteadyStates().items()}
                for k in QSS.keys():
                    QSS[k][iA, iDC] = QSS_1D[k]
        QSS = SmartLookup(lkp.refs, QSS)

        for item in [lkp, QSS]:
            item.move('A', A_axis)
            item.move('DC', -1)

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
            Fdrive, amps=Adrive, charges=Qm, DCs=DC, squeeze_output=True)
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
        logger.debug('A = {:.2f} kPa, DC = {:.0f}%'.format(Adrive * 1e-3, DC * 1e2))

        # Extract fixed points from QSS charge variation profile
        def dfunc(Qm):
            return - self.iNetQSS(Qm, Fdrive, Adrive, DC)
        fixed_points = getFixedPoints(
            lkp.refs['Q'], dQdt, filter='both', der_func=dfunc).tolist()
        dfunc = lambda x: np.array(self.effDerivatives(_, x, lkp))
        classified_fixed_points = {'stable': [], 'unstable': [], 'saddle': []}

        np.set_printoptions(precision=2)

        # For each fixed point
        for i, Qm in enumerate(fixed_points):

            # Re-compute QSS at fixed point
            *_, QSS = self.getQuasiSteadyStates(Fdrive, amps=Adrive, charges=Qm, DCs=DC,
                                                squeeze_output=True)

            # Approximate the system's Jacobian matrix at the fixed-point and compute its eigenvalues
            x = np.array([Qm, *QSS.tables.values()])
            print(f'x = {x}, dfunx(x) = {dfunc(x)}')
            eps_machine = np.sqrt(np.finfo(float).eps)
            J = jacobian(dfunc, x, rel_eps=eps_machine, method='forward')
            # Jfunc = nd.Jacobian(dfunc, order=3)
            # J = Jfunc(x)
            # print('------------------ Jacobian ------------------')
            # names = ['Q'] + self.pneuron.statesNames()
            # for name, Jline in zip(names, J):
            #     print(f'd(d{name}dt) = {Jline}')

            # Determine fixed point stability based on eigenvalues
            eigvals, eigvecs = linalg.eig(J)
            s = ['({0.real:.2e} + {0.imag:.2e}j)'.format(x) for x in eigvals]
            print(f'eigenvalues = {s}')
            is_real_eigvals = np.isreal(eigvals)
            print(is_real_eigvals)
            is_neg_eigvals = eigvals.real < 0
            if np.all(is_neg_eigvals):
                key = 'stable'
            elif np.any(is_neg_eigvals):
                key = 'saddle'
            else:
                key = 'unstable'
            classified_fixed_points[key].append(Qm)
            logger.debug(f'{key} fixed point @ Q = {(Qm * 1e5):.1f} nC/cm2')

        return classified_fixed_points

    def isStableQSS(self, Fdrive, Adrive, DC):
            lookups, QSS = self.getQuasiSteadyStates(
                Fdrive, amps=Adrive, DCs=DC, squeeze_output=True)
            dQdt = -self.pneuron.iNet(lookups['V'], QSS.tables)  # mA/m2
            classified_fixed_points = self.fixedPointsQSS(Fdrive, Adrive, DC, lookups, dQdt)
            return len(classified_fixed_points['stable']) > 0

    def titrateQSS(self, Fdrive, DC=1., Arange=None):

        # Default amplitude interval
        if Arange is None:
            Arange = [0., self.getLookup().refs['A'].max()]

        # Titration function
        def xfunc(x):
            if self.pneuron.name == 'STN':
                return self.isStableQSS(*x)
            else:
                return not self.isStableQSS(*x)

        return binarySearch(
            xfunc,
            [Fdrive, DC], 1, Arange, THRESHOLD_CONV_RANGE_ASTIM)
