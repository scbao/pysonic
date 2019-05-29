# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2019-05-28 14:45:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-29 18:35:10

import abc
import numpy as np
from scipy.integrate import ode, odeint
from tqdm import tqdm

from ..utils import *
from ..constants import *


class Simulator(metaclass=abc.ABCMeta):
    ''' Generic interface to simulator object. '''

    def __init__(self):
        pass

    def resample(self, t, y, stim, target_dt):
        ''' Resample a solution to a new target time step.

            :param t: time vector
            :param y: solution matrix
            :param stim: stimulation state vector
            :target_dt: target time step after resampling
            :return: 3-tuple with the resampled time vector, solution matrix and state vector
        '''
        dt = t[1] - t[0]
        rf = int(np.round(target_dt / dt))
        assert rf >= 1, 'Hyper-sampling not supported'
        logger.debug(
            'Downsampling output arrays by factor %u (Fs = %sHz)',
            rf, si_format(1 / (dt * rf), 2)
        )
        t = t[::rf]
        y = y[::rf, :]
        stim = stim[::rf]
        return t, y, stim

    def initialize(self, y0, t0=0.):
        ''' Initialize global arrays.

            :param y0: vector of initial conditions
            :param t0: starting time
            :return: 3-tuple with the initialized time vector, solution matrix and state vector
        '''
        t = np.array([t0])
        y = np.atleast_2d(y0)
        stim = np.array([1])
        return t, y, stim

    def integrate(self, t, y, stim, tnew, func, is_on):
        ''' Integrate system for a time interval and append to preceding solution arrays.

            :param t: preceding time vector
            :param y: preceding solution matrix
            :param stim: preceding stimulation state vector
            :param tnew: integration time vector for current interval
            :param func: derivative function for current interval
            :param is_on: stimulation state for current interval
            :return: 3-tuple with the appended time vector, solution matrix and state vector
        '''
        if tnew.size > 0:
            ynew = odeint(func, y[-1], tnew)
            t = np.concatenate((t, tnew[1:]))
            y = np.concatenate((y, ynew[1:]), axis=0)
            stim = np.concatenate((stim, np.ones(tnew.size - 1) * is_on))
        return t, y, stim

    @property
    @abc.abstractmethod
    def compute(self):
        return 'Should never reach here'



class PeriodicSimulator(Simulator):

    def __init__(self, dfunc, ivars_to_check=None):
        ''' Initialize simulator with specific derivative function

            :param dfunc: derivative function
            :param ivars_to_check: solution indexes of variables to check for stability
        '''
        self.dfunc = dfunc
        self.ivars_to_check = ivars_to_check

    def getNPerCycle(self, dt, f):
        ''' Compute number of samples per cycle given a time step and a specific periodicity.

            :param dt: integration time step (s)
            :param f: periodic frequency (Hz)
            :return: number of samples per cycle
        '''
        return int(np.round(1 / (f * dt))) + 1

    def getTimeReference(self, dt, f):
        ''' Compute reference integration time vector for a specific periodicity.

            :param dt: integration time step (s)
            :param f: periodic frequency (Hz)
            :return: time vector for 1 periodic cycle
        '''
        return np.linspace(0, 1 / f, self.getNPerCycle(dt, f))

    def isPeriodicStable(self, y, npc, icycle):
        ''' Assess the periodic stabilization of a solution.

            :param y: solution matrix
            :param npc: number of samples per cycle
            :param icycle: index of cycle of interest
            :return: boolean stating whether the solution is stable or not
        '''
        y_target = y[icycle * npc: (icycle + 1) * npc, :]
        y_prec = y[(icycle - 1) * npc: icycle * npc, :]
        x_ratios = []
        for ivar in self.ivars_to_check:
            x_target, x_prec = y_target[:, ivar], y_prec[:, ivar]
            x_ptp = np.ptp(x_target)
            x_rmse = rmse(x_target, x_prec)
            x_ratios.append(x_rmse / x_ptp)

        is_periodically_stable = np.all(np.array(x_ratios) < MAX_RMSE_PTP_RATIO)

        logger.debug(
            'step %u: ratios = [%s] -> %sstable',
            icycle,
            ', '.join(['{:.2e}'.format(r) for r in x_ratios]),
            '' if is_periodically_stable else 'un'
        )

        return is_periodically_stable

    def compute(self, y0, dt, f, t0=0.):
        ''' Simulate system with a specific periodicity until stabilization.

            :param y0: 1D vector of initial conditions
            :param dt: integration time step (s)
            :param f: periodic frequency (Hz)
            :param t0: starting time
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        '''

        # Adjust variables to check if needed
        if self.ivars_to_check is None:
            self.ivars_to_check = range(y0.size)

        # Get reference time vector
        tref = self.getTimeReference(dt, f)

        # Initialize global arrays
        t, y, stim = self.initialize(y0, t0=t0)

        # Integrate system for a few cycles until stabilization
        icycle = 0
        conv = False
        while not conv and icycle < NCYCLES_MAX:
            t, y, stim = self.integrate(t, y, stim, tref + icycle / f, self.dfunc, True)
            if icycle > 0:
                conv = self.isPeriodicStable(y, tref.size - 1, icycle)
            icycle += 1

        # Log stopping criterion
        if icycle == NCYCLES_MAX:
            logger.warning('No convergence: stopping after %u cycles', icycle)
        else:
            logger.debug('Periodic convergence after %u cycles', icycle)

        # Return output variables
        return t, y, stim


class PWSimulator(Simulator):

    def __init__(self, dfunc_on, dfunc_off):
        ''' Initialize simulator with specific derivative functions

            :param dfunc_on: derivative function for ON periods
            :param dfunc_off: derivative function for OFF periods
        '''
        self.dfunc_on = dfunc_on
        self.dfunc_off = dfunc_off

    def getTimeReference(self, dt, tstim, toffset, PRF, DC):
        ''' Compute reference integration time vectors for a specific stimulus application pattern.

            :param dt: integration time step (s)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :return: 3-tuple with time vectors for stimulus ON and OFF periods and stimulus offset
        '''

        # Compute vector sizes
        T_ON = DC / PRF
        T_OFF = (1 - DC) / PRF

        # For high-PRF pulsed protocols: adapt time step to ensure minimal
        # number of samples during TON or TOFF
        dt_warning_msg = 'high-PRF protocol: lowering time step to %.2e s to properly integrate %s'
        for key, T in {'TON': T_ON, 'TOFF': T_OFF}.items():
            if T > 0 and T / dt < MIN_SAMPLES_PER_PULSE_INT:
                dt = T / MIN_SAMPLES_PER_PULSE_INT
                logger.warning(dt_warning_msg, dt, key)

        # Initializing accurate time vectors pulse ON and OFF periods, as well as offset
        t_on = np.linspace(0, T_ON, int(np.round(T_ON / dt)) + 1)
        t_off = np.linspace(T_ON, 1 / PRF, int(np.round(T_OFF / dt)))
        t_offset = np.linspace(tstim, tstim + toffset, int(np.round(toffset / dt)))

        return t_on, t_off, t_offset

    def adjustPRF(self, tstim, PRF, DC, print_progress):
        ''' Adjust the PRF in case of continuous wave stimulus, in order to obtain the desired
            number of integration interval(s) during stimulus.
        '''
        if DC < 1.0:  # if PW stimuli, then no change
            return PRF
        else:  # if CW stimuli, then divide integration according to presence of progress bar
            return {True: 100., False: 1.}[print_progress] / tstim

    def getNPulses(self, tstim, PRF):
        ''' Calculate number of pulses from stimulus temporal pattern.

            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :return: number of pulses during the stimulus
        '''
        return int(np.round(tstim * PRF))

    def compute(self, y0, dt, tstim, toffset, PRF, DC, t0=0, target_dt=None, print_progress=False):
        ''' Simulate system for a specific stimulus application pattern.

            :param y0: 1D vector of initial conditions
            :param dt: integration time step (s)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param t0: starting time
            :target_dt: target time step after resampling
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        '''

        # Adjust PRF and get number of pulses
        PRF = self.adjustPRF(tstim, PRF, DC, print_progress)
        npulses = self.getNPulses(tstim, PRF)

        # Get reference time vectors
        t_on, t_off, t_offset = self.getTimeReference(dt, tstim, toffset, PRF, DC)

        # Initialize global arrays
        t, y, stim = self.initialize(y0, t0=t0)

        # Initialize progress bar
        if print_progress:
            setHandler(logger, TqdmHandler(my_log_formatter))
            ntot = int(npulses * (tstim + toffset) / tstim)
            pbar = tqdm(total=ntot)

        # Integrate ON and OFF intervals of each pulse
        for i in range(npulses):
            for j, (tref, func) in enumerate(zip([t_on, t_off], [self.dfunc_on, self.dfunc_off])):
                t, y, stim = self.integrate(t, y, stim, tref + i / PRF, func, j == 0)

            # Update progress bar
            if print_progress:
                pbar.update(i)

        # Integrate offset interval
        t, y, stim = self.integrate(t, y, stim, t_offset, self.dfunc_off, False)

        # Terminate progress bar
        if print_progress:
            pbar.update(npulses)
            pbar.close()

        # Resample solution if specified
        if target_dt is not None:
            t, y, stim = self.resample(t, y, stim, target_dt)

        # Return output variables
        return t, y, stim


class HybridSimulator(PWSimulator):

    def __init__(self, dfunc_on, dfunc_off, dfunc_sparse, predfunc,
                 is_dense_var=None, ivars_to_check=None):
        ''' Initialize simulator with specific derivative functions

            :param dfunc_on: derivative function for ON periods
            :param dfunc_off: derivative function for OFF periods
        '''
        PWSimulator.__init__(self, dfunc_on, dfunc_off)
        self.sparse_solver = ode(dfunc_sparse)
        self.sparse_solver.set_integrator('dop853', nsteps=SOLVER_NSTEPS, atol=1e-12)
        self.predfunc = predfunc
        self.is_dense_var = is_dense_var
        self.is_sparse_var = np.invert(is_dense_var)
        self.ivars_to_check = ivars_to_check

    def integrate(self, t, y, stim, tnew, func, is_on):

        if tnew.size > 0:

            dt_dense = tnew[1] - tnew[0]

            # Initialize periodic solver
            dense_solver = PeriodicSimulator(func, self.ivars_to_check)
            npc_dense = dense_solver.getNPerCycle(dt_dense, self.f)

            # Until final integration time is reached
            while t[-1] < tnew[-1]:
                logger.debug('t = {:.5f} ms: starting new hybrid integration'.format(t[-1] * 1e3))

                # Integrate dense system until convergence
                tdense, ydense, stimdense = dense_solver.compute(y[-1], dt_dense, self.f)
                tdense += t[-1]
                t = np.concatenate((t, tdense[1:]))
                y = np.concatenate((y, ydense[1:]), axis=0)
                stim = np.concatenate((stim, np.ones(tdense.size - 1) * is_on))

                # Resample signals over last acoustic cycle to match sparse time step
                tlast, ylast, stimlast = self.resample(
                    tdense[-npc_dense:], ydense[-npc_dense:], stimdense[-npc_dense:],
                    self.dt_sparse
                )
                npc_sparse = tlast.size

                # Integrate until either the rest of the interval or max update interval is reached
                t0 = tdense[-1]
                tf = min(tnew[-1], tdense[0] + DT_UPDATE)
                nsparse = int(np.round((tf - t0) / self.dt_sparse))
                tsparse = np.linspace(t0, tf, nsparse)
                ysparse = np.empty((nsparse, y.shape[1]))
                ysparse[0] = y[-1]
                self.sparse_solver.set_initial_value(y[-1, self.is_sparse_var], t[-1])
                for j in range(1, tsparse.size):
                    self.sparse_solver.set_f_params(
                        self.predfunc(ylast[j % npc_sparse]))
                    self.sparse_solver.integrate(tsparse[j])
                    if not self.sparse_solver.successful():
                        raise ValueError(
                            'integration error at t = {:.5f} ms'.format(tsparse[j] * 1e3))
                    ysparse[j, self.is_dense_var] = ylast[j % npc_sparse, self.is_dense_var]
                    ysparse[j, self.is_sparse_var] = self.sparse_solver.y
                t = np.concatenate((t, tsparse[1:]))
                y = np.concatenate((y, ysparse[1:]), axis=0)
                stim = np.concatenate((stim, np.ones(tsparse.size - 1) * is_on))

        return t, y, stim

    def compute(self, y0, dt_dense, dt_sparse, f, tstim, toffset, PRF, DC, print_progress=False):

        # Set periodicity and sparse time step
        self.f = f
        self.dt_sparse = dt_sparse

        # Adjust dense variables
        if self.is_dense_var is None:
            self.is_dense_var = np.array([True] * y0.size)

        return PWSimulator.compute(
            self, y0, dt_dense, tstim, toffset, PRF, DC,
            target_dt=None, print_progress=print_progress)
