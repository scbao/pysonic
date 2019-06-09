# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2019-05-28 14:45:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-09 21:35:56

import abc
import numpy as np
import nolds
from scipy.integrate import ode, odeint, solve_ivp
from tqdm import tqdm

from ..utils import *
from ..constants import *


class Simulator(metaclass=abc.ABCMeta):
    ''' Generic interface to simulator object. '''

    def initialize(self, y0):
        ''' Initialize global arrays.

            :param y0: vector of initial conditions
            :return: 3-tuple with the initialized time vector, solution matrix and state vector
        '''
        t = np.array([0.])
        y = np.atleast_2d(y0)
        stim = np.array([1])
        return t, y, stim

    def appendSolution(self, t, y, stim, tnew, ynew, is_on):
        ''' Append to time vector, solution matrix and state vector.

            :param t: preceding time vector
            :param y: preceding solution matrix
            :param stim: preceding stimulation state vector
            :param tnew: integration time vector for current interval
            :param ynew: derivative function for current interval
            :param is_on: stimulation state for current interval
            :return: 3-tuple with the appended time vector, solution matrix and state vector
        '''
        t = np.concatenate((t, tnew[1:]))
        y = np.concatenate((y, ynew[1:]), axis=0)
        stim = np.concatenate((stim, np.ones(tnew.size - 1) * is_on))
        return t, y, stim

    def integrate(self, t, y, stim, tnew, dfunc, is_on, use_adaptive_dt=False):
        ''' Integrate system for a time interval and append to preceding solution arrays.

            :param t: preceding time vector
            :param y: preceding solution matrix
            :param stim: preceding stimulation state vector
            :param tnew: integration time vector for current interval
            :param dfunc: derivative function for current interval
            :param is_on: stimulation state for current interval
            :return: 3-tuple with the appended time vector, solution matrix and state vector
        '''
        if tnew.size == 0:
            return t, y, stim
        if use_adaptive_dt:
            ynew = solve_ivp(dfunc, [tnew[0], tnew[-1]], y[-1], t_eval=tnew, method='LSODA').y.T
        else:
            ynew = odeint(dfunc, y[-1], tnew, tfirst=True)
        return self.appendSolution(t, y, stim, tnew, ynew, is_on)

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
            rf, si_format(1 / (dt * rf), 2))
        return t[::rf], y[::rf, :], stim[::rf]

    @property
    @abc.abstractmethod
    def compute(self):
        ''' Abstract compute method. '''
        return 'Should never reach here'

    def __call__(self, *args, **kwargs):
        ''' Call and return compute method, with conditional time monitoring. '''
        monitor_time = kwargs.pop('monitor_time')
        if monitor_time:
            start_time = time.perf_counter()
        output = self.compute(*args, **kwargs)
        if monitor_time:
            end_time = time.perf_counter()
            run_time = end_time - start_time
            output = output, run_time
        return output


class PeriodicSimulator(Simulator):

    def __init__(self, dfunc, stopfunc=None, ivars_to_check=None):
        ''' Initialize simulator with specific derivative and stop functions

            :param dfunc: derivative function
            :param stopfunc: function estimating stopping criterion
            :param ivars_to_check: solution indexes of variables to check for stability
        '''
        self.dfunc = dfunc
        self.ivars_to_check = ivars_to_check
        if stopfunc is not None:
            self.stopfunc = stopfunc
        else:
            self.stopfunc = self.isPeriodicallyStable

    def getNPerCycle(self, dt, T):
        ''' Compute number of samples per cycle given a time step and a specific periodicity.

            :param dt: integration time step (s)
            :param T: periodicity (s)
            :return: number of samples per cycle
        '''
        return int(np.round(T / dt)) + 1

    def getTimeReference(self, dt, T):
        ''' Compute reference integration time vector for a specific periodicity.

            :param dt: integration time step (s)
            :param T: periodicity (s)
            :return: time vector for 1 periodic cycle
        '''
        return np.linspace(0, T, self.getNPerCycle(dt, T))

    def isPeriodicallyStable(self, t, y, T):
        ''' Assess the periodic stabilization of a solution, by evaluating the deviation
            of system variables between the last two periods.

            :param t: time vector
            :param y: solution matrix
            :param T: periodicity (s)
            :return: boolean stating whether the solution is periodically stable or not
        '''
        # Extract the 2 cycles of interest from the solution
        n = self.getNPerCycle(t[1] - t[0], T) - 1
        y_last = y[-n:, :]
        y_prec = y[-2 * n:-n, :]

        # For each variable of interest, evaluate the RMSE between the two cycles, the
        # variation range over the last cycle, and the ratio of these 2 quantities
        ratios = np.array([rmse(y_last[:, ivar], y_prec[:, ivar]) / np.ptp(y_last[:, ivar])
                           for ivar in self.ivars_to_check])

        # Classify the solution as periodically stable only if all RMSE/PTP ratios
        # are below critical threshold
        is_periodically_stable = np.all(ratios < MAX_RMSE_PTP_RATIO)
        # logger.debug(
        #     'step %u: ratios = [%s]', icycle,
        #     ', '.join(['{:.2e}'.format(r) for r in ratios]))
        return is_periodically_stable

    def isAsymptoticallyStable(self, t, y, T):
        ''' Assess the asymptotically stabilization of a solution, by evaluating the deviation
            of system variables from their initial values.

            :param t: time vector
            :param y: solution matrix
            :param T: periodicity (s)
            :return: boolean stating whether the solution is asymptotically stable or not
        '''

        # For each variable of interest, evaluate the ...
        lyapunov_exponents = np.array([nolds.lyap_r(y[:, ivar]) for ivar in self.ivars_to_check])
        print(lyapunov_exponents)
        is_asyptotically_stable = np.all(lyapunov_exponents < 1e-5)
        return is_asyptotically_stable

    def compute(self, y0, dt, T, t0=0., nmax=NCYCLES_MAX):
        ''' Simulate system with a specific periodicity until stopping criterion is met.

            :param y0: 1D vector of initial conditions
            :param dt: integration time step (s)
            :param T: periodicity (s)
            :param t0: starting time
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        '''

        # If none specified, set all variables to be checked for stability
        if self.ivars_to_check is None:
            self.ivars_to_check = range(y0.size)

        # Get reference time vector
        tref = self.getTimeReference(dt, T)

        # Initialize global arrays
        t, y, stim = self.initialize(y0)

        # Integrate system for a few cycles until stopping criterion is met
        icycle = 0
        stop = False
        while not stop and icycle < nmax:
            t, y, stim = self.integrate(t, y, stim, tref + icycle * T, self.dfunc, True)
            if icycle > 0:
                stop = self.stopfunc(t, y, T)
            icycle += 1
        t += t0

        # Log stopping criterion
        t_str = 't = {:.5f} ms'.format(t[-1] * 1e3)
        if icycle == nmax:
            logger.warning('%s: criterion not met -> stopping after %u cycles', t_str, icycle)
        else:
            logger.debug('%s: stopping criterion met after %u cycles', t_str, icycle)

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

            :param tstim: duration of US stimulation (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param print_progress: boolean specifying whether to show a progress bar
            :return: adjusted PRF value (Hz)
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

    def compute(self, y0, dt, tstim, toffset, PRF, DC, target_dt=None, print_progress=False):
        ''' Simulate system for a specific stimulus application pattern.

            :param y0: 1D vector of initial conditions
            :param dt: integration time step (s)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param target_dt: target time step after resampling
            :param print_progress: boolean specifying whether to show a progress bar
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        '''

        # Adjust PRF and get number of pulses
        PRF = self.adjustPRF(tstim, PRF, DC, print_progress)
        npulses = self.getNPulses(tstim, PRF)

        # Get reference time vectors
        t_on, t_off, t_offset = self.getTimeReference(dt, tstim, toffset, PRF, DC)

        # Initialize global arrays
        t, y, stim = self.initialize(y0)

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

    def __init__(self, dfunc_on, dfunc_off, dfunc_sparse, predfunc, is_dense_var,
                 stopfunc=None, ivars_to_check=None):
        ''' Initialize simulator with specific derivative functions

            :param dfunc_on: derivative function for ON periods
            :param dfunc_off: derivative function for OFF periods
            :param dfunc_sparse: derivative function for sparse integration
            :param predfunc: function computing the extra arguments necessary for sparse integration
            :param is_dense_var: boolean array stating for each variable if it evolves fast or not
            :param stopfunc: function estimating stopping criterion
            :param ivars_to_check: solution indexes of variables to check for stability
        '''
        PWSimulator.__init__(self, dfunc_on, dfunc_off)
        self.sparse_solver = ode(dfunc_sparse)
        self.sparse_solver.set_integrator('dop853', nsteps=SOLVER_NSTEPS, atol=1e-12)
        self.predfunc = predfunc
        self.is_dense_var = is_dense_var
        self.is_sparse_var = np.invert(is_dense_var)
        self.stopfunc = stopfunc
        self.ivars_to_check = ivars_to_check

    def integrate(self, t, y, stim, tnew, dfunc, is_on):
        ''' Integrate system for a time interval and append to preceding solution arrays,
            using a hybrid scheme:

            - First, the full ODE system is integrated for a few cycles with a dense time
              granularity until a stopping criterion is met
            - Second, the profiles of all variables over the last cycle are resampled to a
              far lower (i.e. sparse) sampling rate
            - Third, a subset of the ODE system is integrated with a sparse time granularity,
              for the remaining of the time interval, while the remaining variables are
              periodically expanded from their last cycle profile.

            :param t: preceding time vector
            :param y: preceding solution matrix
            :param stim: preceding stimulation state vector
            :param tnew: integration time vector for current interval
            :param dfunc: derivative function for current interval
            :param is_on: stimulation state for current interval
            :return: 3-tuple with the appended time vector, solution matrix and state vector
        '''

        if tnew.size == 0:
            return t, y, stim

        # Initialize periodic solver
        dense_solver = PeriodicSimulator(
            dfunc, stopfunc=self.stopfunc, ivars_to_check=self.ivars_to_check)
        dt_dense = tnew[1] - tnew[0]
        npc_dense = dense_solver.getNPerCycle(dt_dense, self.T)

        # Until final integration time is reached
        while t[-1] < tnew[-1]:
            logger.debug('t = {:.5f} ms: starting new hybrid integration'.format(t[-1] * 1e3))

            # Integrate dense system until stopping criterion is met
            tdense, ydense, stimdense = dense_solver.compute(y[-1], dt_dense, self.T, t0=t[-1])
            t, y, stim = self.appendSolution(t, y, stim, tdense, ydense, is_on)

            # Resample signals over last cycle to match sparse time step
            tlast, ylast, stimlast = self.resample(
                tdense[-npc_dense:], ydense[-npc_dense:], stimdense[-npc_dense:],
                self.dt_sparse)
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
                self.sparse_solver.set_f_params(self.predfunc(ylast[j % npc_sparse]))
                self.sparse_solver.integrate(tsparse[j])
                if not self.sparse_solver.successful():
                    raise ValueError(
                        'integration error at t = {:.5f} ms'.format(tsparse[j] * 1e3))
                ysparse[j, self.is_dense_var] = ylast[j % npc_sparse, self.is_dense_var]
                ysparse[j, self.is_sparse_var] = self.sparse_solver.y
            t, y, stim = self.appendSolution(t, y, stim, tsparse, ysparse, is_on)

        return t, y, stim

    def compute(self, y0, dt_dense, dt_sparse, T, tstim, toffset, PRF, DC, print_progress=False):
        ''' Simulate system for a specific stimulus application pattern.

            :param y0: 1D vector of initial conditions
            :param dt_dense: dense integration time step (s)
            :param dt_sparse: sparse integration time step (s)
            :param T: periodicity (s)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :param print_progress: boolean specifying whether to show a progress bar
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        '''

        # Set periodicity and sparse time step
        self.T = T
        self.dt_sparse = dt_sparse

        # Call and return parent compute method
        return PWSimulator.compute(
            self, y0, dt_dense, tstim, toffset, PRF, DC,
            target_dt=None, print_progress=print_progress)
