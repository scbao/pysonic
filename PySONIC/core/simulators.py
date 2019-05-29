# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2019-05-28 14:45:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-29 09:47:07

import numpy as np
from scipy.integrate import odeint
import progressbar as pb

from ..utils import *
from ..constants import MIN_SAMPLES_PER_PULSE_INT, MAX_RMSE_PTP_RATIO, NCYCLES_MAX


class Simulator:
    ''' Generic interface to simulator object. '''

    def __init__(self):
        pass

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

    def compute(self):
        pass


class PeriodicSimulator(Simulator):

    def __init__(self, derf, ivars_to_check=None):
        ''' Initialize simulator with specific derivative function

            :param derf: derivative function
            :param ivars_to_check: solution indexes of variables to check for stability
        '''
        self.derf = derf
        self.ivars_to_check = ivars_to_check

    def getTimeVector(self, dt, f):
        ''' Compute reference integration time vector for a specific periodicity.

            :param dt: integration time step (s)
            :param f: periodic frequency (Hz)
            :return: time vector for 1 periodic cycle
        '''
        return np.linspace(0, 1 / f, int(np.round(1 / (f * dt))) + 1)

    def isPeriodicStable(self, y, ncycle, icycle):
        ''' Assess the periodic stabilization of a solution.

            :param y: solution matrix
            :param ncycle: number of samples per cycle
            :param icycle: index of cycle of interest
            :return: boolean stating whether the solution is stable or not
        '''
        y_target = y[icycle * ncycle: (icycle + 1) * ncycle, :]
        y_prec = y[(icycle - 1) * ncycle: icycle * ncycle, :]
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

    def compute(self, y0, dt, f):
        ''' Simulate system with a specific periodicity until stabilization.

            :param y0: 1D vector of initial conditions
            :param dt: integration time step (s)
            :param f: periodic frequency (Hz)
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        '''

        # Adjust variables to check if needed
        if self.ivars_to_check is None:
            self.ivars_to_check = range(y0.size)

        # Get reference time vector
        tref = self.getTimeVector(dt, f)

        # Initialize global arrays
        t = np.array([0.0])
        y = np.atleast_2d(y0)
        stim = np.array([1])

        # Integrate system for a few cycles until stabilization
        icycle = 0
        conv = False
        while not conv and icycle < NCYCLES_MAX:
            t, y, stim = self.integrate(t, y, stim, tref + icycle / f, self.derf, True)
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

    def __init__(self, derf_on, derf_off):
        ''' Initialize simulator with specific derivative functions

            :param derf_on: derivative function for ON periods
            :param derf_off: derivative function for OFF periods
        '''
        self.derf_on = derf_on
        self.derf_off = derf_off

    def getTimeVectors(self, dt, tstim, toffset, PRF, DC):
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
        logger.info(
            'Downsampling output arrays by factor %u (Fs = %sHz)',
            rf, si_format(1 / (dt * rf), 2)
        )
        t = t[::rf]
        y = y[::rf, :]
        stim = stim[::rf]
        return t, y, stim

    def compute(self, y0, dt, tstim, toffset, PRF, DC, target_dt=None, print_progress=False):
        ''' Simulate system for a specific stimulus application pattern.

            :param y0: 1D vector of initial conditions
            :param dt: integration time step (s)
            :param tstim: duration of US stimulation (s)
            :param toffset: duration of the offset (s)
            :param PRF: pulse repetition frequency (Hz)
            :param DC: pulse duty cycle (-)
            :target_dt: target time step after resampling
            :return: 3-tuple with the time profile, the effective solution matrix and a state vector
        '''

        # If CW stimulus: change PRF to have exactly one integration interval during stimulus
        if DC == 1.0:
            if not print_progress:
                PRF = 1 / tstim
            else:
                PRF = 100 / tstim
        npulses = int(np.round(tstim * PRF))

        # Get reference time vectors
        t_on, t_off, t_offset = self.getTimeVectors(dt, tstim, toffset, PRF, DC)

        # Initialize global arrays
        t = np.array([0.0])
        y = np.atleast_2d(y0)
        stim = np.array([1])

        # Initialize progress bar
        if print_progress:
            widgets = ['Running: ', pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
            pbar = pb.ProgressBar(widgets=widgets,
                                  max_value=int(npulses * (toffset + tstim) / tstim))
            pbar.start()

        # Integrate ON and OFF intervals of each pulse
        for i in range(npulses):
            for j, (tref, func) in enumerate(zip([t_on, t_off], [self.derf_on, self.derf_off])):
                t, y, stim = self.integrate(t, y, stim, tref + i / PRF, func, j == 0)

            # Update progress bar
            if print_progress:
                pbar.update(i)

        # Integrate offset interval
        t, y, stim = self.integrate(t, y, stim, t_offset, self.derf_off, False)

        # Terminate progress bar
        if print_progress:
            pbar.finish()

        # Resample solution if specified
        if target_dt is not None:
            t, y, stim = self.resample(t, y, stim, target_dt)

        # Return output variables
        return t, y, stim
