# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-05-28 14:45:12
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-15 20:00:03

import abc
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import ode, odeint, solve_ivp
from tqdm import tqdm

from ..utils import *
from ..constants import *


class ODESolver(metaclass=abc.ABCMeta):
    ''' Generic interface to ODE solver object. '''

    def __init__(self, ykeys, dfunc, dt=None):
        ''' Initialization.

            :param ykeys: list of differential variables names
            :param dfunc: derivative function
            :param dt: integration time step (s)
        '''
        self.ykeys = ykeys
        self.dfunc = dfunc
        self.dt = dt

    def checkFunc(self, key, value):
        if not callable(value):
            raise ValueError(f'{key} function must be a callable object')

    @property
    def ykeys(self):
        return self._ykeys

    @ykeys.setter
    def ykeys(self, value):
        if not isIterable(value):
            value = list(value)
        for item in value:
            if not isinstance(item, str):
                raise ValueError('ykeys must be a list of strings')
        self._ykeys = value

    @property
    def dfunc(self):
        return self._dfunc

    @dfunc.setter
    def dfunc(self, value):
        self.checkFunc('derivative', value)
        self._dfunc = value

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if value is None:
            self._dt = None
        else:
            if not isinstance(value, float):
                raise ValueError('time step must be float-typed')
            if value <= 0:
                raise ValueError('time step must be strictly positive')
            self._dt = value

    def getNSamples(self, t0, tend, dt=None):
        ''' Get the number of samples required to integrate from an initial to a final time with
            a specific time step.

            :param t0: initial time (s)
            :param tend: final time (s)
            :param dt: integration time step (s)
            :return: number of required samples, rounded to nearest integer
        '''
        if dt is None:
            dt = self.dt
        return int(np.round((tend - t0) / dt))

    def getTimeVector(self, t0, tend, **kwargs):
        ''' Get the time vector required to integrate from an initial to a final time with
            a specific time step.

            :param t0: initial time (s)
            :param tend: final time (s)
            :return: vector going from current time to target time with appropriate step (s)
        '''
        return np.linspace(t0, tend, self.getNSamples(t0, tend, **kwargs))

    def initialize(self, y0, t0=0.):
        ''' Initialize time vector and solution matrix.

            :param y0: dictionary of initial conditions
        '''
        keys = list(y0.keys())
        if len(keys) != len(self.ykeys):
            raise ValueError("Initial conditions do not match system's dimensions")
        for k in keys:
            if k not in self.ykeys:
                raise ValueError(f'{k} is not a differential variable')
        y0 = {k: np.asarray(v) if isIterable(v) else np.array([v]) for k, v in y0.items()}
        ref_size = y0[keys[0]].size
        if not all(v.size == ref_size for v in y0.values()):
            raise ValueError('dimensions of initial conditions are inconsistent')
        self.y = np.array(list(y0.values())).T
        self.t = np.ones(self.y.shape[0]) * t0
        self.x = np.zeros(self.t.size)

    def append(self, t, y):
        ''' Append to time vector and solution matrix.

            :param t: integration time vector for current interval
            :param y: derivative function for current interval
        '''
        self.t = np.concatenate((self.t, t))
        self.y = np.concatenate((self.y, y), axis=0)
        self.x = np.concatenate((self.x, np.ones(t.size) * self.xref))

    def bound(self, tbounds):
        ''' Bound to time vector and solution matrix. '''
        i_bounded = np.logical_and(self.t >= tbounds[0], self.t <= tbounds[1])
        self.t = self.t[i_bounded]
        self.y = self.y[i_bounded, :]
        self.x = self.x[i_bounded]

    def integrateUntil(self, target_t, remove_first=False):
        ''' Integrate system for a time interval and append to preceding solution arrays.

            :param target_t: target time (s)
            :param dt: integration time step (s)
        '''
        if target_t < self.t[-1]:
            raise ValueError(f'target time ({target_t} s) precedes current time {self.t[-1]} s')
        elif target_t == self.t[-1]:
            t, y = self.t[-1], self.y[-1]
        if self.dt is None:
            sol = solve_ivp(
                self.dfunc, [self.t[-1], target_t], self.y[-1], t_eval=t, method='LSODA')
            t, y = sol.t, sol.y.T
        else:
            t = self.getTimeVector(self.t[-1], target_t)
            y = odeint(self.dfunc, self.y[-1], t, tfirst=True)
        if remove_first:
            t, y = t[1:], y[1:]
        self.append(t, y)

    def resampleArray(self, t, y, target_dt):
        tnew = self.getTimeVector(t[0], t[-1], dt=target_dt)
        ynew = np.array([np.interp(tnew, t, x) for x in y.T]).T
        return tnew, ynew

    def resample(self, target_dt):
        ''' Resample solution to a new target time step.

            :target_dt: target time step (s)
        '''
        tnew, self.y = self.resampleArray(self.t, self.y, target_dt)
        self.x = interp1d(self.t, self.x, kind='nearest', assume_sorted=True)(tnew)
        self.t = tnew

    @abc.abstractmethod
    def solve(self, y0):
        ''' Simulate system for a given time interval with specific initial conditions.

            :param y0: dictionary of initial conditions
        '''
        raise NotImplementedError

    @property
    def solution(self):
        ''' Return solution as a pandas dataframe. '''
        return pd.DataFrame({
            't': self.t,
            'stimstate': self.x,
            **{k: self.y[:, i] for i, k in enumerate(self.ykeys)}
        })

    def __call__(self, *args, target_dt=None, **kwargs):
        ''' Call solve method and return solution dataframe. '''
        self.solve(*args, **kwargs)
        if target_dt is not None:
            self.resample(target_dt)
        return self.solution


class PeriodicSolver(ODESolver):
    ''' ODE solver with specific periodicity. '''

    def __init__(self, ykeys, dfunc, T, primary_vars=None, **kwargs):
        ''' Initialization.

            :param ykeys: list of differential variables names
            :param dfunc: derivative function
            :param T: periodicity (s)
            :param primary_vars: keys of the primary solution variables to check for stability
        '''
        super().__init__(ykeys, dfunc, **kwargs)
        self.T = T
        self.primary_vars = primary_vars

    @property
    def T(self):
        return self._T

    @T.setter
    def T(self, value):
        if not isinstance(value, float):
            raise ValueError('periodicity must be float-typed')
        if value <= 0:
            raise ValueError('periodicity must be strictly positive')
        self._T = value

    @property
    def primary_vars(self):
        return self._primary_vars

    @primary_vars.setter
    def primary_vars(self, value):
        if value is None:  # If none specified, set all variables to be checked for stability
            value = self.ykeys
        if not isIterable(value):
            value = [value]
        for item in value:
            if item not in self.ykeys:
                raise ValueError(f'{item} is not a differential variable')
        self._primary_vars = value

    @property
    def i_primary_vars(self):
        return [self.ykeys.index(k) for k in self.primary_vars]

    @property
    def xref(self):
        return 1.

    def getNPerCycle(self, dt):
        ''' Compute number of samples per cycle given a time step and a specific periodicity.

            :param dt: integration time step (s)
            :return: number of samples per cycle
        '''
        if isIterable(dt):  # if time vector is provided, compute dt from its last 2 elements
            dt = dt[-1] - dt[-2]
        return int(np.round(self.T / dt)) + 1

    def getLastCycle(self, i=0):
        ''' Get solution vector for the last ith cycle. '''
        n = self.getNPerCycle(self.t) - 1
        if i == 0:
            return self.y[-n:]
        return self.y[-(i + 1) * n:-i * n]

    def isPeriodicallyStable(self):
        ''' Assess the periodic stabilization of a solution, by evaluating the deviation
            of system variables between the last two periods.

            :return: boolean stating whether the solution is periodically stable or not
        '''
        # Extract the 2 cycles of interest from the solution
        y_last, y_prec = self.getLastCycle(i=0), self.getLastCycle(i=1)

        # For each variable of interest, evaluate the RMSE between the two cycles, the
        # variation range over the last cycle, and the ratio of these 2 quantities
        ratios = np.array([rmse(y_last[:, i], y_prec[:, i]) / np.ptp(y_last[:, i])
                           for i in self.i_primary_vars])

        # Classify the solution as periodically stable only if all RMSE/PTP ratios
        # are below critical threshold
        return np.all(ratios < MAX_RMSE_PTP_RATIO)

    def integrateCycle(self):
        ''' Integrate system for a cycle. '''
        self.integrateUntil(self.t[-1] + self.T, remove_first=True)

    def solve(self, y0, nmax=None, **kwargs):
        ''' Simulate system with a specific periodicity until stopping criterion is met.

            :param y0: dictionary of initial conditions
            :param nmax: maximum number of integration cycles (optional)
        '''
        if nmax is None:
            nmax = NCYCLES_MAX

        # Initialize system
        if y0 is not None:
            self.initialize(y0, **kwargs)

        # Integrate system for 2 cycles
        for i in range(2):
            self.integrateCycle()

        # Keep integrating system cyclically until stopping criterion is met
        while not self.isPeriodicallyStable() and i < nmax:
            self.integrateCycle()
            i += 1

        # Log stopping criterion
        t_str = f't = {self.t[-1] * 1e3:.5f} ms'
        if i == nmax:
            logger.warning(f'{t_str}: criterion not met -> stopping after {i} cycles')
        else:
            logger.debug(f'{t_str}: stopping criterion met after {i} cycles')


class EventDrivenSolver(ODESolver):
    ''' Event-driven ODE solver. '''

    def __init__(self, ykeys, dfunc, eventfunc, event_params=None, **kwargs):
        ''' Initialization.

            :param ykeys: list of differential variables names
            :param dfunc: derivatives  function
            :param eventfunc: function called on each event
            :param event_params: dictionary of parameters used by the derivatives function
        '''
        super().__init__(ykeys, dfunc, **kwargs)
        self.eventfunc = eventfunc
        self.assignEventParams(event_params)

    def assignEventParams(self, event_params):
        ''' Assign event parameters as instance attributes. '''
        if event_params is not None:
            for k, v in event_params.items():
                setattr(self, k, v)

    @property
    def eventfunc(self):
        return self._eventfunc

    @eventfunc.setter
    def eventfunc(self, value):
        self.checkFunc('event', value)
        self._eventfunc = value

    @property
    def xref(self):
        return self._xref

    @xref.setter
    def xref(self, value):
        self._xref = value

    def initialize(self, *args, **kwargs):
        self.xref = 0
        super().initialize(*args, **kwargs)

    def fireEvent(self, xevent):
        ''' Call event function and set new xref value. '''
        if xevent is not None:
            if xevent == 'log':
                self.logProgress()
            else:
                self.eventfunc(xevent)
                self.xref = xevent

    def initLog(self, logfunc, n):
        self.logfunc = logfunc
        if self.logfunc is None:
            setHandler(logger, TqdmHandler(my_log_formatter))
            self.pbar = tqdm(total=n)
        else:
            self.np = n
            logger.debug('integrating stimulus')

    def logProgress(self):
        if self.logfunc is None:
            self.pbar.update()
        else:
            logger.debug(f't = {self.t[-1] * 1e3:.5f} ms: {self.logfunc(self.y[-1])}')

    def terminateLog(self):
        if self.logfunc is None:
            self.pbar.close()
        else:
            logger.debug('integration completed')

    def sortEvents(self, events):
        return sorted(events, key=lambda x: x[0])

    def solve(self, y0, events, tstop, log_period=None, logfunc=None, **kwargs):
        ''' Simulate system for a specific stimulus application pattern.

            :param y0: 1D vector of initial conditions
            :param events: list of events
            :param tstop: stopping time (s)
        '''
        # Sort events according to occurrence time
        events = self.sortEvents(events)

        # Make sure all events occur before tstop
        if events[-1][0] > tstop:
            raise ValueError('all events must occur before stopping time')

        if log_period is not None:  # Add log events if any
            tlogs = np.arange(kwargs.get('t0', 0.), tstop, log_period)[1:]
            if tstop not in tlogs:
                tlogs = np.hstack((tlogs, [tstop]))
            events = self.sortEvents(events + [(t, 'log') for t in tlogs])
            self.initLog(logfunc, tlogs.size)
        else:  # Otherwise, add None event at tstop
            events.append((tstop, None))

        # Initialize system
        self.initialize(y0, **kwargs)

        # For each upcoming event
        for i, (tevent, xevent) in enumerate(events):
            self.integrateUntil(  # integrate until event time
                tevent,
                remove_first=i > 0 and events[i - 1][1] == 'log')
            self.fireEvent(xevent)  # fire event

        # Terminate log if any
        if log_period is not None:
            self.terminateLog()


class HybridSolver(EventDrivenSolver, PeriodicSolver):

    def __init__(self, ykeys, dfunc, dfunc_sparse, predfunc, eventfunc, T,
                 dense_vars, dt_dense, dt_sparse, **kwargs):
        ''' Initialization.

            :param ykeys: list of differential variables names
            :param dfunc: derivatives function
            :param dfunc_sparse: derivatives function for sparse integration periods
            :param predfunc: function computing the extra arguments necessary for sparse integration
            :param eventfunc: function called on each event
            :param T: periodicity (s)
            :param dense_vars: list of fast-evolving differential variables
            :param dt_dense: dense integration time step (s)
            :param dt_sparse: sparse integration time step (s)
        '''
        PeriodicSolver.__init__(
            self, ykeys, dfunc, T, primary_vars=kwargs.get('primary_vars', None), dt=dt_dense)
        self.eventfunc = eventfunc
        self.assignEventParams(kwargs.get('event_params', None))
        self.predfunc = predfunc
        self.dense_vars = dense_vars
        self.dt_sparse = dt_sparse
        self.sparse_solver = ode(dfunc_sparse)
        self.sparse_solver.set_integrator('dop853', nsteps=SOLVER_NSTEPS, atol=1e-12)

    @property
    def predfunc(self):
        return self._predfunc

    @predfunc.setter
    def predfunc(self, value):
        self.checkFunc('prediction', value)
        self._predfunc = value

    @property
    def dense_vars(self):
        return self._dense_vars

    @dense_vars.setter
    def dense_vars(self, value):
        if value is None:  # If none specified, set all variables as dense variables
            value = self.ykeys
        if not isIterable(value):
            value = [value]
        for item in value:
            if item not in self.ykeys:
                raise ValueError(f'{item} is not a differential variable')
        self._dense_vars = value

    @property
    def is_dense_var(self):
        return np.array([x in self.dense_vars for x in self.ykeys])

    @property
    def is_sparse_var(self):
        return np.invert(self.is_dense_var)

    def integrateSparse(self, ysparse, tf):
        ''' Integrate sparse system until a specific time. '''
        # Compute number of samples in the sparse cycle solution
        npc = ysparse.shape[0]

        # Initialize time vector and solution array for the current interval
        n = int(np.ceil((tf - self.t[-1]) / self.dt_sparse))
        t = np.linspace(self.t[-1], tf, n + 1)[1:]
        y = np.empty((n, self.y.shape[1]))

        # Initialize sparse integrator
        self.sparse_solver.set_initial_value(self.y[-1, self.is_sparse_var], self.t[-1])
        for i, tt in enumerate(t):
            # Integrate to next time only if dt is above given threshold
            if tt - self.sparse_solver.t > MIN_SPARSE_DT:
                self.sparse_solver.set_f_params(self.predfunc(ysparse[i % npc]))
                self.sparse_solver.integrate(tt)
                if not self.sparse_solver.successful():
                    raise ValueError(f'integration error at t = {tt * 1e3:.5f} ms')

            # Assign solution values (computed and propagated) to sparse solution array
            y[i, self.is_dense_var] = ysparse[i % npc, self.is_dense_var]
            y[i, self.is_sparse_var] = self.sparse_solver.y

        # Append to global solution
        self.append(t, y)

    def solve(self, y0, events, tstop, logfunc=None, **kwargs):
        ''' Integrate system using a hybrid scheme:

            - First, the full ODE system is integrated for a few cycles with a dense time
              granularity until a stopping criterion is met
            - Second, the profiles of all variables over the last cycle are downsampled to a
              far lower (i.e. sparse) sampling rate
            - Third, a subset of the ODE system is integrated with a sparse time granularity,
              for the remaining of the time interval, while the remaining variables are
              periodically expanded from their last cycle profile.
        '''
        # Sort events according to occurrence time
        events = self.sortEvents(events)

        # Make sure all events occur before tstop
        if events[-1][0] > tstop:
            raise ValueError('all events must occur before stopping time')
        # Add None event at tstop
        events.append((tstop, None))

        # Initialize system
        self.initialize(y0)

        # Initialize event iterator
        ievent = iter(events)
        tevent, xevent = next(ievent)
        stop = False

        # While final event is not reached
        while not stop:
            # Determine end-time of current interval
            tend = min(tevent, self.t[-1] + HYBRID_UPDATE_INTERVAL)

            # If time interval encompasses at least one cycle, solve periodic system
            nmax = int(np.round((tend - self.t[-1]) / self.T))
            if nmax > 0:
                PeriodicSolver.solve(self, None, nmax=nmax)

            # If end-time of current interval has been exceeded, bound solution to that time
            if self.t[-1] > tend:
                self.bound((self.t[0], tend))

            # If end-time of current interval has not been reached
            if self.t[-1] < tend:
                # Get solution over last cycle and resample it to sparse time step
                ylast = self.getLastCycle()
                tlast = self.t[-ylast.shape[0]:]
                _, ysparse = self.resampleArray(tlast, ylast, self.dt_sparse)

                # Integrate sparse system for the rest of the current interval
                self.integrateSparse(ysparse, tend)

            # If end-time corresponds to event, fire it and move to next event
            if self.t[-1] == tevent:
                self.fireEvent(xevent)
                try:
                    tevent, xevent = next(ievent)
                except StopIteration:
                    stop = True
