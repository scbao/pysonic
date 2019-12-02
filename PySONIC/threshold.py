# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-11-28 16:42:50
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-12-02 19:18:42

import numpy as np

from .utils import logger


class OutOfBoundsError(Exception):
    def __init__(self, bounds):
        msg = f'No threshold found within the [{bounds[0]:.2e} - {bounds[1]:.2e}] interval'
        super().__init__(msg)


class MaxNIterations(Exception):
    def __init__(self, max_nit, step, history):
        msg = f'Maximum number of iterations ({max_nit}) reached during {step} step, history = {history}'
        super().__init__(msg)


def threshold(feval, xbounds, x0=None, eps_thr=None, rel_eps_thr=1e-2, max_nit=50, precheck=False, fbound=2, output_history=False):
    ''' Determine the threshold satisfying a given condition within a continuous search interval,
        using a binary search with initial preconditioning.

        :param feval: evaluation function returning whether condition is satisfied for a given value
        :param xbounds: initial search interval for threshold
        :param x0: initial evaluation value
        :param eps_thr: maximum absolute error
        :param rel_eps_thr: maximum relative error
        :param precheck: boolean stating whether to perform an initial check
         for the existence of a threshold within the interval
        :param fbound: integer factor indicating the magnitude of the initial bounding procedure
        :param output_history: boolean stating whether to return history of search procedure
        :return: final threshold, or full search history
    '''
    err_val = np.nan

    # Define internal function to evaluate at the appropriate bound
    def checkAtBound():
        xeval = lb if is_above[-1] else ub
        if feval(xeval) == is_above[-1]:
            raise OutOfBoundsError(xbounds)

    # If factor bounding selected, lower bound cannot be zero
    if xbounds[0] == 0. and fbound is not None:
        # If absolute threshold is provided -> use it to set lower bound
        if eps_thr is not None:
            xbounds = (eps_thr / 2, xbounds[1])
        # Otherwise, use a very small lower bound (machine epsilon)
        else:
            eps_machine = np.sqrt(np.finfo(float).eps)
            xbounds = (eps_machine, xbounds[1])

    # Set absolute threshold to infinity if not specified
    if eps_thr is None:
        eps_thr = np.inf

    # Set initial value to geometric mean of search interval if not specified
    if x0 is None:
        x0 = np.sqrt(xbounds[0] * xbounds[1])

    # Adjust initial value to mid-point of search interval if set to zero
    if x0 == 0.:
        x0 = (xbounds[0] + xbounds[1]) / 2

    # Initialize internal variables
    lb, ub = xbounds           # lower and upper bound
    x = [x0]                   # history of evaluated values
    is_above = [feval(x[-1])]  # history of evaluation outcomes

    try:
        # Pre-checking: evaluate at the interval bound in the direction given by the initial
        # evaluation (above theshold -> lb, below threshold > ub) and return None
        # if evaluation indicates no threshold within interval
        if precheck:
            checkAtBound()

        # Pre-bounding: refine search interval by either multiplying or dividing x
        # by a specific integer factor k until target lies within an interval [x, kx]
        if fbound is not None:
            if 0 in xbounds:
                logger.warning('factor bounding unavailable when 0 is in the search bounds')
            else:
                # Assert compatibility of search interval with factor bounding
                assert ub / lb > 2 * fbound, f'search interval too narrow for factor bounding'

                # If exact match between x * f and ub or between lb * f and x, adapt f slightly
                if x[-1] * fbound == ub or lb * fbound == x[-1]:
                    fbound *= 0.99

                # Carry on only if bounding factor greater than 1
                if fbound >= 1:

                    # Iterate until both bounds have been updated
                    while lb == xbounds[0] or ub == xbounds[1]:
                        # Refine interval and x based on feval result
                        if is_above[-1]:
                            ub = x[-1]
                            x.append(ub / fbound)
                        else:
                            lb = x[-1]
                            x.append(fbound * lb)

                        # If lower bound greater or equal to upper bound -> error
                        if lb >= ub:
                            raise OutOfBoundsError(xbounds)

                        is_above.append(feval(x[-1]))

                        if len(x) >= max_nit:
                            raise MaxNIterations(max_nit, 'pre-bounding', x)

                    # Assert validity of refined interval
                    # assert ub / lb == fbound, f'restricted search interval should be of type [x, {fbound}x]'

                    # Set x to interval mid-point and re-evaluate
                    x.append((ub + lb) / 2)
                    is_above.append(feval(x[-1]))

        # Binary search until search interval is smaller than most stringent threshold criterion
        while not (np.abs(ub - lb) <= 2 * min(rel_eps_thr * lb, eps_thr)):
            # Refine interval based on feval result
            if is_above[-1]:
                ub = x[-1]
            else:
                lb = x[-1]

            # Set x to interval mid-point and re-evaluate
            x.append((ub + lb) / 2)
            is_above.append(feval(x[-1]))

            if len(x) >= max_nit:
                raise MaxNIterations(max_nit, 'binary search', x)

        # If search direction has not changed along the procedure, evaluate at appropriate bound
        if len(set(is_above)) == 1:
            checkAtBound()

        # At this point, upper bound is necessarily above threshold
        # and lower bound is necessarily below threshold

        # If last value is not above threshold
        if not is_above[-1]:
            # Set x to interval mid-point and re-evaluate (to ensure relative convergence)
            lb = x[-1]
            x.append((ub + lb) / 2)
            is_above.append(feval(x[-1]))
            # If last value still not above threshold, replace it by upper bound
            if not is_above[-1]:
                x[-1] = ub
                is_above[-1] = True

        # If precheck was ON, update history a posteriori
        if precheck:
            if is_above[0]:
                x1 = xbounds[0]
                is_above1 = False
            else:
                x1 = xbounds[1]
                is_above1 = True
            x = [x[0], x1] + x[1:]
            is_above = [is_above[0], is_above1] + is_above[1:]

    except (OutOfBoundsError, MaxNIterations) as err:
        # If precheck was ON, update history a posteriori
        if precheck:
            if is_above[0]:
                x1 = xbounds[0]
                is_above1 = False
            else:
                x1 = xbounds[1]
                is_above1 = True
            x = [x[0], x1, x1] + x[1:]
            is_above = [is_above[0], is_above1, is_above1] + is_above[1:]

        logger.error(err)
        x[-1] = err_val

    # Conditional return
    if output_history:
        return np.array(x), np.array(is_above)
    else:
        return x[-1]

