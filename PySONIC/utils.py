#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-19 22:30:46
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-07 18:21:56

''' Definition of generic utility functions used in other modules '''

import csv
from functools import wraps
import operator
import time
import os
import math
import pickle
from tqdm import tqdm
import logging
import tkinter as tk
from tkinter import filedialog
import numpy as np
import colorlog


# Package logger
my_log_formatter = colorlog.ColoredFormatter(
    '%(log_color)s %(asctime)s %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S:',
    reset=True,
    log_colors={
        'DEBUG': 'green',
        'INFO': 'white',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    },
    style='%')


def setHandler(logger, handler):
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.addHandler(handler)
    return logger


def setLogger(name, formatter):
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    logger = colorlog.getLogger(name)
    logger.addHandler(handler)
    return logger


class TqdmHandler(logging.StreamHandler):

    def __init__(self, formatter):
        logging.StreamHandler.__init__(self)
        self.setFormatter(formatter)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)


logger = setLogger('PySONIC', my_log_formatter)


# SI units prefixes
si_prefixes = {
    'y': 1e-24,  # yocto
    'z': 1e-21,  # zepto
    'a': 1e-18,  # atto
    'f': 1e-15,  # femto
    'p': 1e-12,  # pico
    'n': 1e-9,   # nano
    'u': 1e-6,   # micro
    'm': 1e-3,   # mili
    '': 1e0,    # None
    'k': 1e3,    # kilo
    'M': 1e6,    # mega
    'G': 1e9,    # giga
    'T': 1e12,   # tera
    'P': 1e15,   # peta
    'E': 1e18,   # exa
    'Z': 1e21,   # zetta
    'Y': 1e24,   # yotta
}


def si_format(x, precision=0, space=' '):
    ''' Format a float according to the SI unit system, with the appropriate prefix letter. '''
    if isinstance(x, float) or isinstance(x, int) or isinstance(x, np.float) or\
       isinstance(x, np.int32) or isinstance(x, np.int64):
        if x == 0:
            factor = 1e0
            prefix = ''
        else:
            sorted_si_prefixes = sorted(si_prefixes.items(), key=operator.itemgetter(1))
            vals = [tmp[1] for tmp in sorted_si_prefixes]
            # vals = list(si_prefixes.values())
            ix = np.searchsorted(vals, np.abs(x)) - 1
            if np.abs(x) == vals[ix + 1]:
                ix += 1
            factor = vals[ix]
            prefix = sorted_si_prefixes[ix][0]
            # prefix = list(si_prefixes.keys())[ix]
        return '{{:.{}f}}{}{}'.format(precision, space, prefix).format(x / factor)
    elif isinstance(x, list) or isinstance(x, tuple):
        return [si_format(item, precision, space) for item in x]
    elif isinstance(x, np.ndarray) and x.ndim == 1:
        return [si_format(float(item), precision, space) for item in x]
    else:
        print(type(x))


def pow10_format(number, precision=2):
    ''' Format a number in power of 10 notation. '''
    ret_string = '{0:.{1:d}e}'.format(number, precision)
    a, b = ret_string.split("e")
    a = float(a)
    b = int(b)
    return '{}10^{{{}}}'.format('{} * '.format(a) if a != 1. else '', b)


def plural(n):
    if n < 0:
        raise ValueError('Cannot format negative integer (n = {})'.format(n))
    if n == 0:
        return ''
    else:
        return 's'


def rmse(x1, x2):
    ''' Compute the root mean square error between two 1D arrays '''
    return np.sqrt(((x1 - x2) ** 2).mean())


def rsquared(x1, x2):
    ''' compute the R-squared coefficient between two 1D arrays '''
    residuals = x1 - x2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((x1 - np.mean(x1))**2)
    return 1 - (ss_res / ss_tot)


def Pressure2Intensity(p, rho=1075.0, c=1515.0):
    ''' Return the spatial peak, pulse average acoustic intensity (ISPPA)
        associated with the specified pressure amplitude.

        :param p: pressure amplitude (Pa)
        :param rho: medium density (kg/m3)
        :param c: speed of sound in medium (m/s)
        :return: spatial peak, pulse average acoustic intensity (W/m2)
    '''
    return p**2 / (2 * rho * c)


def Intensity2Pressure(I, rho=1075.0, c=1515.0):
    ''' Return the pressure amplitude associated with the specified
        spatial peak, pulse average acoustic intensity (ISPPA).

        :param I: spatial peak, pulse average acoustic intensity (W/m2)
        :param rho: medium density (kg/m3)
        :param c: speed of sound in medium (m/s)
        :return: pressure amplitude (Pa)
    '''
    return np.sqrt(2 * rho * c * I)


def OpenFilesDialog(filetype, dirname=''):
    ''' Open a FileOpenDialogBox to select one or multiple file.

        The default directory and file type are given.

        :param dirname: default directory
        :param filetype: default file type
        :return: tuple of full paths to the chosen filenames
    '''
    root = tk.Tk()
    root.withdraw()
    filenames = filedialog.askopenfilenames(
        filetypes=[(filetype + " files", '.' + filetype)],
        initialdir=dirname
    )
    if filenames:
        par_dir = os.path.abspath(os.path.join(filenames[0], os.pardir))
    else:
        par_dir = None
    return (filenames, par_dir)


def selectDirDialog(title='Select directory'):
    ''' Open a dialog box to select a directory.

        :return: full path to selected directory
    '''
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory(title=title)


def SaveFileDialog(filename, dirname=None, ext=None):
    ''' Open a dialog box to save file.

        :param filename: filename
        :param dirname: initial directory
        :param ext: default extension
        :return: full path to the chosen filename
    '''
    root = tk.Tk()
    root.withdraw()
    filename_out = filedialog.asksaveasfilename(
        defaultextension=ext, initialdir=dirname, initialfile=filename)
    return filename_out


def loadData(fpath, frequency=1):
    ''' Load dataframe and metadata dictionary from pickle file. '''
    logger.info('Loading data from "%s"', os.path.basename(fpath))
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
        df = frame['data'].iloc[::frequency]
        meta = frame['meta']
        return df, meta


def rescale(x, lb=None, ub=None, lb_new=0, ub_new=1):
    ''' Rescale a value to a specific interval by linear transformation. '''

    if lb is None:
        lb = x.min()
    if ub is None:
        ub = x.max()
    xnorm = (x - lb) / (ub - lb)
    return xnorm * (ub_new - lb_new) + lb_new


def isIterable(x):
    for t in [list, tuple, np.ndarray]:
        if isinstance(x, t):
            return True
    return False


def isWithin(name, val, bounds, rel_tol=1e-9):
    ''' Check if a floating point number is within an interval.

        If the value falls outside the interval, an error is raised.

        If the value falls just outside the interval due to rounding errors,
        the associated interval bound is returned.

        :param val: float value
        :param bounds: interval bounds (float tuple)
        :return: original or corrected value
    '''
    if isIterable(val):
        return [isWithin(name, v, bounds, rel_tol) for v in val]
    if val >= bounds[0] and val <= bounds[1]:
        return val
    elif val < bounds[0] and math.isclose(val, bounds[0], rel_tol=rel_tol):
        logger.warning('Rounding %s value (%s) to interval lower bound (%s)', name, val, bounds[0])
        return bounds[0]
    elif val > bounds[1] and math.isclose(val, bounds[1], rel_tol=rel_tol):
        logger.warning('Rounding %s value (%s) to interval upper bound (%s)', name, val, bounds[1])
        return bounds[1]
    else:
        raise ValueError('{} value ({}) out of [{}, {}] interval'.format(
            name, val, bounds[0], bounds[1]))


def getDistribution(xmin, xmax, nx, scale='lin'):
    if scale == 'log':
        xmin, xmax = np.log10(xmin), np.log10(xmax)
    return {'lin': np.linspace, 'log': np.logspace}[scale](xmin, xmax, nx)


def getDistFromList(xlist):
    if not isinstance(xlist, list):
        raise TypeError('Input must be a list')
    if len(xlist) != 4:
        raise ValueError('List must contain exactly 4 arguments ([type, min, max, n])')
    scale = xlist[0]
    if scale not in ('log', 'lin'):
        raise ValueError('Unknown distribution type (must be "lin" or "log")')
    xmin, xmax = [float(x) for x in xlist[1:-1]]
    if xmin >= xmax:
        raise ValueError('Specified minimum higher or equal than specified maximum')
    nx = int(xlist[-1])
    if nx < 2:
        raise ValueError('Specified number must be at least 2')
    return getDistribution(xmin, xmax, nx, scale=scale)


def getIndex(container, value):
    ''' Return the index of a float / string value in a list / array

        :param container: list / 1D-array of elements
        :param value: value to search for
        :return: index of value (if found)
    '''
    if isinstance(value, float):
        container = np.array(container)
        imatches = np.where(np.isclose(container, value, rtol=1e-9, atol=1e-16))[0]
        if len(imatches) == 0:
            raise ValueError('{} not found in {}'.format(value, container))
        return imatches[0]
    elif isinstance(value, str):
        return container.index(value)


def debug(func):
    ''' Print the function signature and return value. '''
    @wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = '{}({})'.format(func.__name__, ', '.join(args_repr + kwargs_repr))
        print('Calling {}'.format(signature))
        value = func(*args, **kwargs)
        print(f"{func.__name__!r} returned {value!r}")
        return value
    return wrapper_debug


def timer(func):
    ''' Monitor and return the runtime of the decorated function. '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        return value, run_time
    return wrapper


def logCache(fpath, delimiter='\t', out_type=float):
    ''' Add an extra IO memoization functionality to a function using file caching,
        to avoid repetitions of tedious computations with identical inputs.
    '''
    def wrapper_with_args(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # If function has history -> do not log
            if 'history' in kwargs:
                return func(*args, **kwargs)

            # Translate function arguments into string signature
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = '{}({})'.format(func.__name__, ', '.join(args_repr + kwargs_repr))

            # If entry present in log, return corresponding output
            if os.path.isfile(fpath):
                with open(fpath, 'r', newline='') as f:
                    reader = csv.reader(f, delimiter=delimiter)
                    for row in reader:
                        if row[0] == signature:
                            logger.info('entry found in "{}"'.format(os.path.basename(fpath)))
                            return out_type(row[1])

            # Otherwise, compute output and log it into file before returning
            out = func(*args, **kwargs)
            with open(fpath, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                writer.writerow([signature, str(out)])

            return out

        return wrapper

    return wrapper_with_args


def fileCache(root, fcode_func, load_func=pickle.load, dump_func=pickle.dump):

    def wrapper_with_args(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get file path from root and function arguments, using fcode function
            fpath = os.path.join(os.path.abspath(root), '{}.pkl'.format(fcode_func(*args)))

            # If file exists, load output from it
            if os.path.isfile(fpath):
                print('loading data from "{}"'.format(fpath))
                with open(fpath, 'rb') as f:
                    out = load_func(f)

            # Otherwise, execute function and create the file to dump the output
            else:
                out = func(*args, **kwargs)
                print('dumping data in "{}"'.format(fpath))
                with open(fpath, 'wb') as f:
                    dump_func(out, f)
            return out

        return wrapper

    return wrapper_with_args


def binarySearch(bool_func, args, ix, xbounds, dx_thr, history=None):
    ''' Use a binary search to determine the threshold satisfying a given condition
        within a continuous search interval.

        :param bool_func: boolean function returning whether condition is satisfied
        :param args: list of function arguments other than refined value
        :param xbounds: search interval for threshold (progressively refined)
        :param dx_thr: accuracy criterion for threshold
        :return: excitation threshold
    '''

    # Assign empty history if first function call
    if history is None:
        history = []

    # Compute function output at interval mid-point
    x = (xbounds[0] + xbounds[1]) / 2
    sim_args = args[:]
    sim_args.insert(ix, x)
    history.append(bool_func(sim_args))

    # If titration interval is small enough
    conv = False
    if (xbounds[1] - xbounds[0]) <= dx_thr:
        logger.debug('titration interval smaller than defined threshold')

        # If both conditions have been encountered during titration process,
        # we're going towards convergence
        if (0 in history and 1 in history):
            logger.debug('converging around threshold')

            # If current value satisfies condition, convergence is achieved
            # -> return threshold
            if history[-1]:
                logger.debug('currently satisfying condition -> convergence')
                return x

        # If only one condition has been encountered during titration process,
        # then no titration is impossible within the defined interval -> return NaN
        else:
            logger.warning('titration does not converge within this interval')
            return np.nan

    # Return threshold if convergence is reached, otherwise refine interval and iterate
    if conv:
        return x
    else:
        if x > 0.:
            xbounds = (xbounds[0], x) if history[-1] else (x, xbounds[1])
        else:
            xbounds = (x, xbounds[1]) if history[-1] else (xbounds[0], x)
        return binarySearch(bool_func, args, ix, xbounds, dx_thr, history=history)
