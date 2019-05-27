#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-19 22:30:46
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-27 14:40:21

''' Definition of generic utility functions used in other modules '''

import csv
from functools import wraps
import operator
import os
import math
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np
import colorlog
from scipy.interpolate import interp1d


# Package logger
def setLogger():
    log_formatter = colorlog.ColoredFormatter(
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
        style='%'
    )
    log_handler = colorlog.StreamHandler()
    log_handler.setFormatter(log_formatter)
    color_logger = colorlog.getLogger('PySONIC')
    color_logger.addHandler(log_handler)
    return color_logger


logger = setLogger()

titrations_logfile = os.path.join(os.path.split(__file__)[0], 'neurons', 'titrations.log')


# Figure naming conventions
def figtitle(meta):
    ''' Return appropriate title based on simulation metadata. '''
    if 'Cm0' in meta:
        return '{:.0f}nm radius BLS structure: MECH-STIM {:.0f}kHz, {:.2f}kPa, {:.1f}nC/cm2'.format(
            meta['a'] * 1e9, meta['Fdrive'] * 1e-3, meta['Adrive'] * 1e-3, meta['Qm'] * 1e5)
    else:
        if meta['DC'] < 1:
            wavetype = 'PW'
            suffix = ', {:.2f}Hz PRF, {:.0f}% DC'.format(meta['PRF'], meta['DC'] * 1e2)
        else:
            wavetype = 'CW'
            suffix = ''
        if 'Astim' in meta:
            return '{} neuron: {} E-STIM {:.2f}mA/m2, {:.0f}ms{}'.format(
                meta['neuron'], wavetype, meta['Astim'], meta['tstim'] * 1e3, suffix)
        else:
            return '{} neuron ({:.1f}nm): {} A-STIM {:.0f}kHz {:.2f}kPa, {:.0f}ms{} - {} model'.format(
                meta['neuron'], meta['a'] * 1e9, wavetype, meta['Fdrive'] * 1e-3,
                meta['Adrive'] * 1e-3, meta['tstim'] * 1e3, suffix, meta['method'])


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


def loadData(fpath, frequency=1):
    ''' Load dataframe and metadata dictionary from pickle file. '''
    logger.info('Loading data from "%s"', os.path.basename(fpath))
    with open(fpath, 'rb') as fh:
        frame = pickle.load(fh)
        df = frame['data'].iloc[::frequency]
        meta = frame['meta']
        return df, meta


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
    filenames = filedialog.askopenfilenames(filetypes=[(filetype + " files", '.' + filetype)],
                                            initialdir=dirname)
    if filenames:
        par_dir = os.path.abspath(os.path.join(filenames[0], os.pardir))
    else:
        par_dir = None
    return (filenames, par_dir)


def selectDirDialog():
    ''' Open a dialog box to select a directory.

        :return: full path to selected directory
    '''
    root = tk.Tk()
    root.withdraw()
    return filedialog.askdirectory()


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


def downsample(t_dense, y, nsparse):
    ''' Decimate periodic signals to a specified number of samples.'''

    if(y.ndim) > 1:
        nsignals = y.shape[0]
    else:
        nsignals = 1
        y = np.array([y])

    # determine time step and period of input signal
    T = t_dense[-1] - t_dense[0]
    dt_dense = t_dense[1] - t_dense[0]

    # resample time vector linearly
    t_ds = np.linspace(t_dense[0], t_dense[-1], nsparse)

    # create MAV window
    nmav = int(0.03 * T / dt_dense)
    if nmav % 2 == 0:
        nmav += 1
    mav = np.ones(nmav) / nmav

    # determine signals padding
    npad = int((nmav - 1) / 2)

    # determine indexes of sampling on convolved signals
    ids = np.round(np.linspace(0, t_dense.size - 1, nsparse)).astype(int)

    y_ds = np.empty((nsignals, nsparse))

    # loop through signals
    for i in range(nsignals):
        # pad, convolve and resample
        pad_left = y[i, -(npad + 2):-2]
        pad_right = y[i, 1:npad + 1]
        y_ext = np.concatenate((pad_left, y[i, :], pad_right), axis=0)
        y_mav = np.convolve(y_ext, mav, mode='valid')
        y_ds[i, :] = y_mav[ids]

    if nsignals == 1:
        y_ds = y_ds[0, :]

    return (t_ds, y_ds)


def rescale(x, lb=None, ub=None, lb_new=0, ub_new=1):
    ''' Rescale a value to a specific interval by linear transformation. '''

    if lb is None:
        lb = x.min()
    if ub is None:
        ub = x.max()
    xnorm = (x - lb) / (ub - lb)
    return xnorm * (ub_new - lb_new) + lb_new


def getNeuronLookupsFile(mechname, a=None, Fdrive=None, Adrive=None, fs=False):
    fpath = os.path.join(
        os.path.split(__file__)[0],
        'neurons',
        '{}_lookups'.format(mechname)
    )
    if a is not None:
        fpath += '_{:.0f}nm'.format(a * 1e9)
    if Fdrive is not None:
        fpath += '_{:.0f}kHz'.format(Fdrive * 1e-3)
    if Adrive is not None:
        fpath += '_{:.0f}kPa'.format(Adrive * 1e-3)
    if fs is True:
        fpath += '_fs'
    return '{}.pkl'.format(fpath)


def getLookups4D(mechname):
    ''' Retrieve 4D lookup tables and reference vectors for a given membrane mechanism.

        :param mechname: name of membrane density mechanism
        :return: 4-tuple with 1D numpy arrays of reference input vectors (charge density and
            one other variable), a dictionary of associated 2D lookup numpy arrays, and
            a dictionary with information about the other variable.
    '''

    # Check lookup file existence
    lookup_path = getNeuronLookupsFile(mechname)
    if not os.path.isfile(lookup_path):
        raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))

    # Load lookups dictionary
    # logger.debug('Loading %s lookup table', mechname)
    with open(lookup_path, 'rb') as fh:
        df = pickle.load(fh)
        inputs = df['input']
        lookups4D = df['lookup']

    # Retrieve 1D inputs from lookups dictionary
    aref = inputs['a']
    Fref = inputs['f']
    Aref = inputs['A']
    Qref = inputs['Q']

    return aref, Fref, Aref, Qref, lookups4D


def getLookupsOff(mechname):
    ''' Retrieve appropriate US-OFF lookup tables and reference vectors
        for a given membrane mechanism.

        :param mechname: name of membrane density mechanism
        :return: 2-tuple with 1D numpy array of reference charge density
            and dictionary of associated 1D lookup numpy arrays.
    '''

    # Get 4D lookups and input vectors
    aref, Fref, Aref, Qref, lookups4D = getLookups4D(mechname)

    # Perform 2D projection in appropriate dimensions
    logger.debug('Interpolating lookups at A = 0')
    lookups_off = {key: y4D[0, 0, 0, :] for key, y4D in lookups4D.items()}

    return Qref, lookups_off


def getLookups2D(mechname, a=None, Fdrive=None, Adrive=None):
    ''' Retrieve appropriate 2D lookup tables and reference vectors
        for a given membrane mechanism, projected at a specific combination
        of sonophore radius, US frequency and/or acoustic pressure amplitude.

        :param mechname: name of membrane density mechanism
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param Adrive: Acoustic peak pressure amplitude (Hz)
        :return: 4-tuple with 1D numpy arrays of reference input vectors (charge density and
            one other variable), a dictionary of associated 2D lookup numpy arrays, and
            a dictionary with information about the other variable.
    '''

    # Get 4D lookups and input vectors
    aref, Fref, Aref, Qref, lookups4D = getLookups4D(mechname)

    # Check that inputs are within lookup range
    if a is not None:
        a = isWithin('radius', a, (aref.min(), aref.max()))
    if Fdrive is not None:
        Fdrive = isWithin('frequency', Fdrive, (Fref.min(), Fref.max()))
    if Adrive is not None:
        Adrive = isWithin('amplitude', Adrive, (Aref.min(), Aref.max()))

    # Determine projection dimensions based on inputs
    var_a = {'name': 'a', 'label': 'sonophore radius', 'val': a, 'unit': 'm', 'factor': 1e9,
             'ref': aref, 'axis': 0}
    var_Fdrive = {'name': 'f', 'label': 'frequency', 'val': Fdrive, 'unit': 'Hz', 'factor': 1e-3,
                  'ref': Fref, 'axis': 1}
    var_Adrive = {'name': 'A', 'label': 'amplitude', 'val': Adrive, 'unit': 'Pa', 'factor': 1e-3,
                  'ref': Aref, 'axis': 2}
    if not isinstance(Adrive, float):
        var1 = var_a
        var2 = var_Fdrive
        var3 = var_Adrive
    elif not isinstance(Fdrive, float):
        var1 = var_a
        var2 = var_Adrive
        var3 = var_Fdrive
    elif not isinstance(a, float):
        var1 = var_Fdrive
        var2 = var_Adrive
        var3 = var_a

    # Perform 2D projection in appropriate dimensions
    # logger.debug('Interpolating lookups at (%s = %s%s, %s = %s%s)',
    #              var1['name'], si_format(var1['val'], space=' '), var1['unit'],
    #              var2['name'], si_format(var2['val'], space=' '), var2['unit'])
    lookups3D = {key: interp1d(var1['ref'], y4D, axis=var1['axis'])(var1['val'])
                 for key, y4D in lookups4D.items()}
    if var2['axis'] > var1['axis']:
        var2['axis'] -= 1
    lookups2D = {key: interp1d(var2['ref'], y3D, axis=var2['axis'])(var2['val'])
                 for key, y3D in lookups3D.items()}

    if var3['val'] is not None:
        logger.debug('Interpolating lookups at %d new %s values between %s%s and %s%s',
                     len(var3['val']), var3['name'],
                     si_format(min(var3['val']), space=' '), var3['unit'],
                     si_format(max(var3['val']), space=' '), var3['unit'])
        lookups2D = {key: interp1d(var3['ref'], y2D, axis=0)(var3['val'])
                     for key, y2D in lookups2D.items()}
        var3['ref'] = np.array(var3['val'])

    return var3['ref'], Qref, lookups2D, var3


def getLookups2Dfs(mechname, a, Fdrive, fs):

    # Check lookup file existence
    lookup_path = getNeuronLookupsFile(mechname, a=a, Fdrive=Fdrive, fs=True)
    if not os.path.isfile(lookup_path):
        raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))

    # Load lookups dictionary
    logger.debug('Loading %s lookup table with fs = %.0f%%', mechname, fs * 1e2)
    with open(lookup_path, 'rb') as fh:
        df = pickle.load(fh)
        inputs = df['input']
        lookups3D = df['lookup']

    # Retrieve 1D inputs from lookups dictionary
    fsref = inputs['fs']
    Aref = inputs['A']
    Qref = inputs['Q']

    # Check that fs is within lookup range
    fs = isWithin('coverage', fs, (fsref.min(), fsref.max()))

    # Perform projection at fs
    logger.debug('Interpolating lookups at fs = %s%%', fs * 1e2)
    lookups2D = {key: interp1d(fsref, y3D, axis=2)(fs) for key, y3D in lookups3D.items()}

    return Aref, Qref, lookups2D


def getLookupsDCavg(mechname, a, Fdrive, amps=None, charges=None, DCs=1.0):
    ''' Get the DC-averaged lookups of a specific neuron for a combination of US amplitudes,
        charge densities and duty cycles, at a specific US frequency.

        :param mechname: name of membrane density mechanism
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param amps: US amplitudes (Pa)
        :param charges: membrane charge densities (C/m2)
        :param DCs: duty cycle value(s)
        :return: 4-tuple with reference values of US amplitude and charge density,
            as well as interpolated Vmeff and QSS gating variables
    '''

    # Get lookups for specific (a, f, A) combination
    Aref, Qref, lookups2D, _ = getLookups2D(mechname, a=a, Fdrive=Fdrive)
    if 'ng' in lookups2D:
        lookups2D.pop('ng')

    # Derive inputs from lookups reference if not provided
    if amps is None:
        amps = Aref
    if charges is None:
        charges = Qref

    # Transform inputs into arrays if single value provided
    if isinstance(amps, float):
        amps = np.array([amps])
    if isinstance(charges, float):
        charges = np.array([charges])
    if isinstance(DCs, float):
        DCs = np.array([DCs])
    nA, nQ, nDC = amps.size, charges.size, DCs.size
    cs = {True: 's', False: ''}
    # logger.debug('%u amplitude%s, %u charge%s, %u DC%s',
    #              nA, cs[nA > 1], nQ, cs[nQ > 1], nDC, cs[nDC > 1])

    # Re-interpolate lookups at input charges
    lookups2D = {key: interp1d(Qref, y2D, axis=1)(charges) for key, y2D in lookups2D.items()}

    # Interpolate US-ON (for each input amplitude) and US-OFF (A = 0) lookups
    amps = isWithin('amplitude', amps, (Aref.min(), Aref.max()))
    lookups_on = {key: interp1d(Aref, y2D, axis=0)(amps) for key, y2D in lookups2D.items()}
    lookups_off = {key: interp1d(Aref, y2D, axis=0)(0.0) for key, y2D in lookups2D.items()}

    # Compute DC-averaged lookups
    lookups_DCavg = {}
    for key in lookups2D.keys():
        x_on, x_off = lookups_on[key], lookups_off[key]
        x_avg = np.empty((nA, nQ, nDC))
        for iA, Adrive in enumerate(amps):
            for iDC, DC in enumerate(DCs):
                x_avg[iA, :, iDC] = x_on[iA, :] * DC + x_off * (1 - DC)
        lookups_DCavg[key] = x_avg

    return amps, charges, lookups_DCavg


def isWithin(name, val, bounds, rel_tol=1e-9):
    ''' Check if a floating point number is within an interval.

        If the value falls outside the interval, an error is raised.

        If the value falls just outside the interval due to rounding errors,
        the associated interval bound is returned.

        :param val: float value
        :param bounds: interval bounds (float tuple)
        :return: original or corrected value
    '''
    if isinstance(val, list) or isinstance(val, np.ndarray) or isinstance(val, tuple):
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


def getLookupsCompTime(mechname):

    # Check lookup file existence
    lookup_path = getNeuronLookupsFile(mechname)
    if not os.path.isfile(lookup_path):
        raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))

    # Load lookups dictionary
    logger.debug('Loading comp times')
    with open(lookup_path, 'rb') as fh:
        df = pickle.load(fh)
        tcomps4D = df['tcomp']

    return np.sum(tcomps4D)


def getLowIntensitiesSTN():
    ''' Return an array of acoustic intensities (W/m2) used to study the STN neuron in
        Tarnaud, T., Joseph, W., Martens, L., and Tanghe, E. (2018). Computational Modeling
        of Ultrasonic Subthalamic Nucleus Stimulation. IEEE Trans Biomed Eng.
    '''
    return np.hstack((
        np.arange(10, 101, 10),
        np.arange(101, 131, 1),
        np.array([140])
    ))  # W/m2



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


def cache(fpath, delimiter='\t', out_type=float):
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


# def checkForFile(fpath):

#     def wrapper_with_args(func):

#         @wraps(func)
#         def wrapper(*args, **kwargs):

#             # Translate function arguments into string signature
#             args_repr = [repr(a) for a in args]
#             kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
#             signature = '{}({})'.format(func.__name__, ', '.join(args_repr + kwargs_repr))

#             # If entry present in log, return corresponding output
#             if os.path.isfile(fpath):
#                 with open(fpath, 'r', newline='') as f:
#                     reader = csv.reader(f, delimiter=delimiter)
#                     for row in reader:
#                         if row[0] == signature:
#                             logger.info('entry found in "{}"'.format(os.path.basename(fpath)))
#                             return out_type(row[1])

#             # Otherwise, compute output and log it into file before returning
#             out = func(*args, **kwargs)
#             with open(fpath, 'a', newline='') as csvfile:
#                 writer = csv.writer(csvfile, delimiter=delimiter)
#                 writer.writerow([signature, str(out)])

#             return out

#         return wrapper

#     return wrapper_with_args




@cache(titrations_logfile)
def titrate(xfunc, xargs, xbounds, dx_thr, history=None):
    ''' Use a binary search to determine the threshold satisfying a given condition
        within a specific search interval.

        :param xfunc: boolean function returning whether condition is satisfied
        :param xargs: list of function arguments other than refined value
        :param xbounds: search interval for threshold (progressively refined)
        :param dx_thr: accuracy criterion for threshold
        :return: excitation threshold
    '''

    # Assign empty history if first function call
    if history is None:
        history = []

    # Compute function output at interval mid-point
    x = (xbounds[0] + xbounds[1]) / 2
    history.append(xfunc(x, *xargs))

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
        return titrate(xfunc, xargs, xbounds, dx_thr, history=history)


def resolveDependencies(deps, join_items=True):
    ''' Solve a dictionary of dependencies.

        :param arg: dependency dictionary in which the values are the dependencies
         of their respective keys.
        :param join_items: boolean specifying whether or not to serialize output
        :return: list of inter-dependent elements in resolved order
    '''

    # Transform input dictionary of lists into dictionary of sets,
    # while removing circular (auto) dependencies
    deps = dict((k, set([x for x in deps[k] if x != k])) for k in deps)

    # Initialize empty list of resolved dependencies
    resolved_deps = []

    # Iterate while dependencies not entirely resolved
    while deps:
        # Extract latest items without dependencies (values that are not in keys
        # and keys without value) into a set
        nd_items = set(i for v in deps.values() for i in v) - set(deps.keys())
        nd_items.update(k for k, v in deps.items() if not v)

        # Append new set of non-dependent items to output list
        resolved_deps.append(nd_items)

        # Remove those items from remaining dependencies in input dictionary
        deps = dict(((k, v - nd_items) for k, v in deps.items() if v))

    # If specified, merge list of sets into a unique list (while preserving order)
    if join_items:
        tmp = []
        for item in resolved_deps:
            tmp += list(item)
        resolved_deps = tmp

    return resolved_deps
