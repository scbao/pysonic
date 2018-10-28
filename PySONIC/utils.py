#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-19 22:30:46
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-28 15:26:03

''' Definition of generic utility functions used in other modules '''

import operator
import os
import math
import pickle
import re
import tkinter as tk
from tkinter import filedialog
import numpy as np
import colorlog
from scipy.interpolate import interp1d
import matplotlib


# Matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'


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


# File naming conventions
def ESTIM_filecode(neuron, Astim, tstim, PRF, DC):
    return 'ESTIM_{}_{}_{:.1f}mA_per_m2_{:.0f}ms{}'.format(
        neuron, 'CW' if DC == 1 else 'PW', Astim, tstim * 1e3,
        '_PRF{:.2f}Hz_DC{:.2f}%'.format(PRF, DC * 1e2) if DC < 1. else '')


def ASTIM_filecode(neuron, a, Fdrive, Adrive, tstim, PRF, DC, method):
    return 'ASTIM_{}_{}_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms_{}{}'.format(
        neuron, 'CW' if DC == 1 else 'PW', a * 1e9, Fdrive * 1e-3, Adrive * 1e-3, tstim * 1e3,
        'PRF{:.2f}Hz_DC{:.2f}%_'.format(PRF, DC * 1e2) if DC < 1. else '', method)


def MECH_filecode(a, Fdrive, Adrive, Qm):
    return 'MECH_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.1f}nCcm2'.format(
        a * 1e9, Fdrive * 1e-3, Adrive * 1e-3, Qm * 1e5)


rgxp = re.compile('(ESTIM|ASTIM)_([A-Za-z]*)_(.*).pkl')
rgxp_mech = re.compile('(MECH)_(.*).pkl')


# Figure naming conventions
def figtitle(meta):
    ''' Return appropriate title based on simulation metadata. '''
    if 'Cm0' in meta:
        return '{:.0f}nm BLS structure: MECH-STIM {:.0f}kHz, {:.0f}kPa'.format(
            meta['a'] * 1e9, meta['Fdrive'] * 1e-3, meta['Adrive'] * 1e-3)
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
            return '{} neuron: {} A-STIM {:.0f}kHz {:.0f}kPa, {:.0f}ms{}'.format(
                meta['neuron'], wavetype, meta['Fdrive'] * 1e-3, meta['Adrive'] * 1e-3,
                meta['tstim'] * 1e3, suffix)


timeunits = {
    'ASTIM': 't_ms',
    'ESTIM': 't_ms',
    'MECH': 't_us'
}


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


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


def getStimPulses(t, states):
    ''' Determine the onset and offset times of pulses from a stimulation vector.

        :param t: time vector (s).
        :param states: a vector of stimulation state (ON/OFF) at each instant in time.
        :return: 3-tuple with number of patches, timing of STIM-ON an STIM-OFF instants.
    '''

    # Compute states derivatives and identify bounds indexes of pulses
    dstates = np.diff(states)
    ipulse_on = np.insert(np.where(dstates > 0.0)[0] + 1, 0, 0)
    ipulse_off = np.where(dstates < 0.0)[0] + 1
    if ipulse_off.size < ipulse_on.size:
        ioff = t.size - 1
        if ipulse_off.size == 0:
            ipulse_off = np.array([ioff])
        else:
            ipulse_off = np.insert(ipulse_off, ipulse_off.size - 1, ioff)

    # Get time instants for pulses ON and OFF
    npulses = ipulse_on.size
    tpulse_on = t[ipulse_on]
    tpulse_off = t[ipulse_off]

    # return 3-tuple with #pulses, pulse ON and pulse OFF instants
    return npulses, tpulse_on, tpulse_off


def getNeuronLookupsFile(mechname):
    return os.path.join(
        os.path.split(__file__)[0],
        'neurons',
        '{}_lookups.pkl'.format(mechname))


def getLookups2D(mechname, a, Fdrive):
    ''' Retrieve appropriate 2D lookup tables and reference vectors
        for a given membrane mechanism, sonophore diameter and US frequency.

        :param mechname: name of membrane density mechanism
        :param a: sonophore diameter (m)
        :param Fdrive: US frequency (Hz)
        :return: 3-tuple with 1D numpy arrays of reference acoustic amplitudes and charge densities,
            and a dictionary of 2D lookup numpy arrays
    '''

    # Check lookup file existence
    lookup_path = getNeuronLookupsFile(mechname)
    if not os.path.isfile(lookup_path):
        raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))

    # Load lookups dictionary
    logger.debug('Loading lookup table')
    with open(lookup_path, 'rb') as fh:
        lookups4D = pickle.load(fh)

    # Retrieve 1D inputs from lookups dictionary
    aref = lookups4D.pop('a')
    Fref = lookups4D.pop('f')
    Aref = lookups4D.pop('A')
    Qref = lookups4D.pop('Q')

    # Check that sonophore diameter and US frequency are within lookup range
    a = isWithin('diameter', a, (aref.min(), aref.max()))
    Fdrive = isWithin('frequency', Fdrive, (Fref.min(), Fref.max()))

    # Interpolate 4D lookups at sonophore diameter and then at US frequency
    logger.debug('Interpolating lookups at (a = {}m, f = {}Hz)'.format(
        *si_format([a, Fdrive], space=' ')))
    lookups3D = {key: interp1d(aref, y4D, axis=0)(a) for key, y4D in lookups4D.items()}
    lookups2D = {key: interp1d(Fref, y3D, axis=0)(Fdrive) for key, y3D in lookups3D.items()}

    return Aref, Qref, lookups2D


def isWithin(name, val, bounds, rel_tol=1e-9):
    ''' Check if a floating point number is within an interval.

        If the value falls outside the interval, an error is raised.

        If the value falls just outside the interval due to rounding errors,
        the associated interval bound is returned.

        :param val: float value
        :param bounds: interval bounds (float tuple)
        :return: original or corrected value
    '''
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
