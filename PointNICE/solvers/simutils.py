# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-07-04 11:48:51

""" Utility functions used in simulations """

import os
import time
import logging
import pickle
import shutil
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from openpyxl import load_workbook
import lockfile

from ..bls import BilayerSonophore
from .SolverUS import SolverUS
from .SolverElec import SolverElec
from ..constants import *
from ..utils import getNeuronsDict, InputError, PmCompMethod, si_format, getCycleAverage


# Get package logger
logger = logging.getLogger('PointNICE')

# Naming nomenclature for output files
MECH_code = 'MECH_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.1f}nCcm2'
ESTIM_CW_code = 'ESTIM_{}_CW_{:.1f}mA_per_m2_{:.0f}ms'
ESTIM_PW_code = 'ESTIM_{}_PW_{:.1f}mA_per_m2_{:.0f}ms_PRF{:.2f}Hz_DC{:.2f}%'
ASTIM_CW_code = 'ASTIM_{}_CW_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms_{}'
ASTIM_PW_code = 'ASTIM_{}_PW_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms_PRF{:.2f}Hz_DC{:.2f}%_{}'

# Parameters units
ASTIM_params = {
    'f': {'index': 0, 'factor': 1e-3, 'unit': 'kHz'},
    'A': {'index': 1, 'factor': 1e-3, 'unit': 'kPa'},
    't': {'index': 2, 'factor': 1e3, 'unit': 'ms'},
    'PRF': {'index': 4, 'factor': 1e-3, 'unit': 'kHz'},
    'DC': {'index': 5, 'factor': 1e2, 'unit': '%'}
}

ESTIM_params = {
    'A': {'index': 0, 'factor': 1e0, 'unit': 'mA/m2'},
    't': {'index': 1, 'factor': 1e3, 'unit': 'ms'},
    'PRF': {'index': 3, 'factor': 1e-3, 'unit': 'kHz'},
    'DC': {'index': 4, 'factor': 1e2, 'unit': '%'}
}

# Default geometry
default_diam = 32e-9
default_embedding = 0.0e-6


def setBatchDir():
    ''' Select batch directory for output files.α

        :return: full path to batch directory
    '''

    root = tk.Tk()
    root.withdraw()
    batch_dir = filedialog.askdirectory()
    if not batch_dir:
        raise InputError('No output directory chosen')
    return batch_dir


def checkBatchLog(batch_dir, batch_type):
    ''' Check for appropriate log file in batch directory, and create one if it is absent.

        :param batch_dir: full path to batch directory
        :param batch_type: type of simulation batch
        :return: 2 tuple with full path to log file and boolean stating if log file was created
    '''

    # Check for directory existence
    if not os.path.isdir(batch_dir):
        raise InputError('"{}" output directory does not exist'.format(batch_dir))

    # Determine log template from batch type
    if batch_type == 'MECH':
        logfile = 'log_MECH.xlsx'
    elif batch_type == 'A-STIM':
        logfile = 'log_ASTIM.xlsx'
    elif batch_type == 'E-STIM':
        logfile = 'log_ESTIM.xlsx'
    else:
        raise InputError('Unknown batch type', batch_type)

    # Get template in package subdirectory
    this_dir, _ = os.path.split(__file__)
    parent_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
    logsrc = parent_dir + '/templates/' + logfile
    assert os.path.isfile(logsrc), 'template log file "{}" not found'.format(logsrc)

    # Copy template in batch directory if no appropriate log file
    logdst = batch_dir + '/' + logfile
    is_log = os.path.isfile(logdst)
    if not is_log:
        shutil.copy2(logsrc, logdst)

    return (logdst, not is_log)


def createSimQueue(amps, durations, offsets, PRFs, DCs):
    ''' Create a serialized 2D array of all parameter combinations for a series of individual
        parameter sweeps, while avoiding repetition of CW protocols for a given PRF sweep.

        :param amps: list (or 1D-array) of acoustic amplitudes
        :param durations: list (or 1D-array) of stimulus durations
        :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
        :param PRFs: list (or 1D-array) of pulse-repetition frequencies
        :param DCs: list (or 1D-array) of duty cycle values
        :return: 2D-array with (amplitude, duration, offset, PRF, DC) for each stimulation protocol
    '''

    # Convert input to 1D-arrays
    amps = np.array(amps)
    durations = np.array(durations)
    offsets = np.array(offsets)
    PRFs = np.array(PRFs)
    DCs = np.array(DCs)

    # Create index arrays
    iamps = range(len(amps))
    idurs = range(len(durations))

    # Create empty output matrix
    queue = np.empty((1, 5))

    # Continuous protocols
    if 1.0 in DCs:
        nCW = len(amps) * len(durations)
        arr1 = np.ones(nCW)
        iCW_queue = np.array(np.meshgrid(iamps, idurs)).T.reshape(nCW, 2)
        CW_queue = np.vstack((amps[iCW_queue[:, 0]], durations[iCW_queue[:, 1]],
                              offsets[iCW_queue[:, 1]], PRFs.min() * arr1, arr1)).T
        queue = np.vstack((queue, CW_queue))


    # Pulsed protocols
    if np.any(DCs != 1.0):
        pulsed_DCs = DCs[DCs != 1.0]
        iPRFs = range(len(PRFs))
        ipulsed_DCs = range(len(pulsed_DCs))
        nPW = len(amps) * len(durations) * len(PRFs) * len(pulsed_DCs)
        iPW_queue = np.array(np.meshgrid(iamps, idurs, iPRFs, ipulsed_DCs)).T.reshape(nPW, 4)
        PW_queue = np.vstack((amps[iPW_queue[:, 0]], durations[iPW_queue[:, 1]],
                              offsets[iPW_queue[:, 1]], PRFs[iPW_queue[:, 2]],
                              pulsed_DCs[iPW_queue[:, 3]])).T
        queue = np.vstack((queue, PW_queue))

    # Return
    return queue[1:, :]


def xlslog(filename, sheetname, data):
    """ Append log data on a new row to specific sheet of excel workbook, using a lockfile
        to avoid read/write errors between concurrent processes.

        :param filename: absolute or relative path to the Excel workbook
        :param sheetname: name of the Excel spreadsheet to which data is appended
        :param data: data structure to be added to specific columns on a new row
        :return: boolean indicating success (1) or failure (0) of operation
    """
    try:
        lock = lockfile.FileLock(filename)
        lock.acquire()
        wb = load_workbook(filename)
        ws = wb[sheetname]
        keys = data.keys()
        i = 1
        row_data = {}
        for k in keys:
            row_data[k] = data[k]
            i += 1
        ws.append(row_data)
        wb.save(filename)
        lock.release()
        return 1
    except PermissionError:
        # If file cannot be accessed for writing because already opened
        logger.warning('Cannot write to "%s". Close the file and type "Y"', filename)
        user_str = input()
        if user_str in ['y', 'Y']:
            return xlslog(filename, sheetname, data)
        else:
            return 0


def detectPeaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                kpsh=False, valley=False, ax=None):
    '''
        Detect peaks in data based on their amplitude and other features.
        Adapted from Marco Duarte:
        http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    :param x: 1D array_like data.
    :param mph: minimum peak height (default = None).
    :param mpd: minimum peak distance in indexes (default = 1)
    :param threshold : minimum peak prominence (default = 0)
    :param edge : for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a flat peak (None).
        (default = 'rising')
    :param kpsh: keep peaks with same height even if they are closer than `mpd` (default = False).
    :param valley: detect valleys (local minima) instead of peaks (default = False).
    :param show: plot data in matplotlib figure (default = False).
    :param ax: a matplotlib.axes.Axes instance, optional (default = None).
    :return: 1D array with the indices of the peaks
'''
    print('min peak height:', mph, ', min peak distance:', mpd,
          ', min peak prominence:', threshold)

    # Convert input to numpy array
    x = np.atleast_1d(x).astype('float64')

    # Revert signal sign for valley detection
    if valley:
        x = -x

    # Differentiate signal
    dx = np.diff(x)

    # Find indices of all peaks with edge criterion
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))

    # Remove first and last values of x if they are detected as peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]

    print('{} raw peaks'.format(ind.size))

    # Remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
        print('{} height-filtered peaks'.format(ind.size))

    # Remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
        print('{} prominence-filtered peaks'.format(ind.size))

    # Detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
        print('{} distance-filtered peaks'.format(ind.size))

    return ind


def detectPeaksTime(t, y, mph, mtd, mpp=0):
    """ Extension of the detectPeaks function to detect peaks in data based on their
        amplitude and time difference, with a non-uniform time vector.

        :param t: time vector (not necessarily uniform)
        :param y: signal
        :param mph: minimal peak height
        :param mtd: minimal time difference
        :mpp: minmal peak prominence
        :return: array of peak indexes
    """

    # Determine whether time vector is uniform (threshold in time step variation)
    dt = np.diff(t)
    if (dt.max() - dt.min()) / dt.min() < 1e-2:
        isuniform = True
    else:
        isuniform = False

    if isuniform:
        print('uniform time vector')
        dt = t[1] - t[0]
        mpd = int(np.ceil(mtd / dt))
        ipeaks = detectPeaks(y, mph, mpd=mpd, threshold=mpp)
    else:
        print('non-uniform time vector')
        # Detect peaks on signal with no restriction on inter-peak distance
        irawpeaks = detectPeaks(y, mph, mpd=1, threshold=mpp)
        npeaks = irawpeaks.size
        if npeaks > 0:
            # Filter relevant peaks with temporal distance
            ipeaks = [irawpeaks[0]]
            for i in range(1, npeaks):
                i1 = ipeaks[-1]
                i2 = irawpeaks[i]
                if t[i2] - t[i1] < mtd:
                    if y[i2] > y[i1]:
                        ipeaks[-1] = i2
                else:
                    ipeaks.append(i2)
        else:
            ipeaks = []
        ipeaks = np.array(ipeaks)

    return ipeaks


def detectSpikes(t, Qm, min_amp, min_dt):
    ''' Detect spikes on a charge density signal, and
        return their number, latency and rate.

        :param t: time vector (s)
        :param Qm: charge density vector (C/m2)
        :param min_amp: minimal charge amplitude to detect spikes (C/m2)
        :param min_dt: minimal time interval between 2 spikes (s)
        :return: 3-tuple with number of spikes, latency (s) and spike rate (sp/s)
    '''
    i_spikes = detectPeaksTime(t, Qm, min_amp, min_dt)
    if len(i_spikes) > 0:
        latency = t[i_spikes[0]]  # s
        n_spikes = i_spikes.size
        if n_spikes > 1:
            first_to_last_spike = t[i_spikes[-1]] - t[i_spikes[0]]  # s
            spike_rate = (n_spikes - 1) / first_to_last_spike  # spikes/s
        else:
            spike_rate = 'N/A'
    else:
        latency = 'N/A'
        spike_rate = 'N/A'
        n_spikes = 0
    return (n_spikes, latency, spike_rate)


def findPeaks(y, mph=None, mpd=None, mpp=None):
    ''' Detect peaks in a signal based on their height, prominence and/or separating distance.

        :param y: signal vector
        :param mph: minimum peak height (in signal units, default = None).
        :param mpd: minimum inter-peak distance (in indexes, default = None)
        :param mpp: minimum peak prominence (in signal units, default = None)
        :return: 4-tuple of arrays with the indexes of peaks occurence, peaks prominence,
         peaks width at half-prominence and peaks half-prominence bounds (left and right)

        Adapted from:
        - Marco Duarte's detect_peaks function
          (http://nbviewer.jupyter.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb)
        - MATLAB findpeaks function (https://ch.mathworks.com/help/signal/ref/findpeaks.html)
    '''

    # Define empty output
    empty = (np.array([]),) * 4

    # Find all peaks and valleys
    dy = np.diff(y)
    s = np.sign(dy)
    ipeaks = np.where(np.diff(s) < 0)[0] + 1
    ivalleys = np.where(np.diff(s) > 0)[0] + 1

    # Return empty output if no peak detected
    if ipeaks.size == 0:
        return empty

    # Ensure each peak is bounded by two valleys, adding signal boundaries as valleys if necessary
    if ivalleys.size == 0 or ipeaks[0] < ivalleys[0]:
        ivalleys = np.insert(ivalleys, 0, 0)
    if ipeaks[-1] > ivalleys[-1]:
        ivalleys = np.insert(ivalleys, ivalleys.size, y.size - 1)
    # assert ivalleys.size - ipeaks.size == 1, 'Number of peaks and valleys not matching'
    if ivalleys.size - ipeaks.size != 1:
        logger.warning('detection incongruity: %u peaks vs. %u valleys detected',
                       ipeaks.size, ivalleys.size)
        return empty

    # Remove peaks < minimum peak height
    if mph is not None:
        ipeaks = ipeaks[y[ipeaks] >= mph]
    if ipeaks.size == 0:
        return empty

    # Detect small peaks closer than minimum peak distance
    if mpd is not None:
        ipeaks = ipeaks[np.argsort(y[ipeaks])][::-1]  # sort ipeaks by descending peak height
        idel = np.zeros(ipeaks.size, dtype=bool)  # initialize boolean deletion array (all false)
        for i in range(ipeaks.size):  # for each peak
            if not idel[i]:  # if not marked for deletion
                closepeaks = (ipeaks >= ipeaks[i] - mpd) & (ipeaks <= ipeaks[i] + mpd)  # close peaks
                idel = idel | closepeaks  # mark for deletion along with previously marked peaks
                # idel = idel | (ipeaks >= ipeaks[i] - mpd) & (ipeaks <= ipeaks[i] + mpd)
                idel[i] = 0  # keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ipeaks = np.sort(ipeaks[~idel])

    # Detect smallest valleys between consecutive relevant peaks
    ibottomvalleys = []
    if ipeaks[0] > ivalleys[0]:
        itrappedvalleys = ivalleys[ivalleys < ipeaks[0]]
        ibottomvalleys.append(itrappedvalleys[np.argmin(y[itrappedvalleys])])
    for i, j in zip(ipeaks[:-1], ipeaks[1:]):
        itrappedvalleys = ivalleys[np.logical_and(ivalleys > i, ivalleys < j)]
        ibottomvalleys.append(itrappedvalleys[np.argmin(y[itrappedvalleys])])
    if ipeaks[-1] < ivalleys[-1]:
        itrappedvalleys = ivalleys[ivalleys > ipeaks[-1]]
        ibottomvalleys.append(itrappedvalleys[np.argmin(y[itrappedvalleys])])
    ipeaks = ipeaks
    ivalleys = np.array(ibottomvalleys, dtype=int)

    # Ensure each peak is bounded by two valleys, adding signal boundaries as valleys if necessary
    if ipeaks[0] < ivalleys[0]:
        ivalleys = np.insert(ivalleys, 0, 0)
    if ipeaks[-1] > ivalleys[-1]:
        ivalleys = np.insert(ivalleys, ivalleys.size, y.size - 1)

    # Remove peaks < minimum peak prominence
    if mpp is not None:

        # Compute peaks prominences as difference between peaks and their closest valley
        prominences = y[ipeaks] - np.amax((y[ivalleys[:-1]], y[ivalleys[1:]]), axis=0)

        # initialize peaks and valleys deletion tables
        idelp = np.zeros(ipeaks.size, dtype=bool)
        idelv = np.zeros(ivalleys.size, dtype=bool)

        # for each peak (sorted by ascending prominence order)
        for ind in np.argsort(prominences):
            ipeak = ipeaks[ind]  # get peak index

            # get peak bases as first valleys on either side not marked for deletion
            indleftbase = ind
            indrightbase = ind + 1
            while idelv[indleftbase]:
                indleftbase -= 1
            while idelv[indrightbase]:
                indrightbase += 1
            ileftbase = ivalleys[indleftbase]
            irightbase = ivalleys[indrightbase]

            # Compute peak prominence and mark for deletion if < mpp
            indmaxbase = indleftbase if y[ileftbase] > y[irightbase] else indrightbase
            if y[ipeak] - y[ivalleys[indmaxbase]] < mpp:
                idelp[ind] = True  # mark peak for deletion
                idelv[indmaxbase] = True  # mark highest surrouding valley for deletion

        # remove irrelevant peaks and valleys, and sort back the indices by their occurrence
        ipeaks = np.sort(ipeaks[~idelp])
        ivalleys = np.sort(ivalleys[~idelv])

    if ipeaks.size == 0:
        return empty

    # Compute peaks prominences and reference half-prominence levels
    prominences = y[ipeaks] - np.amax((y[ivalleys[:-1]], y[ivalleys[1:]]), axis=0)
    refheights = y[ipeaks] - prominences / 2

    # Compute half-prominence bounds
    ibounds = np.empty((ipeaks.size, 2))
    for i in range(ipeaks.size):

        # compute the index of the left-intercept at half max
        ileft = ipeaks[i]
        while ileft >= ivalleys[i] and y[ileft] > refheights[i]:
            ileft -= 1
        if ileft < ivalleys[i]:  # intercept exactly on valley
            ibounds[i, 0] = ivalleys[i]
        else:  # interpolate intercept linearly between signal boundary points
            a = (y[ileft + 1] - y[ileft]) / 1
            b = y[ileft] - a * ileft
            ibounds[i, 0] = (refheights[i] - b) / a

        # compute the index of the right-intercept at half max
        iright = ipeaks[i]
        while iright <= ivalleys[i + 1] and y[iright] > refheights[i]:
            iright += 1
        if iright > ivalleys[i + 1]:  # intercept exactly on valley
            ibounds[i, 1] = ivalleys[i + 1]
        else:  # interpolate intercept linearly between signal boundary points
            if iright == y.size - 1:  # special case: if end of signal is reached, decrement iright
                iright -= 1
            a = (y[iright + 1] - y[iright]) / 1
            b = y[iright] - a * iright
            ibounds[i, 1] = (refheights[i] - b) / a

    # Compute peaks widths at half-prominence
    widths = np.diff(ibounds, axis=1)

    return (ipeaks, prominences, widths, ibounds)


def runMech(batch_dir, log_filepath, bls, Fdrive, Adrive, Qm):
    ''' Run a single simulation of the mechanical system with specific parameters and
        an imposed value of charge density, and save the results in a PKL file.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param bls: BilayerSonophore instance
        :param Fdrive: acoustic drive frequency (Hz)
        :param Adrive: acoustic drive amplitude (Pa)
        :param Qm: applided membrane charge density (C/m2)
        :return: full path to the output file
    '''

    simcode = MECH_code.format(bls.a * 1e9, Fdrive * 1e-3, Adrive * 1e-3, Qm * 1e5)

    # Get date and time info
    date_str = time.strftime("%Y.%m.%d")
    daytime_str = time.strftime("%H:%M:%S")

    # Run simulation
    tstart = time.time()
    (t, y, states) = bls.run(Fdrive, Adrive, Qm)
    (Z, ng) = y

    U = np.insert(np.diff(Z) / np.diff(t), 0, 0.0)
    tcomp = time.time() - tstart
    logger.debug('completed in %ss', si_format(tcomp, 1))

    # Store dataframe and metadata
    df = pd.DataFrame({'t': t, 'states': states, 'U': U, 'Z': Z, 'ng': ng})
    meta = {'a': bls.a, 'd': bls.d, 'Cm0': bls.Cm0, 'Qm0': bls.Qm0, 'Fdrive': Fdrive,
            'Adrive': Adrive, 'phi': np.pi, 'Qm': Qm, 'tcomp': tcomp}

    # Export into to PKL file
    output_filepath = '{}/{}.pkl'.format(batch_dir, simcode)
    with open(output_filepath, 'wb') as fh:
        pickle.dump({'meta': meta, 'data': df}, fh)
    logger.debug('simulation data exported to "%s"', output_filepath)

    # Compute key output metrics
    Zmax = np.amax(Z)
    Zmin = np.amin(Z)
    Zabs_max = np.amax(np.abs([Zmin, Zmax]))
    eAmax = bls.arealstrain(Zabs_max)
    Tmax = bls.TEtot(Zabs_max)
    Pmmax = bls.PMavgpred(Zmin)
    ngmax = np.amax(ng)
    dUdtmax = np.amax(np.abs(np.diff(U) / np.diff(t)**2))

    # Export key metrics to log file
    log = {
        'A': date_str,
        'B': daytime_str,
        'C': bls.a * 1e9,
        'D': bls.d * 1e6,
        'E': Fdrive * 1e-3,
        'F': Adrive * 1e-3,
        'G': Qm * 1e5,
        'H': t.size,
        'I': tcomp,
        'J': bls.kA + bls.kA_tissue,
        'K': Zmax * 1e9,
        'L': eAmax,
        'M': Tmax * 1e3,
        'N': (ngmax - bls.ng0) / bls.ng0,
        'O': Pmmax * 1e-3,
        'P': dUdtmax
    }

    if xlslog(log_filepath, 'Data', log) == 1:
        logger.info('log exported to "%s"', log_filepath)
    else:
        logger.error('log export to "%s" aborted', log_filepath)

    return output_filepath


def runMechBatch(batch_dir, log_filepath, Cm0, Qm0, stim_params, a=default_diam, d=default_embedding):
    ''' Run batch simulations of the mechanical system with imposed values of charge density,
        for various sonophore spans and stimulation parameters.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param Cm0: membrane resting capacitance (F/m2)
        :param Qm0: membrane resting charge density (C/m2)
        :param stim_params: dictionary containing sweeps for all stimulation parameters
        :param a: BLS in-plane diameter (m)
        :param d: depth of embedding tissue around plasma membrane (m)
    '''

    # Checking validity of stimulation parameters
    mandatory_params = ['freqs', 'amps', 'charges']
    for mp in mandatory_params:
        if mp not in stim_params:
            raise InputError('Missing stimulation parameter field: "{}"'.format(mp))

    # Define logging format
    MECH_log = ('Mechanical simulation %u/%u (a = %sm, d = %sm, f = %sHz, A = %sPa, Q = %sC/cm2)')

    logger.info("Starting mechanical simulation batch")

    # Unpack stimulation parameters
    amps = np.array(stim_params['amps'])
    charges = np.array(stim_params['charges'])

    # Generate simulations queue
    nA = len(amps)
    nQ = len(charges)
    sim_queue = np.array(np.meshgrid(amps, charges)).T.reshape(nA * nQ, 2)
    nqueue = sim_queue.shape[0]

    # Run simulations
    nsims = len(stim_params['freqs']) * nqueue
    simcount = 0
    filepaths = []
    for Fdrive in stim_params['freqs']:

        # Create BilayerSonophore instance (modulus of embedding tissue depends on frequency)
        bls = BilayerSonophore(a, Fdrive, Cm0, Qm0, d)

        for i in range(nqueue):
            simcount += 1
            Adrive, Qm = sim_queue[i, :]

            # Log
            logger.info(MECH_log, simcount, nsims, si_format(a, 1), si_format(d, 1),
                        si_format(Fdrive, 1), si_format(Adrive, 2), si_format(Qm * 1e-4))

            # Run simulation
            try:
                output_filepath = runMech(batch_dir, log_filepath, bls, Fdrive, Adrive, Qm)
                filepaths.append(output_filepath)
            except (Warning, AssertionError) as inst:
                logger.warning('Integration error: %s. Continue batch? (y/n)', extra={inst})
                user_str = input()
                if user_str not in ['y', 'Y']:
                    return filepaths

    return filepaths


def runEStim(batch_dir, log_filepath, solver, neuron, Astim, tstim, toffset, PRF, DC):
    ''' Run a single E-STIM simulation a given neuron for specific stimulation parameters,
        and save the results in a PKL file.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param solver: SolverElec instance
        :param Astim: pulse amplitude (mA/m2)
        :param tstim: pulse duration (s)
        :param toffset: offset duration (s)
        :param PRF: pulse repetition frequency (Hz)
        :param DC: pulse duty cycle (-)
        :return: full path to the output file
    '''

    if DC == 1.0:
        simcode = ESTIM_CW_code.format(neuron.name, Astim, tstim * 1e3)
    else:
        simcode = ESTIM_PW_code.format(neuron.name, Astim, tstim * 1e3, PRF, DC * 1e2)

    # Get date and time info
    date_str = time.strftime("%Y.%m.%d")
    daytime_str = time.strftime("%H:%M:%S")

    # Run simulation
    tstart = time.time()
    (t, y, states) = solver.run(neuron, Astim, tstim, toffset, PRF, DC)
    Vm, *channels = y
    tcomp = time.time() - tstart
    logger.debug('completed in %ss', si_format(tcomp, 1))

    # Store dataframe and metadata
    df = pd.DataFrame({'t': t, 'states': states, 'Vm': Vm, 'Qm': Vm * neuron.Cm0 * 1e-3})
    for j in range(len(neuron.states_names)):
        df[neuron.states_names[j]] = channels[j]
    meta = {'neuron': neuron.name, 'Astim': Astim, 'tstim': tstim, 'toffset': toffset,
            'PRF': PRF, 'DC': DC, 'tcomp': tcomp}

    # Export into to PKL file
    output_filepath = '{}/{}.pkl'.format(batch_dir, simcode)
    with open(output_filepath, 'wb') as fh:
        pickle.dump({'meta': meta, 'data': df}, fh)
    logger.debug('simulation data exported to "%s"', output_filepath)

    # Detect spikes on Vm signal
    # n_spikes, lat, sr = detectSpikes(t, Vm, SPIKE_MIN_VAMP, SPIKE_MIN_DT)
    dt = t[1] - t[0]
    ipeaks, *_ = findPeaks(Vm, SPIKE_MIN_VAMP, int(np.ceil(SPIKE_MIN_DT / dt)), SPIKE_MIN_VPROM)
    n_spikes = ipeaks.size
    lat = t[ipeaks[0]] if n_spikes > 0 else 'N/A'
    sr = np.mean(1 / np.diff(t[ipeaks])) if n_spikes > 1 else 'N/A'
    logger.debug('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

    # Export key metrics to log file
    log = {
        'A': date_str,
        'B': daytime_str,
        'C': neuron.name,
        'D': Astim,
        'E': tstim * 1e3,
        'F': PRF * 1e-3 if DC < 1 else 'N/A',
        'G': DC,
        'H': t.size,
        'I': round(tcomp, 2),
        'J': n_spikes,
        'K': lat * 1e3 if isinstance(lat, float) else 'N/A',
        'L': sr * 1e-3 if isinstance(sr, float) else 'N/A'
    }

    if xlslog(log_filepath, 'Data', log) == 1:
        logger.debug('log exported to "%s"', log_filepath)
    else:
        logger.error('log export to "%s" aborted', log_filepath)

    return output_filepath


def titrateEStim(solver, neuron, Astim, tstim, toffset, PRF=1.5e3, DC=1.0):
    """ Use a dichotomic recursive search to determine the threshold value of a specific
        electric stimulation parameter needed to obtain neural excitation, keeping all other
        parameters fixed. The titration parameter can be stimulation amplitude, duration or
        any variable for which the number of spikes is a monotonically increasing function.

        This function is called recursively until an accurate threshold is found.

        :param solver: solver instance
        :param neuron: neuron object
        :param Astim: injected current density amplitude (mA/m2)
        :param tstim: duration of US stimulation (s)
        :param toffset: duration of the offset (s)
        :param PRF: pulse repetition frequency (Hz)
        :param DC: pulse duty cycle (-)
        :return: 5-tuple with the determined amplitude threshold, time profile,
                 solution matrix, state vector and response latency
    """

    # Determine titration type
    if isinstance(Astim, tuple):
        t_type = 'A'
        interval = Astim
        thr = TITRATION_ESTIM_DA_MAX
        maxval = TITRATION_ESTIM_A_MAX
    elif isinstance(tstim, tuple):
        t_type = 't'
        interval = tstim
        thr = TITRATION_DT_THR
        maxval = TITRATION_T_MAX
    elif isinstance(DC, tuple):
        t_type = 'DC'
        interval = DC
        thr = TITRATION_DDC_THR
        maxval = TITRATION_DC_MAX
    else:
        logger.error('Invalid titration type')
        return 0.

    t_var = ESTIM_params[t_type]

    # Check amplitude interval and define current value
    if interval[0] >= interval[1]:
        raise InputError('Invaid {} interval: {} (must be defined as [lb, ub])'
                         .format(t_type, interval))
    value = (interval[0] + interval[1]) / 2

    # Define stimulation parameters
    if t_type == 'A':
        stim_params = [value, tstim, toffset, PRF, DC]
    elif t_type == 't':
        stim_params = [Astim, value, toffset, PRF, DC]
    elif t_type == 'DC':
        stim_params = [Astim, tstim, toffset, PRF, value]

    # Run simulation and detect spikes
    (t, y, states) = solver.run(neuron, *stim_params)
    # n_spikes, latency, _ = detectSpikes(t, y[0, :], SPIKE_MIN_VAMP, SPIKE_MIN_DT)
    dt = t[1] - t[0]
    ipeaks, *_ = findPeaks(y[0, :], SPIKE_MIN_VAMP, int(np.ceil(SPIKE_MIN_DT / dt)), SPIKE_MIN_VPROM)
    n_spikes = ipeaks.size
    latency = t[ipeaks[0]] if n_spikes > 0 else None
    logger.debug('%.2f %s ---> %u spike%s detected', value * t_var['factor'], t_var['unit'],
                 n_spikes, "s" if n_spikes > 1 else "")

    # If accurate threshold is found, return simulation results
    if (interval[1] - interval[0]) <= thr and n_spikes == 1:
        return (value, t, y, states, latency)

    # Otherwise, refine titration interval and iterate recursively
    else:
        if n_spikes == 0:
            if (maxval - interval[1]) <= thr:  # if upper bound too close to max then stop
                logger.warning('no spikes detected within titration interval')
                return (np.nan, t, y, states, latency)
            new_interval = (value, interval[1])
        else:
            new_interval = (interval[0], value)

        stim_params[t_var['index']] = new_interval
        return titrateEStim(solver, neuron, *stim_params)


def runEStimBatch(batch_dir, log_filepath, neurons, stim_params):
    ''' Run batch E-STIM simulations of the system for various neuron types and
        stimulation parameters.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param neurons: list of neurons names
        :param stim_params: dictionary containing sweeps for all stimulation parameters
        :return: list of full paths to the output files
    '''

    mandatory_params = ['amps', 'durations', 'offsets', 'PRFs', 'DCs']
    for mp in mandatory_params:
        if mp not in stim_params:
            raise InputError('Missing stimulation parameter field: "{}"'.format(mp))

    # Define logging format
    ESTIM_CW_log = 'E-STIM simulation %u/%u: %s neuron, A = %sA/m2, t = %ss'
    ESTIM_PW_log = 'E-STIM simulation %u/%u: %s neuron, A = %sA/m2, t = %ss, PRF = %sHz, DC = %.2f%%'

    logger.info("Starting E-STIM simulation batch")

    # Generate simulations queue
    sim_queue = createSimQueue(stim_params['amps'], stim_params['durations'],
                               stim_params['offsets'], stim_params['PRFs'], stim_params['DCs'])
    nqueue = sim_queue.shape[0]

    # Initialize solver
    solver = SolverElec()

    # Run simulations
    nsims = len(neurons) * nqueue
    simcount = 0
    filepaths = []
    for nname in neurons:
        neuron = getNeuronsDict()[nname]()
        for i in range(nqueue):
            simcount += 1
            Astim, tstim, toffset, PRF, DC = sim_queue[i, :]
            if DC == 1.0:
                logger.info(ESTIM_CW_log, simcount, nsims, neuron.name, si_format(Astim * 1e-3, 1),
                            si_format(tstim, 1))
            else:
                logger.info(ESTIM_PW_log, simcount, nsims, neuron.name, si_format(Astim * 1e-3, 1),
                            si_format(tstim, 1), si_format(PRF, 2), DC * 1e2)
            try:
                output_filepath = runEStim(batch_dir, log_filepath, solver, neuron,
                                           Astim, tstim, toffset, PRF, DC)
                filepaths.append(output_filepath)
            except (Warning, AssertionError) as inst:
                logger.warning('Integration error: %s. Continue batch? (y/n)', extra={inst})
                user_str = input()
                if user_str not in ['y', 'Y']:
                    return filepaths

    return filepaths


def titrateEStimBatch(batch_dir, log_filepath, neurons, stim_params):
    ''' Run batch electrical titrations of the system for various neuron types and
        stimulation parameters, to determine the threshold of a specific stimulus parameter
        for neural excitation.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param neurons: list of neurons names
        :param stim_params: dictionary containing sweeps for all stimulation parameters
        :return: list of full paths to the output files
    '''

    # Define logging format
    ESTIM_titration_log = '%s neuron - E-STIM titration %u/%u (%s)'

    logger.info("Starting E-STIM titration batch")

    # Determine titration parameter and titrations list
    if 'durations' not in stim_params:
        t_type = 't'
        sim_queue = createSimQueue(stim_params['amps'], [None], [TITRATION_T_OFFSET],
                                   stim_params['PRFs'], stim_params['DCs'])
    elif 'amps' not in stim_params:
        t_type = 'A'
        sim_queue = createSimQueue([None], stim_params['durations'],
                                   [TITRATION_T_OFFSET] * len(stim_params['durations']),
                                   stim_params['PRFs'], stim_params['DCs'])
    elif 'DC' not in stim_params:
        t_type = 'DC'
        sim_queue = createSimQueue(stim_params['amps'], stim_params['durations'],
                                   [TITRATION_T_OFFSET] * len(stim_params['durations']),
                                   stim_params['PRFs'], [None])

    nqueue = sim_queue.shape[0]
    t_var = ESTIM_params[t_type]

    # Create SolverElec instance
    solver = SolverElec()

    # Run titrations
    nsims = len(neurons) * nqueue
    simcount = 0
    filepaths = []
    for nname in neurons:
        neuron = getNeuronsDict()[nname]()
        for i in range(nqueue):
            simcount += 1

            # Extract parameters
            Astim, tstim, toffset, PRF, DC = sim_queue[i, :]
            if Astim is None:
                Astim = (0., 2 * TITRATION_ESTIM_A_MAX)
            elif tstim is None:
                tstim = (0., 2 * TITRATION_T_MAX)
            elif DC is None:
                DC = (0., 2 * TITRATION_DC_MAX)
            curr_params = [Astim, tstim, PRF, DC]

            # Generate log str
            log_str = ''
            pnames = list(ESTIM_params.keys())
            j = 0
            for cp in curr_params:
                pn = pnames[j]
                pi = ESTIM_params[pn]
                if not isinstance(cp, tuple):
                    if log_str:
                        log_str += ', '
                    log_str += '{} = {:.2f} {}'.format(pn, pi['factor'] * cp, pi['unit'])
                j += 1

            # Get date and time info
            date_str = time.strftime("%Y.%m.%d")
            daytime_str = time.strftime("%H:%M:%S")

            # Log
            logger.info(ESTIM_titration_log, neuron.name, simcount, nsims, log_str)

            # Run titration
            tstart = time.time()

            try:
                (output_thr, t, y, states, lat) = titrateEStim(solver, neuron, Astim,
                                                               tstim, toffset, PRF, DC)

                Vm, *channels = y
                tcomp = time.time() - tstart
                logger.info('completed in %ss, threshold = %.2f %s', si_format(tcomp, 2),
                            output_thr * t_var['factor'], t_var['unit'])

                # Determine output variable
                if t_type == 'A':
                    Astim = output_thr
                elif t_type == 't':
                    tstim = output_thr
                elif t_type == 'DC':
                    DC = output_thr

                # Define output naming
                if DC == 1.0:
                    simcode = ESTIM_CW_code.format(neuron.name, Astim, tstim * 1e3)
                else:
                    simcode = ESTIM_PW_code.format(neuron.name, Astim, tstim * 1e3,
                                                   PRF, DC * 1e2)

                # Store dataframe and metadata
                df = pd.DataFrame({'t': t, 'states': states, 'Vm': Vm})
                for j in range(len(neuron.states_names)):
                    df[neuron.states_names[j]] = channels[j]
                meta = {'neuron': neuron.name, 'Astim': Astim, 'tstim': tstim, 'toffset': toffset,
                        'PRF': PRF, 'DC': DC, 'tcomp': tcomp}

                # Export into to PKL file
                output_filepath = '{}/{}.pkl'.format(batch_dir, simcode)
                with open(output_filepath, 'wb') as fh:
                    pickle.dump({'meta': meta, 'data': df}, fh)
                logger.info('simulation data exported to "%s"', output_filepath)
                filepaths.append(output_filepath)

                # Detect spikes on Qm signal
                # n_spikes, lat, sr = detectSpikes(t, Vm, SPIKE_MIN_VAMP, SPIKE_MIN_DT)
                dt = t[1] - t[0]
                ipeaks, *_ = findPeaks(Vm, SPIKE_MIN_VAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                                       SPIKE_MIN_VPROM)
                n_spikes = ipeaks.size
                lat = t[ipeaks[0]] if n_spikes > 0 else 'N/A'
                sr = np.mean(1 / np.diff(t[ipeaks])) if n_spikes > 1 else 'N/A'
                logger.info('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

                # Export key metrics to log file
                log = {
                    'A': date_str,
                    'B': daytime_str,
                    'C': neuron.name,
                    'D': Astim,
                    'E': tstim * 1e3,
                    'F': PRF * 1e-3 if DC < 1 else 'N/A',
                    'G': DC,
                    'H': t.size,
                    'I': round(tcomp, 2),
                    'J': n_spikes,
                    'K': lat * 1e3 if isinstance(lat, float) else 'N/A',
                    'L': sr * 1e-3 if isinstance(sr, float) else 'N/A'
                }

                if xlslog(log_filepath, 'Data', log) == 1:
                    logger.info('log exported to "%s"', log_filepath)
                else:
                    logger.error('log export to "%s" aborted', log_filepath)

            except (Warning, AssertionError) as inst:
                logger.warning('Integration error: %s. Continue batch? (y/n)', extra={inst})
                user_str = input()
                if user_str not in ['y', 'Y']:
                    return filepaths

    return filepaths


def runAStim(batch_dir, log_filepath, solver, neuron, Fdrive, Adrive, tstim, toffset, PRF, DC,
             int_method='effective'):
    ''' Run a single A-STIM simulation a given neuron for specific stimulation parameters,
        and save the results in a PKL file.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param solver: SolverUS instance
        :param Fdrive: acoustic drive frequency (Hz)
        :param Adrive: acoustic drive amplitude (Pa)
        :param tstim: duration of US stimulation (s)
        :param toffset: duration of the offset (s)
        :param PRF: pulse repetition frequency (Hz)
        :param DC: pulse duty cycle (-)
        :param int_method: selected integration method
        :return: full path to the output file
    '''

    if DC == 1.0:
        simcode = ASTIM_CW_code.format(neuron.name, solver.a * 1e9, Fdrive * 1e-3, Adrive * 1e-3,
                                       tstim * 1e3, int_method)
    else:
        simcode = ASTIM_PW_code.format(neuron.name, solver.a * 1e9, Fdrive * 1e-3, Adrive * 1e-3,
                                       tstim * 1e3, PRF, DC * 1e2, int_method)

    # Get date and time info
    date_str = time.strftime("%Y.%m.%d")
    daytime_str = time.strftime("%H:%M:%S")

    # Run simulation
    tstart = time.time()
    (t, y, states) = solver.run(neuron, Fdrive, Adrive, tstim, toffset, PRF, DC, int_method)
    Z, ng, Qm, Vm, *channels = y
    U = np.insert(np.diff(Z) / np.diff(t), 0, 0.0)
    tcomp = time.time() - tstart
    logger.debug('completed in %ss', si_format(tcomp, 2))

    # Store dataframe and metadata
    df = pd.DataFrame({'t': t, 'states': states, 'U': U, 'Z': Z, 'ng': ng, 'Qm': Qm,
                       'Vm': Vm})
    for j in range(len(neuron.states_names)):
        df[neuron.states_names[j]] = channels[j]
    meta = {'neuron': neuron.name, 'a': solver.a, 'd': solver.d, 'Fdrive': Fdrive,
            'Adrive': Adrive, 'phi': np.pi, 'tstim': tstim, 'toffset': toffset, 'PRF': PRF,
            'DC': DC, 'tcomp': tcomp}

    # Export into to PKL file
    output_filepath = '{}/{}.pkl'.format(batch_dir, simcode)
    with open(output_filepath, 'wb') as fh:
        pickle.dump({'meta': meta, 'data': df}, fh)
    logger.debug('simulation data exported to "%s"', output_filepath)

    # Detect spikes on Qm signal
    # n_spikes, lat, sr = detectSpikes(t, Qm, SPIKE_MIN_QAMP, SPIKE_MIN_DT)
    dt = t[1] - t[0]
    ipeaks, *_ = findPeaks(Qm, SPIKE_MIN_QAMP, int(np.ceil(SPIKE_MIN_DT / dt)), SPIKE_MIN_QPROM)
    n_spikes = ipeaks.size
    lat = t[ipeaks[0]] if n_spikes > 0 else 'N/A'
    sr = np.mean(1 / np.diff(t[ipeaks])) if n_spikes > 1 else 'N/A'
    logger.debug('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

    # Export key metrics to log file
    log = {
        'A': date_str,
        'B': daytime_str,
        'C': neuron.name,
        'D': solver.a * 1e9,
        'E': solver.d * 1e6,
        'F': Fdrive * 1e-3,
        'G': Adrive * 1e-3,
        'H': tstim * 1e3,
        'I': PRF * 1e-3 if DC < 1 else 'N/A',
        'J': DC,
        'K': int_method,
        'L': t.size,
        'M': round(tcomp, 2),
        'N': n_spikes,
        'O': lat * 1e3 if isinstance(lat, float) else 'N/A',
        'P': sr * 1e-3 if isinstance(sr, float) else 'N/A'
    }

    if xlslog(log_filepath, 'Data', log) == 1:
        logger.debug('log exported to "%s"', log_filepath)
    else:
        logger.error('log export to "%s" aborted', log_filepath)

    return output_filepath


def titrateAStim(solver, neuron, Fdrive, Adrive, tstim, toffset, PRF=1.5e3, DC=1.0,
                 int_method='effective'):
    """ Use a dichotomic recursive search to determine the threshold value of a specific
        acoustic stimulation parameter needed to obtain neural excitation, keeping all other
        parameters fixed. The titration parameter can be stimulation amplitude, duration or
        any variable for which the number of spikes is a monotonically increasing function.

        This function is called recursively until an accurate threshold is found.

        :param solver: solver instance
        :param neuron: neuron object
        :param Fdrive: acoustic drive frequency (Hz)
        :param Adrive: acoustic drive amplitude (Pa)
        :param tstim: duration of US stimulation (s)
        :param toffset: duration of the offset (s)
        :param PRF: pulse repetition frequency (Hz)
        :param DC: pulse duty cycle (-)
        :param int_method: selected integration method
        :return: 5-tuple with the determined amplitude threshold, time profile,
                 solution matrix, state vector and response latency
    """

    # Determine titration type
    if isinstance(Adrive, tuple):
        t_type = 'A'
        interval = Adrive
        thr = TITRATION_ASTIM_DA_MAX
        maxval = TITRATION_ASTIM_A_MAX
    elif isinstance(tstim, tuple):
        t_type = 't'
        interval = tstim
        thr = TITRATION_DT_THR
        maxval = TITRATION_T_MAX
    elif isinstance(DC, tuple):
        t_type = 'DC'
        interval = DC
        thr = TITRATION_DDC_THR
        maxval = TITRATION_DC_MAX
    else:
        logger.error('Invalid titration type')
        return 0.

    t_var = ASTIM_params[t_type]

    # Check amplitude interval and define current value
    if interval[0] >= interval[1]:
        raise InputError('Invaid {} interval: {} (must be defined as [lb, ub])'
                         .format(t_type, interval))
    value = (interval[0] + interval[1]) / 2

    # Define stimulation parameters
    if t_type == 'A':
        stim_params = [Fdrive, value, tstim, toffset, PRF, DC]
    elif t_type == 't':
        stim_params = [Fdrive, Adrive, value, toffset, PRF, DC]
    elif t_type == 'DC':
        stim_params = [Fdrive, Adrive, tstim, toffset, PRF, value]

    # Run simulation and detect spikes
    (t, y, states) = solver.run(neuron, *stim_params, int_method)
    # n_spikes, latency, _ = detectSpikes(t, y[2, :], SPIKE_MIN_QAMP, SPIKE_MIN_DT)
    dt = t[1] - t[0]
    ipeaks, *_ = findPeaks(y[2, :], SPIKE_MIN_QAMP, int(np.ceil(SPIKE_MIN_DT / dt)), SPIKE_MIN_QPROM)
    n_spikes = ipeaks.size
    latency = t[ipeaks[0]] if n_spikes > 0 else None
    logger.debug('%.2f %s ---> %u spike%s detected', value * t_var['factor'], t_var['unit'],
                 n_spikes, "s" if n_spikes > 1 else "")

    # If accurate threshold is found, return simulation results
    if (interval[1] - interval[0]) <= thr and n_spikes == 1:
        return (value, t, y, states, latency)

    # Otherwise, refine titration interval and iterate recursively
    else:
        if n_spikes == 0:
            if (maxval - interval[1]) <= thr:  # if upper bound too close to max then stop
                logger.warning('no spikes detected within titration interval')
                return (np.nan, t, y, states, latency)
            new_interval = (value, interval[1])
        else:
            new_interval = (interval[0], value)

        stim_params[t_var['index']] = new_interval
        return titrateAStim(solver, neuron, *stim_params, int_method)


def runAStimBatch(batch_dir, log_filepath, neurons, stim_params, a=default_diam,
                  int_method='effective'):
    ''' Run batch simulations of the system for various neuron types, sonophore and
        stimulation parameters.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param neurons: list of neurons names
        :param stim_params: dictionary containing sweeps for all stimulation parameters
        :param a: BLS structure diameter (m)
        :param int_method: selected integration method
        :return: list of full paths to the output files
    '''

    mandatory_params = ['freqs', 'amps', 'durations', 'offsets', 'PRFs', 'DCs']
    for mp in mandatory_params:
        if mp not in stim_params:
            raise InputError('Missing stimulation parameter field: "{}"'.format(mp))

    # Define logging format
    ASTIM_CW_log = 'A-STIM %s simulation %u/%u: %s neuron, a = %sm, f = %sHz, A = %sPa, t = %ss'
    ASTIM_PW_log = ('A-STIM %s simulation %u/%u: %s neuron, a = %sm, f = %sHz, '
                    'A = %sPa, t = %ss, PRF = %sHz, DC = %.2f%%')

    logger.info("Starting A-STIM simulation batch")

    # Generate simulations queue
    sim_queue = createSimQueue(stim_params['amps'], stim_params['durations'],
                               stim_params['offsets'], stim_params['PRFs'], stim_params['DCs'])
    nqueue = sim_queue.shape[0]

    # Run simulations
    nsims = len(neurons) * len(stim_params['freqs']) * nqueue
    simcount = 0
    filepaths = []
    for nname in neurons:
        neuron = getNeuronsDict()[nname]()
        for Fdrive in stim_params['freqs']:

            # Initialize SolverUS
            solver = SolverUS(a, neuron, Fdrive)

            for i in range(nqueue):

                simcount += 1
                Adrive, tstim, toffset, PRF, DC = sim_queue[i, :]

                # Log and define naming
                if DC == 1.0:
                    logger.info(ASTIM_CW_log, int_method, simcount, nsims, neuron.name,
                                si_format(a, 1), si_format(Fdrive, 1), si_format(Adrive, 2),
                                si_format(tstim, 1))
                else:
                    logger.info(ASTIM_PW_log, int_method, simcount, nsims, neuron.name,
                                si_format(a, 1), si_format(Fdrive, 1), si_format(Adrive, 2),
                                si_format(tstim, 1), si_format(PRF, 2), DC * 1e2)

                # Run simulation
                try:
                    output_filepath = runAStim(batch_dir, log_filepath, solver, neuron, Fdrive,
                                               Adrive, tstim, toffset, PRF, DC, int_method)
                    filepaths.append(output_filepath)
                except (Warning, AssertionError) as inst:
                    logger.warning('Integration error: %s. Continue batch? (y/n)', extra={inst})
                    user_str = input()
                    if user_str not in ['y', 'Y']:
                        return filepaths

    return filepaths


def titrateAStimBatch(batch_dir, log_filepath, neurons, stim_params, a=default_diam,
                      int_method='effective'):
    ''' Run batch acoustic titrations of the system for various neuron types, sonophore and
        stimulation parameters, to determine the threshold of a specific stimulus parameter
        for neural excitation.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param neurons: list of neurons names
        :param stim_params: dictionary containing sweeps for all stimulation parameters
        :param a: BLS structure diameter (m)
        :param int_method: selected integration method
        :return: list of full paths to the output files
    '''

    # Define logging format
    ASTIM_titration_log = '%s neuron - A-STIM titration %u/%u (a = %sm, %s)'

    logger.info("Starting A-STIM titration batch")

    # Define default parameters
    int_method = 'effective'

    # Determine titration parameter and titrations list
    if 'durations' not in stim_params:
        t_type = 't'
        sim_queue = createSimQueue(stim_params['amps'], [None], [TITRATION_T_OFFSET],
                                   stim_params['PRFs'], stim_params['DCs'])
    elif 'amps' not in stim_params:
        t_type = 'A'
        sim_queue = createSimQueue([None], stim_params['durations'],
                                   [TITRATION_T_OFFSET] * len(stim_params['durations']),
                                   stim_params['PRFs'], stim_params['DCs'])
    elif 'DC' not in stim_params:
        t_type = 'DC'
        sim_queue = createSimQueue(stim_params['amps'], stim_params['durations'],
                                   [TITRATION_T_OFFSET] * len(stim_params['durations']),
                                   stim_params['PRFs'], [None])

    nqueue = sim_queue.shape[0]
    t_var = ASTIM_params[t_type]

    # Run titrations
    nsims = len(neurons) * len(stim_params['freqs']) * nqueue
    simcount = 0
    filepaths = []
    for nname in neurons:
        neuron = getNeuronsDict()[nname]()
        for Fdrive in stim_params['freqs']:
            # Create SolverUS instance (modulus of embedding tissue depends on frequency)
            solver = SolverUS(a, neuron, Fdrive)

            for i in range(nqueue):
                simcount += 1

                # Extract parameters
                Adrive, tstim, toffset, PRF, DC = sim_queue[i, :]
                if Adrive is None:
                    Adrive = (0., 2 * TITRATION_ASTIM_A_MAX)
                elif tstim is None:
                    tstim = (0., 2 * TITRATION_T_MAX)
                elif DC is None:
                    DC = (0., 2 * TITRATION_DC_MAX)
                curr_params = [Fdrive, Adrive, tstim, PRF, DC]

                # Generate log str
                log_str = ''
                pnames = list(ASTIM_params.keys())
                j = 0
                for cp in curr_params:
                    pn = pnames[j]
                    pi = ASTIM_params[pn]
                    if not isinstance(cp, tuple):
                        if log_str:
                            log_str += ', '
                        log_str += '{} = {:.2f} {}'.format(pn, pi['factor'] * cp, pi['unit'])
                    j += 1

                # Get date and time info
                date_str = time.strftime("%Y.%m.%d")
                daytime_str = time.strftime("%H:%M:%S")

                # Log
                logger.info(ASTIM_titration_log, neuron.name, simcount, nsims, si_format(a, 1),
                            log_str)

                # Run titration
                tstart = time.time()
                try:
                    (output_thr, t, y, states, lat) = titrateAStim(solver, neuron, Fdrive,
                                                                   Adrive, tstim, toffset,
                                                                   PRF, DC)

                    Z, ng, Qm, Vm, *channels = y
                    U = np.insert(np.diff(Z) / np.diff(t), 0, 0.0)
                    tcomp = time.time() - tstart
                    logger.info('completed in %.2f s, threshold = %.2f %s', tcomp,
                                output_thr * t_var['factor'], t_var['unit'])

                    # Determine output variable
                    if t_type == 'A':
                        Adrive = output_thr
                    elif t_type == 't':
                        tstim = output_thr
                    elif t_type == 'DC':
                        DC = output_thr

                    # Define output naming
                    if DC == 1.0:
                        simcode = ASTIM_CW_code.format(neuron.name, a * 1e9, Fdrive * 1e-3,
                                                       Adrive * 1e-3, tstim * 1e3, int_method)
                    else:
                        simcode = ASTIM_PW_code.format(neuron.name, a * 1e9, Fdrive * 1e-3,
                                                       Adrive * 1e-3, tstim * 1e3, PRF,
                                                       DC * 1e2, int_method)

                    # Store dataframe and metadata
                    df = pd.DataFrame({'t': t, 'states': states, 'U': U, 'Z': Z, 'ng': ng, 'Qm': Qm,
                                       'Vm': Vm})
                    for j in range(len(neuron.states_names)):
                        df[neuron.states_names[j]] = channels[j]
                    meta = {'neuron': neuron.name, 'a': solver.a, 'd': solver.d, 'Fdrive': Fdrive,
                            'Adrive': Adrive, 'phi': np.pi, 'tstim': tstim, 'toffset': toffset,
                            'PRF': PRF, 'DC': DC, 'tcomp': tcomp}

                    # Export into to PKL file
                    output_filepath = '{}/{}.pkl'.format(batch_dir, simcode)
                    with open(output_filepath, 'wb') as fh:
                        pickle.dump({'meta': meta, 'data': df}, fh)
                    logger.debug('simulation data exported to "%s"', output_filepath)
                    filepaths.append(output_filepath)

                    # Detect spikes on Qm signal
                    # n_spikes, lat, sr = detectSpikes(t, Qm, SPIKE_MIN_QAMP, SPIKE_MIN_DT)
                    dt = t[1] - t[0]
                    ipeaks, *_ = findPeaks(Qm, SPIKE_MIN_QAMP, int(np.ceil(SPIKE_MIN_DT / dt)),
                                           SPIKE_MIN_QPROM)
                    n_spikes = ipeaks.size
                    lat = t[ipeaks[0]] if n_spikes > 0 else 'N/A'
                    sr = np.mean(1 / np.diff(t[ipeaks])) if n_spikes > 1 else 'N/A'
                    logger.info('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

                    # Export key metrics to log file
                    log = {
                        'A': date_str,
                        'B': daytime_str,
                        'C': neuron.name,
                        'D': solver.a * 1e9,
                        'E': solver.d * 1e6,
                        'F': Fdrive * 1e-3,
                        'G': Adrive * 1e-3,
                        'H': tstim * 1e3,
                        'I': PRF * 1e-3 if DC < 1 else 'N/A',
                        'J': DC,
                        'K': int_method,
                        'L': t.size,
                        'M': round(tcomp, 2),
                        'N': n_spikes,
                        'O': lat * 1e3 if isinstance(lat, float) else 'N/A',
                        'P': sr * 1e-3 if isinstance(sr, float) else 'N/A'
                    }

                    if xlslog(log_filepath, 'Data', log) == 1:
                        logger.info('log exported to "%s"', log_filepath)
                    else:
                        logger.error('log export to "%s" aborted', log_filepath)
                except (Warning, AssertionError) as inst:
                    logger.warning('Integration error: %s. Continue batch? (y/n)', extra={inst})
                    user_str = input()
                    if user_str not in ['y', 'Y']:
                        return filepaths

    return filepaths


def computeSpikeMetrics(filenames):
    ''' Analyze the charge density profile from a list of files and compute for each one of them
        the following spiking metrics:
        - latency (ms)
        - firing rate mean and standard deviation (Hz)
        - spike amplitude mean and standard deviation (nC/cm2)
        - spike width mean and standard deviation (ms)

        :param filenames: list of files to analyze
        :return: a dataframe with the computed metrics
    '''

    # Initialize metrics dictionaries
    keys = [
        'latencies (ms)',
        'mean firing rates (Hz)',
        'std firing rates (Hz)',
        'mean spike amplitudes (nC/cm2)',
        'std spike amplitudes (nC/cm2)',
        'mean spike widths (ms)',
        'std spike widths (ms)'
    ]
    metrics = {k: [] for k in keys}

    # Compute spiking metrics
    for fname in filenames:

        # Load data from file
        logger.debug('loading data from file "{}"'.format(fname))
        with open(fname, 'rb') as fh:
            frame = pickle.load(fh)
        df = frame['data']
        meta = frame['meta']
        tstim = meta['tstim']
        t = df['t'].values
        Qm = df['Qm'].values
        dt = t[1] - t[0]

        # Detect spikes on charge profile
        mpd = int(np.ceil(SPIKE_MIN_DT / dt))
        ispikes, prominences, widths, _ = findPeaks(Qm, SPIKE_MIN_QAMP, mpd, SPIKE_MIN_QPROM)
        widths *= dt

        if ispikes.size > 0:
            # Compute latency
            latency = t[ispikes[0]]

            # Select prior-offset spikes
            ispikes_prior = ispikes[t[ispikes] < tstim]
        else:
            latency = np.nan
            ispikes_prior = np.array([])

        # Compute spikes widths and amplitude
        if ispikes_prior.size > 0:
            widths_prior = widths[:ispikes_prior.size]
            prominences_prior = prominences[:ispikes_prior.size]
        else:
            widths_prior = np.array([np.nan])
            prominences_prior = np.array([np.nan])

        # Compute inter-spike intervals and firing rates
        if ispikes_prior.size > 1:
            ISIs_prior = np.diff(t[ispikes_prior])
            FRs_prior = 1 / ISIs_prior
        else:
            ISIs_prior = np.array([np.nan])
            FRs_prior = np.array([np.nan])

        # Log spiking metrics
        logger.debug('%u spikes detected (%u prior to offset)', ispikes.size, ispikes_prior.size)
        logger.debug('latency: %.2f ms', latency * 1e3)
        logger.debug('average spike width within stimulus: %.2f +/- %.2f ms',
                     np.nanmean(widths_prior) * 1e3, np.nanstd(widths_prior) * 1e3)
        logger.debug('average spike amplitude within stimulus: %.2f +/- %.2f nC/cm2',
                     np.nanmean(prominences_prior) * 1e5, np.nanstd(prominences_prior) * 1e5)
        logger.debug('average ISI within stimulus: %.2f +/- %.2f ms',
                     np.nanmean(ISIs_prior) * 1e3, np.nanstd(ISIs_prior) * 1e3)
        logger.debug('average FR within stimulus: %.2f +/- %.2f Hz',
                     np.nanmean(FRs_prior), np.nanstd(FRs_prior))

        # Complete metrics dictionaries
        metrics['latencies (ms)'].append(latency * 1e3)
        metrics['mean firing rates (Hz)'].append(np.mean(FRs_prior))
        metrics['std firing rates (Hz)'].append(np.std(FRs_prior))
        metrics['mean spike amplitudes (nC/cm2)'].append(np.mean(prominences_prior) * 1e5)
        metrics['std spike amplitudes (nC/cm2)'].append(np.std(prominences_prior) * 1e5)
        metrics['mean spike widths (ms)'].append(np.mean(widths_prior) * 1e3)
        metrics['std spike widths (ms)'].append(np.std(widths_prior) * 1e3)

    # Return dataframe with metrics
    return pd.DataFrame(metrics, columns=metrics.keys())


def getCycleProfiles(a, f, A, Cm0, Qm0, Qm):
    ''' Run a mechanical simulation until periodic stabilization, and compute pressure profiles
        over the last acoustic cycle.

        :param a: in-plane diameter of the sonophore structure within the membrane (m)
        :param f: acoustic drive frequency (Hz)
        :param A: acoustic drive amplitude (Pa)
        :param Cm0: membrane resting capacitance (F/m2)
        :param Qm0: membrane resting charge density (C/m2)
        :param Qm: imposed membrane charge density (C/m2)
        :return: a dataframe with the time, kinematic and pressure profiles over the last cycle.
    '''

    # Create sonophore object
    bls = BilayerSonophore(a, f, Cm0, Qm0)

    # Run default simulation and compute relevant profiles
    logger.info('Running mechanical simulation (a = %sm, f = %sHz, A = %sPa)',
                si_format(a, 1), si_format(f, 1), si_format(A, 1))
    t, y, _ = bls.run(f, A, Qm, Pm_comp_method=PmCompMethod.direct)
    dt = (t[-1] - t[0]) / (t.size - 1)
    Z, ng = y[:, -NPC_FULL:]
    t = t[-NPC_FULL:]
    t -= t[0]

    logger.info('Computing pressure cyclic profiles')
    R = bls.v_curvrad(Z)
    U = np.diff(Z) / dt
    U = np.hstack((U, U[-1]))
    data = {
        't': t,
        'Z': Z,
        'Cm': bls.v_Capct(Z),
        'P_M': bls.v_PMavg(Z, R, bls.surface(Z)),
        'P_Q': bls.Pelec(Z, Qm),
        'P_{VE}': bls.PEtot(Z, R) + bls.PVleaflet(U, R),
        'P_V': bls.PVfluid(U, R),
        'P_G': bls.gasmol2Pa(ng, bls.volume(Z)),
        'P_0': - np.ones(Z.size) * bls.P0
    }
    return pd.DataFrame(data, columns=data.keys())


def runSweepSA(bls, f, A, Qm, params, rel_sweep):
    ''' Run mechanical simulations while varying multiple model parameters around their default value,
        and compute the relative changes in cycle-averaged sonophore membrane potential over the last
        acoustic period upon periodic stabilization.

        :param bls: BilayerSonophore object
        :param f: acoustic drive frequency (Hz)
        :param A: acoustic drive amplitude (Pa)
        :param Qm: imposed membrane charge density (C/m2)
        :param params: list of model parameters to explore
        :param rel_sweep: array of relative parameter changes
        :return: a dataframe with the cycle-averaged sonophore membrane potentials for
        the parameter variations, for each parameter.
    '''

    nsweep = len(rel_sweep)
    logger.info('Starting sensitivity analysis (%u parameters, sweep size = %u)',
                len(params), nsweep)
    t0 = time.time()

    # Run default simulation and compute cycle-averaged membrane potential
    _, y, _ = bls.run(f, A, Qm, Pm_comp_method=PmCompMethod.direct)
    Z = y[0, -NPC_FULL:]
    Cm = bls.v_Capct(Z)  # F/m2
    Vmavg_default = np.mean(Qm / Cm) * 1e3  # mV

    # Create data dictionary for computed output changes
    data = {'relative input change': rel_sweep - 1}

    nsims = len(params) * nsweep
    for j, p in enumerate(params):

        default = getattr(bls, p)
        sweep = rel_sweep * default
        Vmavg = np.empty(nsweep)
        logger.info('Computing system\'s sentitivty to %s (default = %.2e)', p, default)

        for i, val in enumerate(sweep):

            # Re-initialize BLS object with modififed attribute
            setattr(bls, p, val)
            bls.reinit()

            # Run simulation and compute cycle-averaged membrane potential
            _, y, _ = bls.run(f, A, Qm, Pm_comp_method=PmCompMethod.direct)
            Z = y[0, -NPC_FULL:]
            Cm = bls.v_Capct(Z)  # F/m2
            Vmavg[i] = np.mean(Qm / Cm) * 1e3  # mV

            logger.info('simulation %u/%u: %s = %.2e (%+.1f %%) --> |Vm| = %.1f mV (%+.3f %%)',
                        j * nsweep + i + 1, nsims, p, val, (val - default) / default * 1e2,
                        Vmavg[i], (Vmavg[i] - Vmavg_default) / Vmavg_default * 1e2)

        # Fill in data dictionary
        data[p] = Vmavg

        # Set parameter back to default
        setattr(bls, p, default)

    tcomp = time.time() - t0
    logger.info('Sensitivity analysis susccessfully completed in %.0f s', tcomp)

    # return pandas dataframe
    return pd.DataFrame(data, columns=data.keys())


def getActivationMap(root, neuron, a, f, tstim, toffset, PRF, amps, DCs):
    ''' Compute the activation map of a neuron at a given frequency and PRF, by computing
        the spiking metrics of simulation results over a 2D space (amplitude x duty cycle).

        :param root: directory containing the input data files
        :param neuron: neuron name
        :param a: sonophore diameter
        :param f: acoustic drive frequency (Hz)
        :param tstim: duration of US stimulation (s)
        :param toffset: duration of the offset (s)
        :param PRF: pulse repetition frequency (Hz)
        :param amps: vector of acoustic amplitudes (Pa)
        :param DCs: vector of duty cycles (-)
        :return the activation matrix
    '''

    # Initialize activation map
    actmap = np.empty((amps.size, DCs.size))

    # Loop through amplitudes and duty cycles
    nfiles = DCs.size * amps.size
    for i, A in enumerate(amps):
        for j, DC in enumerate(DCs):

            # Define filename
            PW_str = '_PRF{:.2f}Hz_DC{:.2f}%'.format(PRF, DC * 1e2) if DC < 1 else ''
            fname = ('ASTIM_{}_{}W_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms{}_effective.pkl'
                     .format(neuron, 'P' if DC < 1 else 'C', a * 1e9, f * 1e-3, A * 1e-3, tstim * 1e3,
                             PW_str))

            # Extract charge profile from data file
            fpath = os.path.join(root, fname)
            if os.path.isfile(fpath):
                logger.debug('Loading file {}/{}: "{}"'.format(i * amps.size + j + 1, nfiles, fname))
                with open(fpath, 'rb') as fh:
                    frame = pickle.load(fh)
                df = frame['data']
                meta = frame['meta']
                tstim = meta['tstim']
                t = df['t'].values
                Qm = df['Qm'].values
                dt = t[1] - t[0]

                # Detect spikes on charge profile during stimulus
                mpd = int(np.ceil(SPIKE_MIN_DT / dt))
                ispikes, *_ = findPeaks(Qm[t <= tstim], SPIKE_MIN_QAMP, mpd, SPIKE_MIN_QPROM)

                # Compute firing metrics
                if ispikes.size == 0:  # if no spike, assign -1
                    actmap[i, j] = -1
                elif ispikes.size == 1:  # if only 1 spike, assign 0
                    actmap[i, j] = 0
                else:  # if more than 1 spike, assign firing rate
                    FRs = 1 / np.diff(t[ispikes])
                    actmap[i, j] = np.mean(FRs)
            else:
                logger.error('"{}" file not found'.format(fname))
                actmap[i, j] = np.nan

    return actmap


def getMaxMap(key, root, neuron, a, f, tstim, toffset, PRF, amps, DCs, mode='max', cavg=False):
    ''' Compute the max. value map of a neuron's specific variable at a given frequency and PRF
        over a 2D space (amplitude x duty cycle).

        :param key: the variable name to find in the simulations dataframes
        :param root: directory containing the input data files
        :param neuron: neuron name
        :param a: sonophore diameter
        :param f: acoustic drive frequency (Hz)
        :param tstim: duration of US stimulation (s)
        :param toffset: duration of the offset (s)
        :param PRF: pulse repetition frequency (Hz)
        :param amps: vector of acoustic amplitudes (Pa)
        :param DCs: vector of duty cycles (-)
        :param mode: string indicating whether to search for maximum, minimum or absolute maximum
        :return the maximum matrix
    '''

    # Initialize max map
    maxmap = np.empty((amps.size, DCs.size))

    # Loop through amplitudes and duty cycles
    nfiles = DCs.size * amps.size
    for i, A in enumerate(amps):
        for j, DC in enumerate(DCs):

            # Define filename
            PW_str = '_PRF{:.2f}Hz_DC{:.2f}%'.format(PRF, DC * 1e2) if DC < 1 else ''
            fname = ('ASTIM_{}_{}W_{:.0f}nm_{:.0f}kHz_{:.1f}kPa_{:.0f}ms{}_effective.pkl'
                     .format(neuron, 'P' if DC < 1 else 'C', a * 1e9, f * 1e-3, A * 1e-3, tstim * 1e3,
                             PW_str))

            # Extract charge profile from data file
            fpath = os.path.join(root, fname)
            if os.path.isfile(fpath):
                logger.debug('Loading file {}/{}: "{}"'.format(i * amps.size + j + 1, nfiles, fname))
                with open(fpath, 'rb') as fh:
                    frame = pickle.load(fh)
                df = frame['data']
                t = df['t'].values
                if key in df:
                    x = df[key].values
                else:
                    x = eval(key)
                if cavg:
                    x = getCycleAverage(t, x, 1 / PRF)
                if mode == 'min':
                    maxmap[i, j] = x.min()
                elif mode == 'max':
                    maxmap[i, j] = x.max()
                elif mode == 'absmax':
                    maxmap[i, j] = np.abs(x).max()
            else:
                maxmap[i, j] = np.nan

    return maxmap
