# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-25 14:52:53

""" Utility functions used in simulations """

import os
import time
import logging
import pickle
import shutil
import tkinter as tk
from tkinter import filedialog
import numpy as np
from openpyxl import load_workbook

# from . import SolverUS
from .SolverUS import SolverUS
from .SolverElec import SolverElec
from ..constants import *


# Get package logger
logger = logging.getLogger('PointNICE')


# Naming and logging settings
ESTIM_CW_code = 'ESTIM_{}_CW_{:.1f}mA_per_m2_{:.0f}ms'
ESTIM_PW_code = 'ESTIM_{}_PW_{:.1f}mA_per_m2_{:.0f}ms_PRF{:.2f}kHz_DF{:.2f}'
ASTIM_CW_code = 'ASTIM_{}_CW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_{}'
ASTIM_PW_code = 'ASTIM_{}_PW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_PRF{:.2f}kHz_DF{:.2f}_{}'
ESTIM_CW_log = '%s neuron - CW E-STIM simulation %u/%u (A = %.1f mA/m2, t = %.1f ms)'
ESTIM_PW_log = ('%s neuron - PW E-STIM simulation %u/%u (A = %.1f mA/m2, t = %.1f ms, '
                'PRF = %.2f kHz, DF = %.2f)')
ASTIM_CW_log = ('%s neuron - CW A-STIM %s simulation %u/%u (a = %.1f nm, f = %.2f kHz, '
                'A = %.2f kPa, t = %.2f ms')
ASTIM_PW_log = ('%s neuron - PW A-STIM %s simulation %u/%u (a = %.1f nm, f = %.2f kHz, '
                'A = %.2f kPa, t = %.2f ms, PRF = %.2f kHz, DF = %.2f)')

ASTIM_titration_log = '%s neuron - A-STIM titration %u/%u (a = %.1f nm, %s)'
ESTIM_titration_log = '%s neuron - E-STIM titration %u/%u (%s)'

# Define parameters units
ASTIM_params = {
    'f': {'index': 0, 'factor': 1e-3, 'unit': 'kHz'},
    'A': {'index': 1, 'factor': 1e-3, 'unit': 'kPa'},
    't': {'index': 2, 'factor': 1e3, 'unit': 'ms'},
    'PRF': {'index': 4, 'factor': 1e-3, 'unit': 'kHz'},
    'DF': {'index': 5, 'factor': 1e2, 'unit': '%'}
}
ESTIM_params = {
    'A': {'index': 0, 'factor': 1e0, 'unit': 'mA/m2'},
    't': {'index': 1, 'factor': 1e3, 'unit': 'ms'},
    'PRF': {'index': 3, 'factor': 1e-3, 'unit': 'kHz'},
    'DF': {'index': 4, 'factor': 1e2, 'unit': '%'}
}


def checkBatchLog(batch_type):
    ''' Determine batch directory, and add a log file to the directory if it is absent.

        :param batch_type: name of the log file to search for
        :return: 2-tuple with full paths to batch directory and log file
    '''

    # Get batch directory from user
    root = tk.Tk()
    root.withdraw()
    batch_dir = filedialog.askdirectory()
    assert batch_dir, 'No batch directory chosen'

    # Determine log template from batch type
    if batch_type == 'mech':
        logfile = 'log_mech.xlsx'
    elif batch_type == 'A-STIM':
        logfile = 'log_ASTIM.xlsx'
    elif batch_type == 'E-STIM':
        logfile = 'log_ESTIM.xlsx'
    else:
        raise ValueError('Unknown batch type', batch_type)
    this_dir, _ = os.path.split(__file__)
    parent_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
    logsrc = parent_dir + '/templates/' + logfile
    assert os.path.isfile(logsrc), 'template log file "{}" not found'.format(logsrc)

    # Copy template in batch directory if no appropriate log file
    logdst = batch_dir + '/' + logfile
    if not os.path.isfile(logdst):
        shutil.copy2(logsrc, logdst)

    return (batch_dir, logdst)


def createSimQueue(amps, durations, offsets, PRFs, DFs):
    ''' Create a serialized 2D array of all parameter combinations for a series of individual
        parameter sweeps, while avoiding repetition of CW protocols for a given PRF sweep.

        :param amps: list (or 1D-array) of acoustic amplitudes
        :param durations: list (or 1D-array) of stimulus durations
        :param offsets: list (or 1D-array) of stimulus offsets (paired with durations array)
        :param PRFs: list (or 1D-array) of pulse-repetition frequencies
        :param DFs: list (or 1D-array) of duty cycle values
        :return: 2D-array with (amplitude, duration, offset, PRF, DF) for each stimulation protocol
    '''

    # Convert input to 1D-arrays
    amps = np.array(amps)
    durations = np.array(durations)
    offsets = np.array(offsets)
    PRFs = np.array(PRFs)
    DFs = np.array(DFs)

    # Create index arrays
    iamps = range(len(amps))
    idurs = range(len(durations))

    # Create empty output matrix
    queue = np.empty((1, 5))

    # Continuous protocols
    if 1.0 in DFs:
        nCW = len(amps) * len(durations)
        arr1 = np.ones(nCW)
        iCW_queue = np.array(np.meshgrid(iamps, idurs)).T.reshape(nCW, 2)
        CW_queue = np.hstack((amps[iCW_queue[:, 0]], durations[iCW_queue[:, 1]],
                              offsets[iCW_queue[:, 1]], PRFs.min() * arr1, arr1))
        queue = np.vstack((queue, CW_queue))


    # Pulsed protocols
    if np.any(DFs != 1.0):
        pulsed_DFs = DFs[DFs != 1.0]
        iPRFs = range(len(PRFs))
        ipulsed_DFs = range(len(pulsed_DFs))
        nPW = len(amps) * len(durations) * len(PRFs) * len(pulsed_DFs)
        iPW_queue = np.array(np.meshgrid(iamps, idurs, iPRFs, ipulsed_DFs)).T.reshape(nPW, 4)
        PW_queue = np.hstack((amps[iPW_queue[:, 0]], durations[iPW_queue[:, 1]],
                              offsets[iPW_queue[:, 1]], PRFs[iPW_queue[:, 2]],
                              pulsed_DFs[iPW_queue[:, 3]]))
        queue = np.vstack((queue, PW_queue))

    # Return
    return queue[1:, :]


def xlslog(filename, sheetname, data):
    """ Append log data on a new row to specific sheet of excel workbook.

        :param filename: absolute or relative path to the Excel workbook
        :param sheetname: name of the Excel spreadsheet to which data is appended
        :param data: data structure to be added to specific columns on a new row
        :return: boolean indicating success (1) or failure (0) of operation
    """

    try:
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
        return 1
    except PermissionError:
        # If file cannot be accessed for writing because already opened
        logger.error('Cannot write to "%s". Close the file and type "Y"', filename)
        user_str = input()
        if user_str in ['y', 'Y']:
            return xlslog(filename, sheetname, data)
        else:
            return 0


def detectPeaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                kpsh=False, valley=False, ax=None):
    """ Detect peaks in data based on their amplitude and inter-peak distance. """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
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

    return ind


def detectPeaksTime(t, y, mph, mtd):
    """ Extension of the detectPeaks function to detect peaks in data based on their
        amplitude and time difference, with a non-uniform time vector.

        :param t: time vector (not necessarily uniform)
        :param y: signal
        :param mph: minimal peak height
        :param mtd: minimal time difference
        :return: array of peak indexes
    """

    # Detect peaks on signal with no restriction on inter-peak distance
    raw_indexes = detectPeaks(y, mph, mpd=1)

    if raw_indexes.size > 0:

        # Filter relevant peaks with temporal distance
        n_raw = raw_indexes.size
        filtered_indexes = np.array([raw_indexes[0]])
        for i in range(1, n_raw):
            i1 = filtered_indexes[-1]
            i2 = raw_indexes[i]
            if t[i2] - t[i1] < mtd:
                if y[i2] > y[i1]:
                    filtered_indexes[-1] = i2
            else:
                filtered_indexes = np.append(filtered_indexes, i2)

        # Return peak indexes
        return filtered_indexes
    else:
        return None


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
    if i_spikes is not None:
        latency = t[i_spikes[0]]  # s
        n_spikes = i_spikes.size
        if n_spikes > 1:
            first_to_last_spike = t[i_spikes[-1]] - t[i_spikes[0]]  # s
            spike_rate = n_spikes / first_to_last_spike  # spikes/s
        else:
            spike_rate = 'N/A'
    else:
        latency = 'N/A'
        spike_rate = 'N/A'
        n_spikes = 0
    return (n_spikes, latency, spike_rate)



def runEStimBatch(batch_dir, log_filepath, neurons, stim_params):
    ''' Run batch E-STIM simulations of the system for various neuron types and
        stimulation parameters.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param neurons: array of channel mechanisms
        :param stim_params: dictionary containing sweeps for all stimulation parameters
    '''

    logger.info("Starting E-STIM simulation batch")

    # Generate simulations queue
    sim_queue = createSimQueue(stim_params['amps'], stim_params['durations'],
                               stim_params['offsets'], stim_params['PRFs'], stim_params['DFs'])
    nqueue = sim_queue.shape[0]

    # Initialize solver
    solver = SolverElec()

    # Run simulations
    nsims = len(neurons) * nqueue
    simcount = 0
    filepaths = []
    for ch_mech in neurons:
        try:
            for i in range(nqueue):
                simcount += 1
                Astim, tstim, toffset, PRF, DF = sim_queue[i, :]

                # Get date and time info
                date_str = time.strftime("%Y.%m.%d")
                daytime_str = time.strftime("%H:%M:%S")

                # Log and define naming
                if DF == 1.0:
                    logger.info(ESTIM_CW_log, ch_mech.name, simcount, nsims, Astim, tstim * 1e3)
                    simcode = ESTIM_CW_code.format(ch_mech.name, Astim, tstim * 1e3)
                else:
                    logger.info(ESTIM_PW_log, ch_mech.name, simcount, nsims, Astim, tstim * 1e3,
                                PRF * 1e-3, DF)
                    simcode = ESTIM_PW_code.format(ch_mech.name, Astim, tstim * 1e3, PRF * 1e-3, DF)

                # Run simulation
                tstart = time.time()
                (t, y, states) = solver.run(ch_mech, Astim, tstim, toffset, PRF, DF)
                Vm, *channels = y
                tcomp = time.time() - tstart
                logger.info('completed in %.2f seconds', tcomp)

                # Store data in dictionary
                data = {
                    'Astim': Astim,
                    'tstim': tstim,
                    'toffset': toffset,
                    'PRF': PRF,
                    'DF': DF,
                    't': t,
                    'states': states,
                    'Vm': Vm
                }
                for j in range(len(ch_mech.states_names)):
                    data[ch_mech.states_names[j]] = channels[j]

                # Export data to PKL file
                output_filepath = batch_dir + '/' + simcode + ".pkl"
                with open(output_filepath, 'wb') as fh:
                    pickle.dump(data, fh)
                logger.info('simulation data exported to "%s"', output_filepath)
                filepaths.append(output_filepath)

                # Detect spikes on Vm signal
                n_spikes, lat, sr = detectSpikes(t, Vm, SPIKE_MIN_VAMP, SPIKE_MIN_DT)
                logger.info('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

                # Export key metrics to log file
                log = {
                    'A': date_str,
                    'B': daytime_str,
                    'C': ch_mech.name,
                    'D': Astim,
                    'E': tstim * 1e3,
                    'F': PRF * 1e-3 if DF < 1 else 'N/A',
                    'G': DF,
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

        except AssertionError as err:
            logger.error(err)

    return filepaths




def runAStimBatch(batch_dir, log_filepath, neurons, bls_params, geom, stim_params,
                  sim_type='effective'):
    ''' Run batch simulations of the system for various neuron types, sonophore and
        stimulation parameters.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param neurons: array of channel mechanisms
        :param bls_params: BLS biomechanical and biophysical parameters dictionary
        :param geom: BLS geometric constants dictionary
        :param stim_params: dictionary containing sweeps for all stimulation parameters
        :param sim_type: selected integration method
    '''


    logger.info("Starting A-STIM simulation batch")

    # Unpack geometrical parameters
    a = geom['a']
    d = geom['d']

    # Generate simulations queue
    sim_queue = createSimQueue(stim_params['amps'], stim_params['durations'],
                               stim_params['offsets'], stim_params['PRFs'], stim_params['DFs'])
    nqueue = sim_queue.shape[0]

    # Run simulations
    nsims = len(neurons) * len(stim_params['freqs']) * nqueue
    simcount = 0
    filepaths = []
    for ch_mech in neurons:
        for Fdrive in stim_params['freqs']:
            try:
                # Create SolverUS instance (modulus of embedding tissue depends on frequency)
                solver = SolverUS(geom, bls_params, ch_mech, Fdrive)

                for i in range(nqueue):
                    simcount += 1
                    Adrive, tstim, toffset, PRF, DF = sim_queue[i, :]

                    # Get date and time info
                    date_str = time.strftime("%Y.%m.%d")
                    daytime_str = time.strftime("%H:%M:%S")

                    # Log and define naming
                    if DF == 1.0:
                        logger.info(ASTIM_CW_log, ch_mech.name, sim_type, simcount, nsims,
                                    a * 1e9, Fdrive * 1e-3, Adrive * 1e-3, tstim * 1e3)
                        simcode = ASTIM_CW_code.format(ch_mech.name, a * 1e9, Fdrive * 1e-3,
                                                       Adrive * 1e-3, tstim * 1e3, sim_type)
                    else:
                        logger.info(ASTIM_PW_log, ch_mech.name, sim_type, simcount, nsims, a * 1e9,
                                    Fdrive * 1e-3, Adrive * 1e-3, tstim * 1e3, PRF * 1e-3, DF)
                        simcode = ASTIM_PW_code.format(ch_mech.name, a * 1e9, Fdrive * 1e-3,
                                                       Adrive * 1e-3, tstim * 1e3, PRF * 1e-3,
                                                       DF, sim_type)

                    # Run simulation
                    tstart = time.time()
                    (t, y, states) = solver.run(ch_mech, Fdrive, Adrive, tstim, toffset, PRF, DF,
                                                sim_type)


                    Z, ng, Qm, *channels = y
                    U = np.insert(np.diff(Z) / np.diff(t), 0, 0.0)
                    tcomp = time.time() - tstart
                    logger.info('completed in %.2f seconds', tcomp)

                    # Store data in dictionary
                    bls_params['biophys']['Qm0'] = solver.Qm0
                    data = {
                        'a': a,
                        'd': d,
                        'params': bls_params,
                        'Fdrive': Fdrive,
                        'Adrive': Adrive,
                        'phi': np.pi,
                        'tstim': tstim,
                        'toffset': toffset,
                        'PRF': PRF,
                        'DF': DF,
                        't': t,
                        'states': states,
                        'U': U,
                        'Z': Z,
                        'ng': ng,
                        'Qm': Qm,
                        'Vm': Qm * 1e3 / np.array([solver.Capct(ZZ) for ZZ in Z])
                    }
                    for j in range(len(ch_mech.states_names)):
                        data[ch_mech.states_names[j]] = channels[j]

                    # Export data to PKL file
                    output_filepath = batch_dir + '/' + simcode + ".pkl"
                    with open(output_filepath, 'wb') as fh:
                        pickle.dump(data, fh)
                    logger.info('simulation data exported to "%s"', output_filepath)
                    filepaths.append(output_filepath)

                    # Detect spikes on Qm signal
                    n_spikes, lat, sr = detectSpikes(t, Qm, SPIKE_MIN_QAMP, SPIKE_MIN_DT)
                    logger.info('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

                    # Export key metrics to log file
                    log = {
                        'A': date_str,
                        'B': daytime_str,
                        'C': ch_mech.name,
                        'D': a * 1e9,
                        'E': d * 1e6,
                        'F': Fdrive * 1e-3,
                        'G': Adrive * 1e-3,
                        'H': tstim * 1e3,
                        'I': PRF * 1e-3 if DF < 1 else 'N/A',
                        'J': DF,
                        'K': sim_type,
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

            except AssertionError as err:
                logger.error(err)

    return filepaths


def titrateEStim(solver, ch_mech, Astim, tstim, toffset, PRF=1.5e3, DF=1.0):
    """ Use a dichotomic recursive search to determine the threshold value of a specific
        electric stimulation parameter needed to obtain neural excitation, keeping all other
        parameters fixed. The titration parameter can be stimulation amplitude, duration or
        any variable for which the number of spikes is a monotonically increasing function.

        This function is called recursively until an accurate threshold is found.

        :param solver: solver instance
        :param ch_mech: channels mechanism object
        :param Astim: injected current density amplitude (mA/m2)
        :param tstim: duration of US stimulation (s)
        :param toffset: duration of the offset (s)
        :param PRF: pulse repetition frequency (Hz)
        :param DF: pulse duty factor (-)
        :return: 5-tuple with the determined amplitude threshold, time profile,
                 solution matrix, state vector and response latency
    """

    # Determine titration type
    if isinstance(Astim, tuple):
        t_type = 'A'
        interval = Astim
        thr = TITRATION_ESTIM_DA_MAX
    elif isinstance(tstim, tuple):
        t_type = 't'
        interval = tstim
        thr = TITRATION_DT_THR
    elif isinstance(DF, tuple):
        t_type = 'DF'
        interval = DF
        thr = TITRATION_DDF_THR
    else:
        logger.error('Invalid titration type')
        return 0.

    t_var = ESTIM_params[t_type]

    # Check amplitude interval and define current value
    assert interval[0] < interval[1], '{} interval must be defined as (lb, ub)'.format(t_type)
    value = (interval[0] + interval[1]) / 2

    # Define stimulation parameters
    if t_type == 'A':
        stim_params = [value, tstim, toffset, PRF, DF]
    elif t_type == 't':
        stim_params = [Astim, value, toffset, PRF, DF]
    elif t_type == 'DF':
        stim_params = [Astim, tstim, toffset, PRF, value]

    # Run simulation and detect spikes
    (t, y, states) = solver.run(ch_mech, *stim_params)
    n_spikes, latency, _ = detectSpikes(t, y[0, :], SPIKE_MIN_VAMP, SPIKE_MIN_DT)
    logger.info('%.2f %s ---> %u spike%s detected', value * t_var['factor'], t_var['unit'],
                n_spikes, "s" if n_spikes > 1 else "")

    # If accurate threshold is found, return simulation results
    if (interval[1] - interval[0]) <= thr and n_spikes == 1:
        return (value, t, y, states, latency)

    # Otherwise, refine titration interval and iterate recursively
    else:
        if n_spikes == 0:
            new_interval = (value, interval[1])
        else:
            new_interval = (interval[0], value)

        stim_params[t_var['index']] = new_interval
        return titrateEStim(solver, ch_mech, *stim_params)


def titrateAStim(solver, ch_mech, Fdrive, Adrive, tstim, toffset, PRF=1.5e3, DF=1.0,
                 sim_type='effective'):
    """ Use a dichotomic recursive search to determine the threshold value of a specific
        acoustic stimulation parameter needed to obtain neural excitation, keeping all other
        parameters fixed. The titration parameter can be stimulation amplitude, duration or
        any variable for which the number of spikes is a monotonically increasing function.

        This function is called recursively until an accurate threshold is found.

        :param solver: solver instance
        :param ch_mech: channels mechanism object
        :param Fdrive: acoustic drive frequency (Hz)
        :param Adrive: acoustic drive amplitude (Pa)
        :param tstim: duration of US stimulation (s)
        :param toffset: duration of the offset (s)
        :param PRF: pulse repetition frequency (Hz)
        :param DF: pulse duty factor (-)
        :param sim_type: selected integration method
        :return: 5-tuple with the determined amplitude threshold, time profile,
                 solution matrix, state vector and response latency
    """

    # Determine titration type
    if isinstance(Adrive, tuple):
        t_type = 'A'
        interval = Adrive
        thr = TITRATION_ASTIM_DA_MAX
    elif isinstance(tstim, tuple):
        t_type = 't'
        interval = tstim
        thr = TITRATION_DT_THR
    elif isinstance(DF, tuple):
        t_type = 'DF'
        interval = DF
        thr = TITRATION_DDF_THR
    else:
        logger.error('Invalid titration type')
        return 0.

    t_var = ASTIM_params[t_type]

    # Check amplitude interval and define current value
    assert interval[0] < interval[1], '{} interval must be defined as (lb, ub)'.format(t_type)
    value = (interval[0] + interval[1]) / 2

    # Define stimulation parameters
    if t_type == 'A':
        stim_params = [Fdrive, value, tstim, toffset, PRF, DF]
    elif t_type == 't':
        stim_params = [Fdrive, Adrive, value, toffset, PRF, DF]
    elif t_type == 'DF':
        stim_params = [Fdrive, Adrive, tstim, toffset, PRF, value]

    # Run simulation and detect spikes
    (t, y, states) = solver.run(ch_mech, *stim_params, sim_type)
    n_spikes, latency, _ = detectSpikes(t, y[2, :], SPIKE_MIN_QAMP, SPIKE_MIN_DT)
    logger.info('%.2f %s ---> %u spike%s detected', value * t_var['factor'], t_var['unit'],
                n_spikes, "s" if n_spikes > 1 else "")

    # If accurate threshold is found, return simulation results
    if (interval[1] - interval[0]) <= thr and n_spikes == 1:
        return (value, t, y, states, latency)

    # Otherwise, refine titration interval and iterate recursively
    else:
        if n_spikes == 0:
            new_interval = (value, interval[1])
        else:
            new_interval = (interval[0], value)

        stim_params[t_var['index']] = new_interval
        return titrateAStim(solver, ch_mech, *stim_params, sim_type)


def titrateAStimBatch(batch_dir, log_filepath, neurons, bls_params, geom, stim_params):
    ''' Run batch acoustic titrations of the system for various neuron types, sonophore and
        stimulation parameters, to determine the threshold of a specific stimulus parameter
        for neural excitation.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param neurons: array of channel mechanisms
        :param bls_params: BLS biomechanical and biophysical parameters dictionary
        :param geom: BLS geometric constants dictionary
        :param stim_params: dictionary containing sweeps for all stimulation parameters
    '''

    logger.info("Starting A-STIM titration batch")

    # Unpack geometrical parameters
    a = geom['a']
    d = geom['d']

    # Define default parameters
    sim_type = 'effective'
    offset = 30e-3

    # Determine titration parameter and titrations list
    if 'durations' not in stim_params:
        t_type = 't'
        sim_queue = createSimQueue(stim_params['amps'], [None], [offset],
                                   stim_params['PRFs'], stim_params['DFs'])
        # sim_queue = np.delete(sim_queue, 1, axis=1)
    elif 'amps' not in stim_params:
        t_type = 'A'
        sim_queue = createSimQueue([None], stim_params['durations'],
                                   [offset] * len(stim_params['durations']),
                                   stim_params['PRFs'], stim_params['DFs'])
    elif 'DF' not in stim_params:
        t_type = 'DF'
        sim_queue = createSimQueue(stim_params['amps'], stim_params['durations'],
                                   [offset] * len(stim_params['durations']),
                                   stim_params['PRFs'], [None])

    nqueue = sim_queue.shape[0]
    t_var = ASTIM_params[t_type]

    # Run titrations
    nsims = len(neurons) * len(stim_params['freqs']) * nqueue
    simcount = 0
    filepaths = []
    for ch_mech in neurons:
        for Fdrive in stim_params['freqs']:
            try:
                # Create SolverUS instance (modulus of embedding tissue depends on frequency)
                solver = SolverUS(geom, bls_params, ch_mech, Fdrive)

                for i in range(nqueue):
                    simcount += 1

                    # Extract parameters
                    Adrive, tstim, toffset, PRF, DF = sim_queue[i, :]
                    if Adrive is None:
                        Adrive = (0., 2 * TITRATION_ASTIM_A_MAX)
                    elif tstim is None:
                        tstim = (0., 2 * TITRATION_T_MAX)
                    elif DF is None:
                        DF = (0., 2 * TITRATION_DF_MAX)
                    curr_params = [Fdrive, Adrive, tstim, PRF, DF]

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
                    logger.info(ASTIM_titration_log, ch_mech.name, simcount, nsims, a * 1e9,
                                log_str)

                    # Run titration
                    tstart = time.time()

                    (output_thr, t, y, states, lat) = titrateAStim(solver, ch_mech, Fdrive, Adrive,
                                                                   tstim, toffset, PRF, DF)

                    Z, ng, Qm, *channels = y
                    U = np.insert(np.diff(Z) / np.diff(t), 0, 0.0)
                    tcomp = time.time() - tstart
                    logger.info('completed in %.2f s, threshold = %.2f %s', tcomp,
                                output_thr * t_var['factor'], t_var['unit'])

                    # Determine output variable
                    if t_type == 'A':
                        Adrive = output_thr
                    elif t_type == 't':
                        tstim = output_thr
                    elif t_type == 'DF':
                        DF = output_thr

                    # Define output naming
                    if DF == 1.0:
                        simcode = ASTIM_CW_code.format(ch_mech.name, a * 1e9, Fdrive * 1e-3,
                                                       Adrive * 1e-3, tstim * 1e3, sim_type)
                    else:
                        simcode = ASTIM_PW_code.format(ch_mech.name, a * 1e9, Fdrive * 1e-3,
                                                       Adrive * 1e-3, tstim * 1e3, PRF * 1e-3,
                                                       DF, sim_type)

                    # Store data in dictionary
                    bls_params['biophys']['Qm0'] = solver.Qm0
                    data = {
                        'a': a,
                        'd': d,
                        'params': bls_params,
                        'Fdrive': Fdrive,
                        'Adrive': Adrive,
                        'phi': np.pi,
                        'tstim': tstim,
                        'toffset': toffset,
                        'PRF': PRF,
                        'DF': DF,
                        't': t,
                        'states': states,
                        'U': U,
                        'Z': Z,
                        'ng': ng,
                        'Qm': Qm,
                        'Vm': Qm * 1e3 / np.array([solver.Capct(ZZ) for ZZ in Z])
                    }
                    for j in range(len(ch_mech.states_names)):
                        data[ch_mech.states_names[j]] = channels[j]

                    # Export data to PKL file
                    output_filepath = batch_dir + '/' + simcode + ".pkl"
                    with open(output_filepath, 'wb') as fh:
                        pickle.dump(data, fh)
                    logger.info('simulation data exported to "%s"', output_filepath)
                    filepaths.append(output_filepath)

                    # Detect spikes on Qm signal
                    n_spikes, lat, sr = detectSpikes(t, Qm, SPIKE_MIN_QAMP, SPIKE_MIN_DT)
                    logger.info('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

                    # Export key metrics to log file
                    log = {
                        'A': date_str,
                        'B': daytime_str,
                        'C': ch_mech.name,
                        'D': a * 1e9,
                        'E': d * 1e6,
                        'F': Fdrive * 1e-3,
                        'G': Adrive * 1e-3,
                        'H': tstim * 1e3,
                        'I': PRF * 1e-3 if DF < 1 else 'N/A',
                        'J': DF,
                        'K': sim_type,
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

            except AssertionError as err:
                logger.error(err)

    return filepaths


def titrateEStimBatch(batch_dir, log_filepath, neurons, stim_params):
    ''' Run batch electrical titrations of the system for various neuron types and
        stimulation parameters, to determine the threshold of a specific stimulus parameter
        for neural excitation.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param neurons: array of channel mechanisms
        :param stim_params: dictionary containing sweeps for all stimulation parameters
    '''

    logger.info("Starting E-STIM titration batch")

    # Define default parameters
    offset = 30e-3

    # Determine titration parameter and titrations list
    if 'durations' not in stim_params:
        t_type = 't'
        sim_queue = createSimQueue(stim_params['amps'], [None], [offset],
                                   stim_params['PRFs'], stim_params['DFs'])
    elif 'amps' not in stim_params:
        t_type = 'A'
        sim_queue = createSimQueue([None], stim_params['durations'],
                                   [offset] * len(stim_params['durations']),
                                   stim_params['PRFs'], stim_params['DFs'])
    elif 'DF' not in stim_params:
        t_type = 'DF'
        sim_queue = createSimQueue(stim_params['amps'], stim_params['durations'],
                                   [offset] * len(stim_params['durations']),
                                   stim_params['PRFs'], [None])

    nqueue = sim_queue.shape[0]
    t_var = ESTIM_params[t_type]

    # Create SolverElec instance
    solver = SolverElec()

    # Run titrations
    nsims = len(neurons) * nqueue
    simcount = 0
    filepaths = []
    for ch_mech in neurons:
        try:

            for i in range(nqueue):
                simcount += 1

                # Extract parameters
                Astim, tstim, toffset, PRF, DF = sim_queue[i, :]
                if Astim is None:
                    Astim = (0., 2 * TITRATION_ESTIM_A_MAX)
                elif tstim is None:
                    tstim = (0., 2 * TITRATION_T_MAX)
                elif DF is None:
                    DF = (0., 2 * TITRATION_DF_MAX)
                curr_params = [Astim, tstim, PRF, DF]

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
                logger.info(ESTIM_titration_log, ch_mech.name, simcount, nsims, log_str)

                # Run titration
                tstart = time.time()

                (output_thr, t, y, states, lat) = titrateEStim(solver, ch_mech, Astim,
                                                               tstim, toffset, PRF, DF)

                Vm, *channels = y
                tcomp = time.time() - tstart
                logger.info('completed in %.2f s, threshold = %.2f %s', tcomp,
                            output_thr * t_var['factor'], t_var['unit'])

                # Determine output variable
                if t_type == 'A':
                    Astim = output_thr
                elif t_type == 't':
                    tstim = output_thr
                elif t_type == 'DF':
                    DF = output_thr

                # Define output naming
                if DF == 1.0:
                    simcode = ESTIM_CW_code.format(ch_mech.name, Astim, tstim * 1e3)
                else:
                    simcode = ESTIM_PW_code.format(ch_mech.name, Astim, tstim * 1e3,
                                                   PRF * 1e-3, DF)

                # Store data in dictionary
                data = {
                    'Astim': Astim,
                    'tstim': tstim,
                    'toffset': toffset,
                    'PRF': PRF,
                    'DF': DF,
                    't': t,
                    'states': states,
                    'Vm': Vm
                }
                for j in range(len(ch_mech.states_names)):
                    data[ch_mech.states_names[j]] = channels[j]

                # Export data to PKL file
                output_filepath = batch_dir + '/' + simcode + ".pkl"
                with open(output_filepath, 'wb') as fh:
                    pickle.dump(data, fh)
                logger.info('simulation data exported to "%s"', output_filepath)
                filepaths.append(output_filepath)

                # Detect spikes on Qm signal
                n_spikes, lat, sr = detectSpikes(t, Vm, SPIKE_MIN_VAMP, SPIKE_MIN_DT)
                logger.info('%u spike%s detected', n_spikes, "s" if n_spikes > 1 else "")

                # Export key metrics to log file
                log = {
                    'A': date_str,
                    'B': daytime_str,
                    'C': ch_mech.name,
                    'D': Astim,
                    'E': tstim * 1e3,
                    'F': PRF * 1e-3 if DF < 1 else 'N/A',
                    'G': DF,
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

        except AssertionError as err:
            logger.error(err)

    return filepaths
