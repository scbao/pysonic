# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-23 16:23:51

""" Utility functions used in simulations """

import time
import logging
import pickle
import numpy as np
from openpyxl import load_workbook

# from . import SolverUS
from .SolverUS import SolverUS
from ..constants import *
from ..utils import detectSpikes


# Get package logger
logger = logging.getLogger('PointNICE')


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


def runSimBatch(batch_dir, log_filepath, neurons, bls_params, geom, stim_params, sim_type):
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

    # Define naming and logging settings
    sim_str_CW = 'sim_{}_CW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_{}'
    sim_str_PW = 'sim_{}_PW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_PRF{:.2f}kHz_DF{:.2f}_{}'
    CW_log = ('%s neuron - CW %s simulation %u/%u (a = %.1f nm, f = %.2f kHz, A = %.2f kPa, '
              't = %.1f ms)')
    PW_log = ('%s neuron - PW %s simulation %u/%u (a = %.1f nm, f = %.2f kHz, A = %.2f kPa, '
              ' t = %.1f ms, PRF = %.2f kHz, DF = %.2f)')


    logger.info("Starting NICE simulation batch")

    a = geom['a']
    d = geom['d']

    # Generate simulations queue
    sim_queue = createSimQueue(stim_params['amps'], stim_params['durations'],
                               stim_params['offsets'], stim_params['PRFs'], stim_params['DFs'])
    nqueue = sim_queue.shape[0]

    # Run simulations
    simcount = 0
    nsims = len(neurons) * len(stim_params['freqs']) * nqueue
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
                        logger.info(CW_log, ch_mech.name, sim_type, simcount, nsims,
                                    a * 1e9, Fdrive * 1e-3, Adrive * 1e-3, tstim * 1e3)
                        simcode = sim_str_CW.format(ch_mech.name, a * 1e9, Fdrive * 1e-3,
                                                    Adrive * 1e-3, tstim * 1e3, sim_type)
                    else:
                        logger.info(PW_log, ch_mech.name, sim_type, simcount, nsims, a * 1e9,
                                    Fdrive * 1e-3, Adrive * 1e-3, tstim * 1e3, PRF * 1e-3, DF)
                        simcode = sim_str_PW.format(ch_mech.name, a * 1e9, Fdrive * 1e-3,
                                                    Adrive * 1e-3, tstim * 1e3, PRF * 1e-3,
                                                    DF, sim_type)

                    # Run simulation
                    tstart = time.time()
                    (t, y, states) = solver.runSim(ch_mech, Fdrive, Adrive, tstim, toffset,
                                                   PRF, DF, sim_type)


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
                        'Qm': Qm
                    }
                    for j in range(len(ch_mech.states_names)):
                        data[ch_mech.states_names[j]] = channels[j]

                    # Export data to PKL file
                    datafile_name = batch_dir + '/' + simcode + ".pkl"
                    with open(datafile_name, 'wb') as fh:
                        pickle.dump(data, fh)
                    logger.info('simulation data exported to "%s"', datafile_name)

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



def runTitrationBatch(batch_dir, log_filepath, neurons, bls_params, geom, stim_params):
    ''' Run batch titrations of the system for various neuron types, sonophore and
        stimulation parameters, to determine the threshold of a specific stimulus parameter
        for neural excitation.

        :param batch_dir: full path to output directory of batch
        :param log_filepath: full path log file of batch
        :param neurons: array of channel mechanisms
        :param bls_params: BLS biomechanical and biophysical parameters dictionary
        :param geom: BLS geometric constants dictionary
        :param stim_params: dictionary containing sweeps for all stimulation parameters
    '''

    # Define naming and logging settings
    sim_str_CW = 'sim_{}_CW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_{}'
    sim_str_PW = 'sim_{}_PW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_PRF{:.2f}kHz_DF{:.2f}_{}'
    CW_log = ('%s neuron - CW titration %u/%u (a = %.1f nm, f = %.2f kHz, %s = %.2f %s')
    PW_log = ('%s neuron - PW titration %u/%u (a = %.1f nm, f = %.2f kHz, %s = %.2f %s, '
              'PRF = %.2f kHz, DF = %.2f)')

    logger.info("Starting NICE titration batch")

    # Unpack geometrical parameters
    a = geom['a']
    d = geom['d']

    # Define default parameters
    sim_type = 'effective'
    offset = 30e-3

    # Determine titration parameter (x) and titrations list
    A = {'name': 'A', 'factor': 1e-3, 'unit': 'kPa'}
    t = {'name': 't', 'factor': 1e3, 'unit': 'ms'}
    if 'durations' not in stim_params:
        varin = A
        varout = t
        titr_type = 'duration'
        sim_queue = createSimQueue(stim_params['amps'], [0.], [offset],
                                   stim_params['PRFs'], stim_params['DFs'])
        sim_queue = np.delete(sim_queue, 1, axis=1)

    elif 'amps' not in stim_params:
        varin = t
        varout = A
        titr_type = 'amplitude'
        sim_queue = createSimQueue([0.], stim_params['durations'],
                                   [offset] * len(stim_params['durations']),
                                   stim_params['PRFs'], stim_params['DFs'])
        sim_queue = np.delete(sim_queue, 0, axis=1)

    nqueue = sim_queue.shape[0]

    # Run titrations
    simcount = 0
    nsims = len(neurons) * len(stim_params['freqs']) * nqueue
    for ch_mech in neurons:
        for Fdrive in stim_params['freqs']:
            try:
                # Create SolverUS instance (modulus of embedding tissue depends on frequency)
                solver = SolverUS(geom, bls_params, ch_mech, Fdrive)

                for i in range(nqueue):
                    simcount += 1
                    input_val, toffset, PRF, DF = sim_queue[i, :]

                    # Get date and time info
                    date_str = time.strftime("%Y.%m.%d")
                    daytime_str = time.strftime("%H:%M:%S")

                    # Log and define naming
                    if DF == 1.0:
                        logger.info(CW_log, ch_mech.name, simcount, nsims, a * 1e9, Fdrive * 1e-3,
                                    varin['name'], input_val * varin['factor'], varin['unit'])
                    else:
                        logger.info(PW_log, ch_mech.name, simcount, nsims, a * 1e9, Fdrive * 1e-3,
                                    varin['name'], input_val * varin['factor'], varin['unit'],
                                    PRF * 1e-3, DF)

                    # Run titration
                    tstart = time.time()
                    (output_thr, t, y, states, lat) = solver.titrate(ch_mech, Fdrive, input_val,
                                                                     toffset, PRF, DF, titr_type)
                    Z, ng, Qm, *channels = y
                    U = np.insert(np.diff(Z) / np.diff(t), 0, 0.0)
                    tcomp = time.time() - tstart
                    logger.info('completed in %.2f s, threshold = %.2f %s', tcomp,
                                output_thr * varout['factor'], varout['unit'])

                    # Sort input and output as amplitude and duration
                    if titr_type == 'amplitude':
                        Adrive = output_thr
                        tstim = input_val
                    elif titr_type == 'duration':
                        tstim = output_thr
                        Adrive = input_val

                    # Define output naming
                    if DF == 1.0:
                        sim_str_CW = 'sim_{}_CW_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_{}'
                        simcode = sim_str_CW.format(ch_mech.name, a * 1e9, Fdrive * 1e-3,
                                                    Adrive * 1e-3, tstim * 1e3, sim_type)
                    else:
                        simcode = sim_str_PW.format(ch_mech.name, a * 1e9, Fdrive * 1e-3,
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
                        'Qm': Qm
                    }
                    for j in range(len(ch_mech.states_names)):
                        data[ch_mech.states_names[j]] = channels[j]

                    # Export data to PKL file
                    datafile_name = batch_dir + '/' + simcode + ".pkl"
                    with open(datafile_name, 'wb') as fh:
                        pickle.dump(data, fh)
                    logger.info('simulation data exported to "%s"', datafile_name)

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
