# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-03-21 18:46:09
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-03-29 18:57:42

''' Module of utilitary functions calling MATLAB routines internally. '''

import os
import logging
import pickle
import numpy as np
import matlab.engine
from .utils import InputError

# Get package logger
logger = logging.getLogger('PointNICE')


def findPeaksMATLAB(y, mpp, mpd, mph):
    ''' Find peaks in a signal by calling internally the matlab "finpeaks" function.

        :param y: input signal
        :param mpp: minimal peak prominence (signal units)
        :param mpd: minimal distance between consecutive peaks (number of indexes)
        :param mph: minimal peak height (signal units)
        :return: 4-tuple with indexes, values, witdths and prominences of peaks
    '''

    # Start MATLAB matlab_engine if not existing
    if 'matlab_eng' not in globals():
        global matlab_eng
        logger.info('starting matlab engine')
        matlab_eng = matlab.engine.start_matlab()
        logger.info('matlab engine started')

    _, ipeaks, widths, proms = np.array(matlab_eng.findpeaks(matlab.double(y.tolist()),
                                                             'MinPeakProminence', mpp,
                                                             'MinPeakDistance', mpd,
                                                             'MinPeakHeight', mph,
                                                             nargout=4))
    return (np.array(ipeaks[0], dtype=int) - 1, np.array(widths[0]), np.array(proms[0]))


def detectSpikesMATLAB(t, y, mph, mpp, mtd):
    ''' Detect spikes on a signal using the MATLAB findpeaks function,
        and return their indexes, width and prominence.

        :param t: regular time vector (s)
        :param y: signal vector (signal units)
        :param mph: minimal peak height (signal units)
        :param mpp: minimal peak prominence (signal units)
        :param mtd: minimal distance between consecutive peaks (time difference)
        :return: 3-tuple with indexes, widths and prominences of spikes
    '''
    dt = t[1] - t[0]
    ipeaks, widthpeaks, prompeaks = findPeaksMATLAB(y, mpp, int(np.ceil(mtd / dt)), mph)
    return (ipeaks, widthpeaks * dt, prompeaks)


def detectSpikesInFile(filename, mph, mpp, mtd):
    if not os.path.isfile(filename):
        raise InputError('File does not exist')
    with open(filename, 'rb') as fh:
        frame = pickle.load(fh)
    df = frame['data']
    t = df['t'].values
    Qm = df['Qm'].values
    return detectSpikesMATLAB(t, Qm, mph, mpp, mtd)
