# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-25 11:36:13

''' Utility functions to detect spikes on signals and compute spiking metrics. '''

import pickle
import numpy as np
import pandas as pd

from .constants import *
from .utils import logger


def detectPeaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False, valley=False, ax=None):
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
    ''' Extension of the detectPeaks function to detect peaks in data based on their
        amplitude and time difference, with a non-uniform time vector.

        :param t: time vector (not necessarily uniform)
        :param y: signal
        :param mph: minimal peak height
        :param mtd: minimal time difference
        :mpp: minmal peak prominence
        :return: array of peak indexes
    '''

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

    # Differentiate signal
    dy = np.diff(y)

    # Find all peaks and valleys
    # s = np.sign(dy)
    # ipeaks = np.where(np.diff(s) < 0.0)[0] + 1
    # ivalleys = np.where(np.diff(s) > 0.0)[0] + 1
    ipeaks = np.where((np.hstack((dy, 0)) <= 0) & (np.hstack((0, dy)) > 0))[0]
    ivalleys = np.where((np.hstack((dy, 0)) >= 0) & (np.hstack((0, dy)) < 0))[0]

    # Return empty output if no peak detected
    if ipeaks.size == 0:
        return empty

    logger.debug('%u peaks found, starting at index %u and ending at index %u',
                 ipeaks.size, ipeaks[0], ipeaks[-1])
    if ivalleys.size > 0:
        logger.debug('%u valleys found, starting at index %u and ending at index %u',
                     ivalleys.size, ivalleys[0], ivalleys[-1])
    else:
        logger.debug('no valleys found')

    # Ensure each peak is bounded by two valleys, adding signal boundaries as valleys if necessary
    if ivalleys.size == 0 or ipeaks[0] < ivalleys[0]:
        ivalleys = np.insert(ivalleys, 0, -1)
    if ipeaks[-1] > ivalleys[-1]:
        ivalleys = np.insert(ivalleys, ivalleys.size, y.size - 1)
    if ivalleys.size - ipeaks.size != 1:
        logger.debug('Cleaning up incongruities')
        i = 0
        while i < min(ipeaks.size, ivalleys.size) - 1:
            if ipeaks[i] < ivalleys[i]:  # 2 peaks between consecutive valleys -> remove lowest
                idel = i - 1 if y[ipeaks[i - 1]] < y[ipeaks[i]] else i
                logger.debug('Removing abnormal peak at index %u', ipeaks[idel])
                ipeaks = np.delete(ipeaks, idel)
            if ipeaks[i] > ivalleys[i + 1]:
                idel = i + 1 if y[ivalleys[i]] < y[ivalleys[i + 1]] else i
                logger.debug('Removing abnormal valley at index %u', ivalleys[idel])
                ivalleys = np.delete(ivalleys, idel)
            else:
                i += 1
        logger.debug('Post-cleanup: %u peaks and %u valleys', ipeaks.size, ivalleys.size)

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

    return (ipeaks - 1, prominences, widths, ibounds)


def computeSpikingMetrics(filenames):
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
