# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-08-22 15:29:40

''' Utility functions to detect spikes on signals and compute spiking metrics. '''

import pickle
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.signal import find_peaks, peak_prominences

from .constants import *
from .utils import logger, debug, isIterable, loadData


def detectCrossings(x, thr=0.0, edge='both'):
    '''
        Detect crossings of a threshold value in a 1D signal.

        :param x: 1D array_like data.
        :param edge: 'rising', 'falling', or 'both'
        :return: 1D array with the indices preceding the crossings
    '''
    ine, ire, ife = np.array([[], [], []], dtype=int)
    x_padright = np.hstack((x, x[-1]))
    x_padleft = np.hstack((x[0], x))
    if edge.lower() in ['falling', 'both']:
        ire = np.where((x_padright <= thr) & (x_padleft > thr))[0]
    if edge.lower() in ['rising', 'both']:
        ife = np.where((x_padright >= thr) & (x_padleft < thr))[0]
    ind = np.unique(np.hstack((ine, ire, ife))) - 1
    return ind


def getFixedPoints(x, dx, filter='stable', der_func=None):
    ''' Find fixed points in a 1D plane phase profile.

        :param x: variable (1D array)
        :param dx: derivative (1D array)
        :param filter: string indicating whether to consider only stable/unstable
            fixed points or both
        :param: der_func: derivative function
        :return: array of fixed points values (or None if none is found)
    '''
    fps = []
    edge = {'stable': 'falling', 'unstable': 'rising', 'both': 'both'}[filter]
    izc = detectCrossings(dx, edge=edge)
    if izc.size > 0:
        for i in izc:
            # If derivative function is provided, find root using iterative Brent method
            if der_func is not None:
                fps.append(brentq(der_func, x[i], x[i + 1], xtol=1e-16))

            # Otherwise, approximate root by linear interpolation
            else:
                fps.append(x[i] - dx[i] * (x[i + 1] - x[i]) / (dx[i + 1] - dx[i]))
        return np.array(fps)
    else:
        return np.array([])


def getEqPoint1D(x, dx, x0):
    ''' Determine equilibrium point in a 1D plane phase profile, for a given starting point.

        :param x: variable (1D array)
        :param dx: derivative (1D array)
        :param x0: abscissa of starting point (float)
        :return: abscissa of equilibrium point (or np.nan if none is found)
    '''

    # Find stable fixed points in 1D plane phase profile
    x_SFPs = getFixedPoints(x, dx, filter='stable')
    if x_SFPs.size == 0:
        return np.nan

    # Determine relevant stable fixed point from y0 sign
    y0 = np.interp(x0, x, dx, left=np.nan, right=np.nan)
    inds_subset = x_SFPs >= x0
    ind_SFP = 0
    if y0 < 0:
        inds_subset = ~inds_subset
        ind_SFP = -1
    x_SFPs = x_SFPs[inds_subset]

    if len(x_SFPs) == 0:
        return np.nan

    return x_SFPs[ind_SFP]


def convertTime2SampleCriterion(x, dt, nsamples):
    if isIterable(x) and len(x) == 2:
        return (convertTime2Sample(x[0], dt, nsamples), convertTime2Sample(x[1], dt, nsamples))
    else:
        if isIterable(x) and len(x) == nsamples:
            return np.array([convertTime2Sample(item, dt, nsamples) for item in x])
        elif x is None:
            return None
        else:
            return int(np.ceil(x / dt))

def computeTimeStep(t):
    ''' Compute time step based on time vector.

        :param t: time vector (s)
        :return: average time step (s)
    '''

    # Compute time step vector
    dt = np.diff(t)  # s

    # Raise error if time step vector is not uniform
    is_uniform_dt = np.allclose(np.diff(dt), np.zeros(dt.size - 1), atol=1e-5)
    if not is_uniform_dt:
        raise ValueError(f'non-uniform time vector (variation range = {np.ptp(dt)}')

    # Return average dt value
    return np.mean(dt)  # s


def find_tpeaks(t, y, **kwargs):
    ''' Wrapper around the scipy.signal.find_peaks function that provides a time vector
        associated to the signal, and translates time-based selection criteria into
        index-based criteria before calling the function.

        :param t: time vector
        :param y: signal vector
        :return: 2-tuple with peaks timings and properties dictionary
    '''
    # Compute index vector
    nsamples = t.size
    indexes = np.arange(nsamples)

    # Compute time step
    dt = computeTimeStep(t)  # s

    # Convert provided time-based input criteria into samples-based criteria
    time_based_inputs = ['distance', 'width', 'wlen', 'plateau_size']
    for key in time_based_inputs:
        if key in kwargs:
            kwargs[key] = convertTime2SampleCriterion(kwargs[key], dt, nsamples)
    if 'width' not in kwargs:
        kwargs['width'] = 1

    # Find peaks in the signal and return
    return find_peaks(y, **kwargs)


def convertPeaksProperties(t, properties):
    ''' Convert index-based peaks properties into time-based properties.

        :param t: time vector (s)
        :param properties: properties dictionary (with index-based information)
        :return: properties dictionary (with time-based information)
    '''
    index_based_outputs = [
        'left_bases', 'right_bases',
        'left_ips', 'right_ips',
        'left_edges', 'right_edges'
    ]
    index_distance_based_outputs = ['widths', 'plateau_sizes']
    indexes = np.arange(t.size)
    dt = computeTimeStep(t[1:])
    for key in index_based_outputs:
        if key in properties:
            properties[key] = np.interp(properties[key], indexes, t, left=np.nan, right=np.nan)
    for key in index_distance_based_outputs:
        if key in properties:
            properties[key] = np.array(properties[key]) * dt
    return properties


def detectSpikes(data, key='Qm', mpt=SPIKE_MIN_DT, mph=SPIKE_MIN_QAMP, mpp=SPIKE_MIN_QPROM, ipad=0):
    ''' Detect spikes in simulation output data, by detecting peaks with specific height, prominence
        and distance properties on a given signal.

        :param data: simulation output dataframe
        :param key: key of signal on which to detect peaks
        :param mpt: minimal time interval between two peaks (s)
        :param mph: minimal peak height (in signal units)
        :param mpp: minimal peak prominence (in signal units)
        :return: indexes and properties of detected spikes
    '''
    if key not in data:
        raise ValueError(f'{key} vector not available in dataframe')

    # If first two time samples are equal, remove first row from dataset and call function recursively
    if data['t'].values[1] == data['t'].values[0]:
        logger.debug('Removing first row from dataframe (reccurent time values)')
        data = data.iloc[1:]
        return detectSpikes(data, key=key, mpt=mpt, mph=mph, mpp=mpp, ipad=ipad + 1)

    # Detect peaks
    ipeaks, properties = find_tpeaks(
        data['t'].values,
        data[key].values,
        height=mph,
        distance=mpt,
        prominence=mpp
    )

    # Adjust peak prominences and bases with restricted analysis window length
    # based on smallest peak width
    if len(ipeaks) > 0:
        wlen = 5 * min(properties['widths'])
        properties['prominences'], properties['left_bases'], properties['right_bases'] = peak_prominences(
            data[key].values, ipeaks, wlen=wlen)

    # Correct index of specific outputs
    index_based_outputs = [
        'left_bases', 'right_bases',
        'left_ips', 'right_ips',
        'left_edges', 'right_edges'
    ]
    ipeaks += ipad
    for key in index_based_outputs:
        if key in properties:
            properties[key] += ipad

    return ipeaks, properties


def computeFRProfile(data):
    ''' Compute temporal profile of firing rate from simulaton output.

        :param data: simulation output dataframe
        :return: firing rate profile interpolated along time vector
    '''
    # Detect spikes in data
    ispikes, _ = detectSpikes(data)

    # Compute firing rate as function of spike time
    t = data['t'].values
    tspikes = t[ispikes][:-1]
    sr = 1 / np.diff(t[ispikes])

    # Interpolate firing rate vector along time vector
    return np.interp(t, tspikes, sr, left=np.nan, right=np.nan)


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
        data, meta = loadData(fname)
        tstim = meta['tstim']
        t = data['t'].values

        # Detect spikes in data
        ispikes, properties = detectSpikes(data)

        # Convert index-based outputs into time-based outputs
        properties = convertPeaksProperties(t, properties)
        widths = properties['widths']
        prominences = properties['prominences']

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
