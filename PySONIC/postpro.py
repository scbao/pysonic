# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-28 18:57:40

''' Utility functions to detect spikes on signals and compute spiking metrics. '''

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.signal import find_peaks, peak_prominences

from .constants import *
from .utils import logger, isIterable, loadData


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
        bounds_str = [f'{dt[i]:.2e} s (index {i})' for i in [dt.argmin(), dt.argmax()]]
        raise ValueError(f'non-uniform time step: from {bounds_str[0]} to {bounds_str[1]}')

    # Return average dt value
    return np.mean(dt)  # s


def resample(t, y, dt):
    ''' Resample a dataframe at regular time step. '''
    n = int(np.ptp(t) / dt) + 1
    ts = np.linspace(t.min(), t.max(), n)
    ys = np.interp(ts, t, y)
    return ts, ys


def resolveIndexes(indexes, y, choice='max'):
    if indexes.size == 0:
        return indexes
    icomp = np.array([np.floor(indexes), np.ceil(indexes)]).astype(int).T
    ycomp = np.array([y[i] for i in icomp])
    method = {'min': np.argmin, 'max': np.argmax}[choice]
    ichoice = method(ycomp, axis=1)
    return np.array([x[ichoice[i]] for i, x in enumerate(icomp)])


def resampleDataFrame(data, dt):
    ''' Resample a dataframe at regular time step. '''
    t = data['t'].values
    n = int(np.ptp(t) / dt) + 1
    tnew = np.linspace(t.min(), t.max(), n)
    new_data = {}
    for key in data:
        kind = 'nearest' if key == 'stimstate' else 'linear'
        new_data[key] = interp1d(t, data[key].values, kind=kind)(tnew)
    return pd.DataFrame(new_data)


def prependDataFrame(data, tonset=0.):
    ''' Add an initial value (for t = 0) to all columns of a dataframe. '''
    # Repeat first row
    data = pd.concat([pd.DataFrame([data.iloc[0]]), data], ignore_index=True)
    data['t'][0] = tonset
    data['stimstate'][0] = 0
    return data


def boundDataFrame(data, tbounds):
    ''' Restrict all columns of a dataframe to indexes corresponding to
        time values within specific bounds. '''
    tmin, tmax = tbounds
    return data[np.logical_and(data.t >= tmin, data.t <= tmax)].reset_index(drop=True)


def find_tpeaks(t, y, **kwargs):
    ''' Wrapper around the scipy.signal.find_peaks function that provides a time vector
        associated to the signal, and translates time-based selection criteria into
        index-based criteria before calling the function.

        :param t: time vector
        :param y: signal vector
        :return: 2-tuple with peaks timings and properties dictionary
    '''
    # Remove initial samples from vectors if time values are redundant
    ipad = 0
    while t[ipad + 1] == t[ipad]:
        ipad += 1
    if ipad > 0:
        ss = 'from vectors (redundant time values)'
        if ipad == 1:
            logger.debug(f'Removing index 0 {ss}')
        else:
            logger.debug(f'Removing indexes 0-{ipad - 1} {ss}')
        t = t[ipad:]
        y = y[ipad:]

    # If time step is irregular, resample vectors at a uniform time step
    try:
        dt = computeTimeStep(t)  # s
        t_raw, y_raw = None, None
        indexes_raw = None
    except ValueError:
        new_dt = max(np.diff(t).min(), 1e-7)
        logger.debug(f'Resampling vector at regular time step (dt = {new_dt:.2e}s)')
        t_raw, y_raw = t.copy(), y.copy()
        indexes_raw = np.arange(t_raw.size)
        t, y = resample(t, y, new_dt)
        dt = computeTimeStep(t)  # s

    # Convert provided time-based input criteria into samples-based criteria
    for key in ['distance', 'width', 'wlen', 'plateau_size']:
        if key in kwargs:
            kwargs[key] = convertTime2SampleCriterion(kwargs[key], dt, t.size)
    if 'width' not in kwargs:
        kwargs['width'] = 1

    # Find peaks in the regularly sampled signal
    ipeaks, pps = find_peaks(y, **kwargs)

    # Adjust peak prominences and bases with restricted analysis window length
    # based on smallest peak width
    if len(ipeaks) > 0:
        wlen = 5 * min(pps['widths'])
        pps['prominences'], pps['left_bases'], pps['right_bases'] = peak_prominences(
            y, ipeaks, wlen=wlen)

    # If needed, re-project index-based outputs onto original sampling
    if t_raw is not None:
        logger.debug(f're-projecting index-based outputs onto original sampling')
        # Interpolate peak indexes and round to neighbor integer with max y value
        ipeaks_raw = np.interp(t[ipeaks], t_raw, indexes_raw, left=np.nan, right=np.nan)
        ipeaks = resolveIndexes(ipeaks_raw, y_raw, choice='max')

        # Interpolate peak base indexes and round to neighbor integer with min y value
        for key in ['left_bases', 'right_bases']:
            if key in pps:
                ibase_raw = np.interp(
                    t[pps[key]], t_raw, indexes_raw, left=np.nan, right=np.nan)
                pps[key] = resolveIndexes(ibase_raw, y_raw, choice='min')

        # Interpolate peak half-width interpolated positions
        for key in ['left_ips', 'right_ips']:
            if key in pps:
                pps[key] = np.interp(
                    dt * pps[key], t_raw, indexes_raw, left=np.nan, right=np.nan)

    # If original vectors were cropped, correct offset in index-based outputs
    if ipad > 0:
        logger.debug(f'offseting index-based outputs by {ipad} to compensate initial cropping')
        ipeaks += ipad
        for key in ['left_bases', 'right_bases', 'left_ips', 'right_ips']:
            if key in pps:
                pps[key] += ipad

    # Convert index-based peak widths into time-based widths
    if 'widths' in pps:
        pps['widths'] = np.array(pps['widths']) * dt

    # Return updated properties
    return ipeaks, pps


def detectSpikes(data, key='Qm', mpt=SPIKE_MIN_DT, mph=SPIKE_MIN_QAMP, mpp=SPIKE_MIN_QPROM):
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

    # Detect peaks
    return find_tpeaks(
        data['t'].values,
        data[key].values,
        height=mph,
        distance=mpt,
        prominence=mpp
    )


def convertPeaksProperties(t, properties):
    ''' Convert index-based peaks properties into time-based properties.

        :param t: time vector (s)
        :param properties: properties dictionary (with index-based information)
        :return: properties dictionary (with time-based information)
    '''
    indexes = np.arange(t.size)
    for key in ['left_bases', 'right_bases', 'left_ips', 'right_ips']:
        if key in properties:
            properties[key] = np.interp(properties[key], indexes, t, left=np.nan, right=np.nan)
    return properties


def computeFRProfile(data):
    ''' Compute temporal profile of firing rate from simulaton output.

        :param data: simulation output dataframe
        :return: firing rate profile interpolated along time vector
    '''
    # Detect spikes in data
    ispikes, _ = detectSpikes(data)
    if len(ispikes) == 0:
        return np.ones(len(data)) * np.nan

    # Compute firing rate as function of spike time
    t = data['t'].values
    tspikes = t[ispikes][:-1]
    sr = 1 / np.diff(t[ispikes])
    if len(sr) == 0:
        return np.ones(t.size) * np.nan

    # Interpolate firing rate vector along time vector
    return np.interp(t, tspikes, sr, left=np.nan, right=np.nan)


def computeSpikingMetrics(outputs):
    ''' Analyze the charge density profile from a list of files and compute for each one of them
        the following spiking metrics:
        - latency (ms)
        - firing rate mean and standard deviation (Hz)
        - spike amplitude mean and standard deviation (nC/cm2)
        - spike width mean and standard deviation (ms)

        :param outputs: list / generator of simulation outputs
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
    for output in outputs:

        # Load data
        if isinstance(output, str):
            data, meta = loadData(output)
        else:
            data, meta = output
        tstim = meta['pp'].tstim
        t = data['t'].values

        # Detect spikes in data and extract features
        ispikes, properties = detectSpikes(data)
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
