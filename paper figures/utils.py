# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-10-01 20:45:29
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-10-02 00:31:10

import os
import numpy as np
import pandas as pd

from PySONIC.utils import *
from PySONIC.core import NeuronalBilayerSonophore
from PySONIC.neurons import *
from PySONIC.batches import runBatch
from PySONIC.postpro import computeSpikingMetrics


def getCWtitrations(neurons, a, freqs, tstim, toffset, fpath):
    fkey = 'Fdrive (kHz)'
    freqs = np.array(freqs)
    if os.path.isfile(fpath):
        df = pd.read_csv(fpath, sep=',', index_col=fkey)
    else:
        df = pd.DataFrame(index=freqs * 1e-3)
    for neuron in neurons:
        if neuron not in df:
            neuronobj = getNeuronsDict()[neuron]()
            nbls = NeuronalBilayerSonophore(a, neuronobj)
            for i, Fdrive in enumerate(freqs):
                logger.info('Running CW titration for %s neuron @ %sHz',
                            neuron, si_format(Fdrive))
                Athr = nbls.titrate(Fdrive, tstim, toffset)[0]  # Pa
                df.loc[Fdrive * 1e-3, neuron] = np.ceil(Athr * 1e-2) / 10
    df.sort_index(inplace=True)
    df.to_csv(fpath, sep=',', index_label=fkey)
    return df


def getSims(outdir, neuron, a, queue):
    fpaths = []
    updated_queue = []
    for i, item in enumerate(queue):
        Fdrive, tstim, _, PRF, DC, Adrive, method = item
        fcode = ASTIM_filecode(neuron, a, Fdrive, Adrive, tstim, PRF, DC, method)
        fpath = os.path.join(outdir, '{}.pkl'.format(fcode))
        if not os.path.isfile(fpath):
            print(fpath, 'does not exist')
            updated_queue.append(item)
        fpaths.append(fpath)
    if len(updated_queue) > 0:
        print(updated_queue)
        # neuron = getNeuronsDict()[neuron]()
        # nbls = NeuronalBilayerSonophore(a, neuron)
        # runBatch(nbls, 'runAndSave', updated_queue, extra_params=[outdir], mpi=True)
    return fpaths


def getSpikingMetrics(outdir, neuron, xvar, xkey, full_fpaths, sonic_fpaths, metrics_fpaths):
    if os.path.isfile(metrics_fpaths['full']):
        logger.info('loading spiking metrics from files: "%s" and "%s"',
                    metrics_fpaths['full'], metrics_fpaths['sonic'])
        full_metrics = pd.read_csv(metrics_fpaths['full'], sep=',')
        sonic_metrics = pd.read_csv(metrics_fpaths['sonic'], sep=',')
    else:
        logger.warning('computing spiking metrics vs. %s for %s neuron', xkey, neuron)
        full_metrics = computeSpikingMetrics(full_fpaths)
        full_metrics[xkey] = pd.Series(xvar, index=full_metrics.index)
        full_metrics.to_csv(metrics_fpaths['full'], sep=',', index=False)
        sonic_metrics = computeSpikingMetrics(sonic_fpaths)
        sonic_metrics[xkey] = pd.Series(xvar, index=sonic_metrics.index)
        sonic_metrics.to_csv(metrics_fpaths['sonic'], sep=',', index=False)
    return full_metrics, sonic_metrics


def extractCompTimes(filenames):
    ''' Extract computation times from a list of simulation files. '''
    tcomps = np.empty(len(filenames))
    for i, fn in enumerate(filenames):
        logger.info('Loading data from "%s"', fn)
        with open(fn, 'rb') as fh:
            frame = pickle.load(fh)
            meta = frame['meta']
        tcomps[i] = meta['tcomp']
    return tcomps


def getCompTimesQual(outdir, neurons, xvars, full_fpaths, sonic_fpaths, comptimes_fpaths):
    if os.path.isfile(comptimes_fpaths['full']):
        logger.info('reading computation times from files: "%s" and "%s"',
                    comptimes_fpaths['full'], comptimes_fpaths['sonic'])
        full_comptimes = pd.read_csv(comptimes_fpaths['full'], sep=',', index_col='neuron')
        sonic_comptimes = pd.read_csv(comptimes_fpaths['sonic'], sep=',', index_col='neuron')
    else:
        full_comptimes = pd.DataFrame(index=neurons)
        sonic_comptimes = pd.DataFrame(index=neurons)
        for xvar in xvars:
            for neuron in neurons:
                logger.warning('extracting %s computation times for %s neuron', xvar, neuron)
                full_comptimes.loc[neuron, xvar] = extractCompTimes(full_fpaths[neuron][xvar])
                sonic_comptimes.loc[neuron, xvar] = extractCompTimes(sonic_fpaths[neuron][xvar])
        full_comptimes.to_csv(comptimes_fpaths['full'], sep=',', index_label='neuron')
        sonic_comptimes.to_csv(comptimes_fpaths['sonic'], sep=',', index_label='neuron')
    return full_comptimes, sonic_comptimes


def getCompTimesQuant(outdir, neuron, xvars, xkey, full_fpaths, sonic_fpaths, comptimes_fpath):
    if os.path.isfile(comptimes_fpath):
        logger.info('reading computation times from file: "%s"', comptimes_fpath)
        comptimes = pd.read_csv(comptimes_fpath, sep=',', index_col=xkey)
    else:
        logger.warning('extracting computation times for %s neuron', neuron)
        comptimes = pd.DataFrame(index=xvars)
        for i, xvar in enumerate(xvars):
            comptimes.loc[xvar, 'full'] = extractCompTimes([full_fpaths[i]])
            comptimes.loc[xvar, 'sonic'] = extractCompTimes([sonic_fpaths[i]])
        comptimes.to_csv(comptimes_fpath, sep=',', index_label=xkey)
    return comptimes
