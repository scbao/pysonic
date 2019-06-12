# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-21 14:33:36
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-12 12:17:23

''' Useful functions to generate plots. '''

import numpy as np
import matplotlib

# Matplotlib parameters
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'arial'


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


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def extractPltVar(model, pltvar, df, meta=None, nsamples=0, name=''):
    if 'func' in pltvar:
        s = 'model.{}'.format(pltvar['func'])
        try:
            var = eval(s)
        except AttributeError:
            var = eval(s.replace('model', 'model.pneuron'))
    elif 'key' in pltvar:
        var = df[pltvar['key']]
    elif 'constant' in pltvar:
        var = eval(pltvar['constant']) * np.ones(nsamples)
    else:
        var = df[name]
    var = var.values.copy()

    if var.size == nsamples - 2:
        var = np.hstack((np.array([pltvar.get('y0', var[0])] * 2), var))
    var *= pltvar.get('factor', 1)

    return var


def computeMeshEdges(x, scale='lin'):
    ''' Compute the appropriate edges of a mesh that quads a linear or logarihtmic distribution.

        :param x: the input vector
        :param scale: the type of distribution ('lin' for linear, 'log' for logarihtmic)
        :return: the edges vector
    '''
    if scale == 'log':
        x = np.log10(x)
    dx = x[1] - x[0]
    n = x.size + 1
    return {'lin': np.linspace, 'log': np.logspace}[scale](x[0] - dx / 2, x[-1] + dx / 2, n)
