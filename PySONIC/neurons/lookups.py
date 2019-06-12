# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-06 21:15:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-12 23:05:29

import os
import pickle
import numpy as np
from scipy.interpolate import interp1d

from ..utils import isWithin, logger, si_format


def getNeuronLookupsFile(name, a=None, Fdrive=None, Adrive=None, fs=False):
    fpath = os.path.join(os.path.split(__file__)[0], '{}_lookups'.format(name))
    if a is not None:
        fpath += '_{:.0f}nm'.format(a * 1e9)
    if Fdrive is not None:
        fpath += '_{:.0f}kHz'.format(Fdrive * 1e-3)
    if Adrive is not None:
        fpath += '_{:.0f}kPa'.format(Adrive * 1e-3)
    if fs is True:
        fpath += '_fs'
    return '{}.pkl'.format(fpath)


def getLookups4D(name):
    ''' Retrieve 4D lookup tables and reference vectors for a given point-neuron model

        :param name: name of point-neuron model
        :return: 4-tuple with 1D numpy arrays of reference input vectors (charge density and
            one other variable), a dictionary of associated 2D lookup numpy arrays, and
            a dictionary with information about the other variable.
    '''

    # Check lookup file existence
    lookup_path = getNeuronLookupsFile(name)
    if not os.path.isfile(lookup_path):
        raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))

    # Load lookups dictionary
    # logger.debug('Loading %s lookup table', name)
    with open(lookup_path, 'rb') as fh:
        df = pickle.load(fh)
        inputs = df['input']
        lookups4D = df['lookup']

    # Retrieve 1D inputs from lookups dictionary
    aref = inputs['a']
    Fref = inputs['f']
    Aref = inputs['A']
    Qref = inputs['Q']

    return aref, Fref, Aref, Qref, lookups4D


def getLookupsOff(name):
    ''' Retrieve appropriate US-OFF lookup tables and reference vectors
        for a given point-neuron model

        :param name: name of point-neuron model
        :return: 2-tuple with 1D numpy array of reference charge density
            and dictionary of associated 1D lookup numpy arrays.
    '''

    # Get 4D lookups and input vectors
    aref, Fref, Aref, Qref, lookups4D = getLookups4D(name)

    # Perform 2D projection in appropriate dimensions
    logger.debug('Interpolating lookups at A = 0')
    lookups_off = {key: y4D[0, 0, 0, :] for key, y4D in lookups4D.items()}

    return Qref, lookups_off


def getLookups2D(name, a=None, Fdrive=None, Adrive=None):
    ''' Retrieve appropriate 2D lookup tables and reference vectors
        for a given point-neuron model, projected at a specific combination
        of sonophore radius, US frequency and/or acoustic pressure amplitude.

        :param name: name of point-neuron model
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param Adrive: Acoustic peak pressure amplitude (Hz)
        :return: 4-tuple with 1D numpy arrays of reference input vectors (charge density and
            one other variable), a dictionary of associated 2D lookup numpy arrays, and
            a dictionary with information about the other variable.
    '''

    # Get 4D lookups and input vectors
    aref, Fref, Aref, Qref, lookups4D = getLookups4D(name)

    # Check that inputs are within lookup range
    if a is not None:
        a = isWithin('radius', a, (aref.min(), aref.max()))
    if Fdrive is not None:
        Fdrive = isWithin('frequency', Fdrive, (Fref.min(), Fref.max()))
    if Adrive is not None:
        Adrive = isWithin('amplitude', Adrive, (Aref.min(), Aref.max()))

    # Determine projection dimensions based on inputs
    var_a = {'name': 'a', 'label': 'sonophore radius', 'val': a, 'unit': 'm', 'factor': 1e9,
             'ref': aref, 'axis': 0}
    var_Fdrive = {'name': 'f', 'label': 'frequency', 'val': Fdrive, 'unit': 'Hz', 'factor': 1e-3,
                  'ref': Fref, 'axis': 1}
    var_Adrive = {'name': 'A', 'label': 'amplitude', 'val': Adrive, 'unit': 'Pa', 'factor': 1e-3,
                  'ref': Aref, 'axis': 2}
    if not isinstance(Adrive, float):
        var1 = var_a
        var2 = var_Fdrive
        var3 = var_Adrive
    elif not isinstance(Fdrive, float):
        var1 = var_a
        var2 = var_Adrive
        var3 = var_Fdrive
    elif not isinstance(a, float):
        var1 = var_Fdrive
        var2 = var_Adrive
        var3 = var_a

    # Perform 2D projection in appropriate dimensions
    # logger.debug('Interpolating lookups at (%s = %s%s, %s = %s%s)',
    #              var1['name'], si_format(var1['val'], space=' '), var1['unit'],
    #              var2['name'], si_format(var2['val'], space=' '), var2['unit'])
    lookups3D = {key: interp1d(var1['ref'], y4D, axis=var1['axis'])(var1['val'])
                 for key, y4D in lookups4D.items()}
    if var2['axis'] > var1['axis']:
        var2['axis'] -= 1
    lookups2D = {key: interp1d(var2['ref'], y3D, axis=var2['axis'])(var2['val'])
                 for key, y3D in lookups3D.items()}

    if var3['val'] is not None:
        logger.debug('Interpolating lookups at %d new %s values between %s%s and %s%s',
                     len(var3['val']), var3['name'],
                     si_format(min(var3['val']), space=' '), var3['unit'],
                     si_format(max(var3['val']), space=' '), var3['unit'])
        lookups2D = {key: interp1d(var3['ref'], y2D, axis=0)(var3['val'])
                     for key, y2D in lookups2D.items()}
        var3['ref'] = np.array(var3['val'])

    return var3['ref'], Qref, lookups2D, var3


def getLookups2Dfs(name, a, Fdrive, fs):

    # Check lookup file existence
    lookup_path = getNeuronLookupsFile(name, a=a, Fdrive=Fdrive, fs=True)
    if not os.path.isfile(lookup_path):
        raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))

    # Load lookups dictionary
    logger.debug('Loading %s lookup table with fs = %.0f%%', name, fs * 1e2)
    with open(lookup_path, 'rb') as fh:
        df = pickle.load(fh)
        inputs = df['input']
        lookups3D = df['lookup']

    # Retrieve 1D inputs from lookups dictionary
    fsref = inputs['fs']
    Aref = inputs['A']
    Qref = inputs['Q']

    # Check that fs is within lookup range
    fs = isWithin('coverage', fs, (fsref.min(), fsref.max()))

    # Perform projection at fs
    logger.debug('Interpolating lookups at fs = %s%%', fs * 1e2)
    lookups2D = {key: interp1d(fsref, y3D, axis=2)(fs) for key, y3D in lookups3D.items()}

    return Aref, Qref, lookups2D


def getLookupsDCavg(name, a, Fdrive, amps=None, charges=None, DCs=1.0):
    ''' Get the DC-averaged lookups of a specific neuron for a combination of US amplitudes,
        charge densities and duty cycles, at a specific US frequency.

        :param name: name of point-neuron model
        :param a: sonophore radius (m)
        :param Fdrive: US frequency (Hz)
        :param amps: US amplitudes (Pa)
        :param charges: membrane charge densities (C/m2)
        :param DCs: duty cycle value(s)
        :return: 4-tuple with reference values of US amplitude and charge density,
            as well as interpolated Vmeff and QSS gating variables
    '''

    # Get lookups for specific (a, f, A) combination
    Aref, Qref, lookups2D, _ = getLookups2D(name, a=a, Fdrive=Fdrive)
    if 'ng' in lookups2D:
        lookups2D.pop('ng')

    # Derive inputs from lookups reference if not provided
    if amps is None:
        amps = Aref
    if charges is None:
        charges = Qref

    # Transform inputs into arrays if single value provided
    if isinstance(amps, float):
        amps = np.array([amps])
    if isinstance(charges, float):
        charges = np.array([charges])
    if isinstance(DCs, float):
        DCs = np.array([DCs])
    nA, nQ, nDC = amps.size, charges.size, DCs.size
    # cs = {True: 's', False: ''}
    # logger.debug('%u amplitude%s, %u charge%s, %u DC%s',
    #              nA, cs[nA > 1], nQ, cs[nQ > 1], nDC, cs[nDC > 1])

    # Re-interpolate lookups at input charges
    lookups2D = {key: interp1d(Qref, y2D, axis=1)(charges) for key, y2D in lookups2D.items()}

    # Interpolate US-ON (for each input amplitude) and US-OFF (A = 0) lookups
    amps = isWithin('amplitude', amps, (Aref.min(), Aref.max()))
    lookups_on = {key: interp1d(Aref, y2D, axis=0)(amps) for key, y2D in lookups2D.items()}
    lookups_off = {key: interp1d(Aref, y2D, axis=0)(0.0) for key, y2D in lookups2D.items()}

    # Compute DC-averaged lookups
    lookups_DCavg = {}
    for key in lookups2D.keys():
        x_on, x_off = lookups_on[key], lookups_off[key]
        x_avg = np.empty((nA, nQ, nDC))
        for iA, Adrive in enumerate(amps):
            for iDC, DC in enumerate(DCs):
                x_avg[iA, :, iDC] = x_on[iA, :] * DC + x_off * (1 - DC)
        lookups_DCavg[key] = x_avg

    return amps, charges, lookups_DCavg


def getLookupsCompTime(name):

    # Check lookup file existence
    lookup_path = getNeuronLookupsFile(name)
    if not os.path.isfile(lookup_path):
        raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))

    # Load lookups dictionary
    logger.debug('Loading comp times')
    with open(lookup_path, 'rb') as fh:
        df = pickle.load(fh)
        tcomps4D = df['tcomp']

    return np.sum(tcomps4D)
