# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-06 21:15:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-20 19:30:41

import os
import re
import pickle
import numpy as np
from scipy.interpolate import interp1d

from ..utils import isWithin, logger


NEURONS_LOOKUP_DIR = os.path.split(__file__)[0]


class Lookup:
    ''' Lookup object. '''

    def __init__(self, refs, tables):
        self.refs = refs
        self.tables = tables
        for k, v in self.tables.items():
            if v.shape != self.dims():
                raise ValueError('{} Table dimensions {} does not match references {}'.format(
                    k, v.shape, self.dims()))

    def __repr__(self):
        return 'Lookup{}D({})'.format(self.ndims(), ', '.join(self.inputs()))

    def __getitem__(self, key):
        return self.tables[key]

    def __delitem__(self, key):
        del self.tables[key]

    def __setitem__(self, key, value):
        self.tables[key] = value

    def pop(self, key):
        x = self.tables[key]
        del self.tables[key]
        return x

    def rename(self, key1, key2):
        self.tables[key2] = self.tables.pop(key1)

    def dims(self):
        return tuple([x.size for x in self.refs.values()])

    def ndims(self):
        return len(self.refs)

    def inputs(self):
        return list(self.refs.keys())

    def outputs(self):
        return list(self.tables.keys())

    def getAxisIndex(self, key):
        assert key in self.inputs(), 'Unkown input dimension: {}'.format(key)
        return self.inputs().index(key)

    def project(self, key, value):
        ''' Interpolate tables at specific value(s) along a given dimension. '''
        if isinstance(value, float) or isinstance(value, int):
            delete_input_dim = True
        else:
            delete_input_dim = False
            value = np.asarray(value)

        value = isWithin(key, value, (self.refs[key].min(), self.refs[key].max()))
        axis = self.getAxisIndex(key)
        # print('interpolating lookup along {} (axis {}) at {}'.format(key, axis, value))

        new_tables = {}
        for k, v in self.tables.items():
            new_tables[k] = interp1d(self.refs[key], v, axis=axis)(value)
        new_refs = self.refs.copy()
        if delete_input_dim:
            # print('removing {} input dimension'.format(key))
            del new_refs[key]
        else:
            # print('updating {} reference values'.format(key))
            new_refs[key] = value

        return self.__class__(new_refs, new_tables)

    def projectN(self, projections):
        lkp = self.__class__(self.refs, self.tables)
        for k, v in projections.items():
            lkp = lkp.project(k, v)
        return lkp

    def interpVar(self, ref_value, ref_key, var_key):
        return np.interp(
            ref_value, self.refs[ref_key], self.tables[var_key], left=np.nan, right=np.nan)

    def interpolate1D(self, key, value):
        return {k: self.interpVar(value, key, k) for k in self.outputs()}

    def projectOff(self):
        # Interpolate at zero amplitude
        lkp0 = self.project('A', 0.)

        # Move charge axis to end in all tables
        Qaxis = lkp0.getAxisIndex('Q')
        for k, v in lkp0.tables.items():
            lkp0.tables[k] = np.moveaxis(v, Qaxis, -1)

        # Iterate along dimensions and take first value along corresponding axis
        for i in range(lkp0.ndims() - 1):
            for k, v in lkp0.tables.items():
                lkp0.tables[k] = v[0]

        # Keep only charge vector in references
        lkp0.refs = {'Q': lkp0.refs['Q']}

        return lkp0

    def projectDCavg(self, amps, DCs):
        lkp0 = self.project('A', 0.)
        lkps = self.project('A', amps)
        nQ = self.refs['Q'].size
        tables_DCavg = {}
        for var_key in lkp0.outputs():
            x_on, x_off = lkp0.tables[var_key], lkps.tables[var_key]
            x_avg = np.empty((amps.size, nQ, DCs.size))
            for iA, Adrive in enumerate(amps):
                for iDC, DC in enumerate(DCs):
                    x_avg[iA, :, iDC] = x_on[iA, :] * DC + x_off * (1 - DC)
            tables_DCavg[var_key] = x_avg

        refs_DCavg = {k: v for k, v in self.refs.items()}
        refs_DCavg['DC'] = DCs
        return self.__class__(refs_DCavg, tables_DCavg)

    def projectFs(self):
        pass


class SmartLookup(Lookup):

    # Key patterns
    suffix_pattern = '[A-Za-z0-9_]+'
    xinf_pattern = re.compile('^({})inf$'.format(suffix_pattern))
    taux_pattern = re.compile('^tau({})$'.format(suffix_pattern))

    def __repr__(self):
        return 'Smart' + super().__repr__()

    def alphax(self, x):
        return self.tables['alpha{}'.format(x)]

    def betax(self, x):
        return self.tables['beta{}'.format(x)]

    def taux(self, x):
        return 1 / (self.alphax(x) + self.betax(x))

    def xinf(self, x):
        return self.alphax(x) * self.taux(x)

    def __getitem__(self, key):
        if key in self.tables:
            return self.tables[key]
        else:
            m = self.taux_pattern.match(key)
            if m is not None:
                return self.taux(m.group(1))
            else:
                m = self.xinf_pattern.match(key)
                if m is not None:
                    return self.xinf(m.group(1))


def getNeuronLookupsFileName(name, a=None, Fdrive=None, Adrive=None, fs=False):
    fname = '{}_lookups'.format(name)
    if a is not None:
        fname += '_{:.0f}nm'.format(a * 1e9)
    if Fdrive is not None:
        fname += '_{:.0f}kHz'.format(Fdrive * 1e-3)
    if Adrive is not None:
        fname += '_{:.0f}kPa'.format(Adrive * 1e-3)
    if fs is True:
        fname += '_fs'
    return '{}.pkl'.format(fname)


def getNeuronLookupsFilePath(*args, **kwargs):
    return os.path.join(NEURONS_LOOKUP_DIR, getNeuronLookupsFileName(*args, **kwargs))


def getNeuronLookup(name, **kwargs):
    lookup_path = getNeuronLookupsFilePath(name, **kwargs)
    if not os.path.isfile(lookup_path):
        raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))
    with open(lookup_path, 'rb') as fh:
        frame = pickle.load(fh)
    if 'ng' in frame['lookup']:
        del frame['lookup']['ng']
    return SmartLookup(frame['input'], frame['lookup'])


def getLookupsCompTime(name):

    # Check lookup file existence
    lookup_path = getNeuronLookupsFilePath(name)
    if not os.path.isfile(lookup_path):
        raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))

    # Load lookups dictionary
    logger.debug('Loading comp times')
    with open(lookup_path, 'rb') as fh:
        df = pickle.load(fh)
        tcomps4D = df['tcomp']

    return np.sum(tcomps4D)


# def getLookups2Dfs(name, a, Fdrive, fs):

#     # Check lookup file existence
#     lookup_path = getNeuronLookupsFilePath(name, a=a, Fdrive=Fdrive, fs=True)
#     if not os.path.isfile(lookup_path):
#         raise FileNotFoundError('Missing lookup file: "{}"'.format(lookup_path))

#     # Load lookups dictionary
#     logger.debug('Loading %s lookup table with fs = %.0f%%', name, fs * 1e2)
#     with open(lookup_path, 'rb') as fh:
#         df = pickle.load(fh)
#         inputs = df['input']
#         lookups3D = df['lookup']

#     # Retrieve 1D inputs from lookups dictionary
#     fsref = inputs['fs']
#     Aref = inputs['A']
#     Qref = inputs['Q']

#     # Check that fs is within lookup range
#     fs = isWithin('coverage', fs, (fsref.min(), fsref.max()))

#     # Perform projection at fs
#     logger.debug('Interpolating lookups at fs = %s%%', fs * 1e2)
#     lookups2D = {key: interp1d(fsref, y3D, axis=2)(fs) for key, y3D in lookups3D.items()}

#     return Aref, Qref, lookups2D


# def getLookupsDCavg(name, a, Fdrive, amps=None, charges=None, DCs=1.0):
#     ''' Get the DC-averaged lookups of a specific neuron for a combination of US amplitudes,
#         charge densities and duty cycles, at a specific US frequency.

#         :param name: name of point-neuron model
#         :param a: sonophore radius (m)
#         :param Fdrive: US frequency (Hz)
#         :param amps: US amplitudes (Pa)
#         :param charges: membrane charge densities (C/m2)
#         :param DCs: duty cycle value(s)
#         :return: 4-tuple with reference values of US amplitude and charge density,
#             as well as interpolated Vmeff and QSS gating variables
#     '''

#     # Get lookups for specific (a, f, A) combination
#     Aref, Qref, lookups2D, _ = getLookups2D(name, a=a, Fdrive=Fdrive)
#     if 'ng' in lookups2D:
#         lookups2D.pop('ng')

#     # Derive inputs from lookups reference if not provided
#     if amps is None:
#         amps = Aref
#     if charges is None:
#         charges = Qref

#     # Transform inputs into arrays if single value provided
#     if isinstance(amps, float):
#         amps = np.array([amps])
#     if isinstance(charges, float):
#         charges = np.array([charges])
#     if isinstance(DCs, float):
#         DCs = np.array([DCs])
#     nA, nQ, nDC = amps.size, charges.size, DCs.size
#     # cs = {True: 's', False: ''}
#     # logger.debug('%u amplitude%s, %u charge%s, %u DC%s',
#     #              nA, cs[nA > 1], nQ, cs[nQ > 1], nDC, cs[nDC > 1])

#     # Re-interpolate lookups at input charges
#     lookups2D = {key: interp1d(Qref, y2D, axis=1)(charges) for key, y2D in lookups2D.items()}

#     # Interpolate US-ON (for each input amplitude) and US-OFF (A = 0) lookups
#     amps = isWithin('amplitude', amps, (Aref.min(), Aref.max()))
#     lookups_on = {key: interp1d(Aref, y2D, axis=0)(amps) for key, y2D in lookups2D.items()}
#     lookups_off = {key: interp1d(Aref, y2D, axis=0)(0.0) for key, y2D in lookups2D.items()}

#     # Compute DC-averaged lookups
#     lookups_DCavg = {}
#     for key in lookups2D.keys():
#         x_on, x_off = lookups_on[key], lookups_off[key]
#         x_avg = np.empty((nA, nQ, nDC))
#         for iA, Adrive in enumerate(amps):
#             for iDC, DC in enumerate(DCs):
#                 x_avg[iA, :, iDC] = x_on[iA, :] * DC + x_off * (1 - DC)
#         lookups_DCavg[key] = x_avg

#     return amps, charges, lookups_DCavg
