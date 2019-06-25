# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-06 21:15:32
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-25 23:18:14

import os
import re
import pickle
import numpy as np
from scipy.interpolate import interp1d

from ..utils import isWithin, logger, isIterable


NEURONS_LOOKUP_DIR = os.path.split(__file__)[0]


class Lookup:
    ''' Lookup object. '''

    def __init__(self, refs, tables):
        self.refs = refs
        self.tables = SmartDict(tables)
        for k, v in self.items():
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

    def keys(self):
        return self.tables.keys()

    def values(self):
        return self.tables.values()

    def items(self):
        return self.tables.items()

    def refitems(self):
        return self.refs.items()

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
        return list(self.keys())

    def squeeze(self):
        new_tables = {k: v.squeeze() for k, v in self.items()}
        new_refs = {}
        for k, v in self.refitems():
            if v.size > 1:
                new_refs[k] = v
        return self.__class__(new_refs, new_tables)

    def getAxisIndex(self, key):
        assert key in self.inputs(), 'Unkown input dimension: {}'.format(key)
        return self.inputs().index(key)

    def project(self, key, value):
        ''' Interpolate tables at specific value(s) along a given dimension. '''
        if not isIterable(value):
            delete_input_dim = True
        else:
            delete_input_dim = False
            value = np.asarray(value)

        value = isWithin(key, value, (self.refs[key].min(), self.refs[key].max()))
        axis = self.getAxisIndex(key)
        # print('interpolating lookup along {} (axis {}) at {}'.format(key, axis, value))

        new_tables = {}
        for k, v in self.items():
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
        return SmartDict({k: self.interpVar(value, key, k) for k in self.outputs()})

    def projectOff(self):
        # Interpolate at zero amplitude
        lkp0 = self.project('A', 0.)

        # Move charge axis to end in all tables
        Qaxis = lkp0.getAxisIndex('Q')
        for k, v in lkp0.items():
            lkp0.tables[k] = np.moveaxis(v, Qaxis, -1)

        # Iterate along dimensions and take first value along corresponding axis
        for i in range(lkp0.ndims() - 1):
            for k, v in lkp0.items():
                lkp0.tables[k] = v[0]

        # Keep only charge vector in references
        lkp0.refs = {'Q': lkp0.refs['Q']}

        return lkp0

    def move(self, key, index):
        if index == -1:
            index = self.ndims() - 1
        iref = self.getAxisIndex(key)
        for k in self.keys():
            self.tables[k] = np.moveaxis(self.tables[k], iref, index)
        refkeys = list(self.refs.keys())
        del refkeys[iref]
        refkeys = refkeys[:index] + [key] + refkeys[index:]
        self.refs = {k: self.refs[k] for k in refkeys}

    def projectDCs(self, amps=None, DCs=1.):
        if amps is None:
            amps = self.refs['A']
        elif not isIterable(amps):
            amps = np.array([amps])

        if not isIterable(DCs):
            DCs = np.array([DCs])

        # project lookups at zero and defined amps
        if amps is None:
            amps = self.refs['A']
        lkp0 = self.project('A', 0.)
        lkps = self.project('A', amps)

        # Retrieve amplitude axis index, and move amplitude to first axis
        A_axis = lkps.getAxisIndex('A')
        lkps.move('A', 0)

        # Define empty tables dictionary
        tables_DCavg = {}

        # For each variable
        for var_key in lkp0.outputs():

            # Get OFF and ON (for all amps) variable values
            x_on, x_off = lkps.tables[var_key], lkp0.tables[var_key]

            # Initialize empty table to gather DC-averaged variable (DC size + ON table shape)
            x_avg = np.empty((DCs.size, *x_on.shape))

            # Compute the DC-averaged variable for each amplitude-DC combination
            for iA, Adrive in enumerate(amps):
                for iDC, DC in enumerate(DCs):
                    x_avg[iDC, iA] = x_on[iA] * DC + x_off * (1 - DC)

            # Assign table in dictionary
            tables_DCavg[var_key] = x_avg

        refs_DCavg = {**{'DC': DCs}, **lkps.refs}
        lkp = self.__class__(refs_DCavg, tables_DCavg)

        # Move DC ot last axis and amplitude back to its original axis
        lkp.move('DC', -1)
        lkp.move('A', A_axis)
        return lkp

    def projectFs(self):
        pass


class SmartDict():

    # Key patterns
    suffix_pattern = '[A-Za-z0-9_]+'
    xinf_pattern = re.compile('^({})inf$'.format(suffix_pattern))
    taux_pattern = re.compile('^tau({})$'.format(suffix_pattern))

    def __init__(self, d):
        self.d = d

    def __repr__(self):
        return 'SmartDict(' + ', '.join(self.d.keys())

    def items(self):
        return self.d.items()

    def keys(self):
        return self.d.keys()

    def values(self):
        return self.d.values()

    def alphax(self, x):
        return self.d['alpha{}'.format(x)]

    def betax(self, x):
        return self.d['beta{}'.format(x)]

    def taux(self, x):
        return 1 / (self.alphax(x) + self.betax(x))

    def xinf(self, x):
        return self.alphax(x) * self.taux(x)

    def __getitem__(self, key):
        if key in self.d:
            return self.d[key]
        else:
            m = self.taux_pattern.match(key)
            if m is not None:
                return self.taux(m.group(1))
            else:
                m = self.xinf_pattern.match(key)
                if m is not None:
                    return self.xinf(m.group(1))
                else:
                    raise KeyError(key)

    def __setitem__(self, key, value):
        self.d[key] = value


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
    return Lookup(frame['input'], frame['lookup'])


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
