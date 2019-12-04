# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-27 13:59:02
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-12-04 15:35:56

import re
import numpy as np
from scipy.interpolate import interp1d

from ..utils import isWithin, isIterable


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
        ref_str = ', '.join([f'{x[0]}: {x[1]}' for x in zip(self.inputs(), self.dims())])
        return f'Lookup{self.ndims()}D({ref_str})'

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

    def checkAgainst(self, other):
        if self.inputs() != other.inputs():
            raise ValueError(f'Differing lookups (references names do not match)')
        if self.dims() != other.dims():
            raise ValueError(f'Differing lookup dimensions ({self.dims()} - {other.dims()})')
        for k, v in self.refitems():
            if (other.refs[k] != v).any():
                raise ValueError(f'Differing {k} lookup reference')
        if self.outputs() != other.outputs():
            raise ValueError(f'Differing lookups (table names do not match)')

    def operate(self, other, op):
        ''' Generic arithmetic operator. '''
        if isinstance(other, int):
            other = float(other)
        if isinstance(other, self.__class__):
            self.checkAgainst(other)
            tables = {k: getattr(v, op)(other[k]) for k, v in self.items()}
        elif isinstance(other, float):
            tables = {k: getattr(v, op)(other) for k, v in self.items()}
        else:
            raise ValueError(f'Cannot {op} {self.__class__} object with {type(other)} variable')
        return self.__class__(self.refs, tables)

    def __add__(self, other):
        return self.operate(other, '__add__')

    def __sub__(self, other):
        return self.operate(other, '__sub__')

    def __mul__(self, other):
        return self.operate(other, '__mul__')

    def __div__(self, other):
        return self.operate(other, '__div__')

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
        if self.refs[key].size == 1:
            for k, v in self.items():
                new_tables[k] = v.mean(axis=axis)
        else:
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

    def interpVar(self, ref_value, ref_key, var_key):
        return np.interp(
            ref_value, self.refs[ref_key], self.tables[var_key], left=np.nan, right=np.nan)

    def interpolate1D(self, key, value):
        return SmartDict({k: self.interpVar(value, key, k) for k in self.outputs()})

    def tile(self, ref_name, ref_values):
        ''' Tile the lookups along a new dimension. '''
        itiles = range(ref_values.size)
        tables = {k: np.array([v for i in itiles]) for k, v in self.items()}
        refs = {**{ref_name: ref_values}, **self.refs}
        return self.__class__(refs, tables)


class SmartLookup(Lookup):

    def __repr__(self):
        return 'Smart' + super().__repr__()

    def projectOff(self):
        ''' Project for OFF periods (zero amplitude). '''
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

    def projectDC(self, amps=None, DC=1.):
        ''' Project lookups at a given duty cycle.'''
        # Assign default values
        if amps is None:
            amps = self.refs['A']
        elif not isIterable(amps):
            amps = np.array([amps])

        # project lookups at zero and defined amps
        lkp0 = self.project('A', 0.)
        lkps_ON = self.project('A', amps)

        # Retrieve amplitude axis index, and move amplitude to first axis
        A_axis = lkps_ON.getAxisIndex('A')
        lkps_ON.move('A', 0)

        # Tile the zero-amplitude lookup to match the lkps_ON dimensions
        lkps_OFF = lkp0.tile('A', lkps_ON.refs['A'])

        # Compute a DC averaged lookup
        lkp = lkps_ON * DC + lkps_OFF * (1 - DC)

        # Move amplitude back to its original axis
        lkp.move('A', A_axis)

        return lkp


def fixLookup(lkp):
    if 'fs' in lkp.inputs():
        if lkp.refs['fs'].size == 1:
            if lkp.refs['fs'][0] == 1.:
                lkp = lkp.project('fs', 1.)
    return lkp


class SmartDict():

    # Key patterns
    suffix_pattern = '[A-Za-z0-9_]+'
    xinf_pattern = re.compile('^({})inf$'.format(suffix_pattern))
    taux_pattern = re.compile('^tau({})$'.format(suffix_pattern))

    def __init__(self, d):
        self.d = d

    def __repr__(self):
        return 'SmartDict(' + ', '.join(self.d.keys()) + ')'

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

    def pop(self, key):
        return self.d.pop(key)
