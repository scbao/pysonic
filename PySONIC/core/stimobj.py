# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2020-04-21 11:32:49
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-29 11:01:14

import abc

from ..utils import isIterable, si_format


class StimObject(metaclass=abc.ABCMeta):
    ''' Generic interface to a simulation object. '''
    fcode_replace_pairs = [
        ('/', '_per_'),
        (',', '_'),
        ('(', ''),
        (')', ''),
        (' ', '')
    ]

    @abc.abstractmethod
    def copy(self):
        ''' String representation. '''
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def inputs():
        raise NotImplementedError

    def xformat(self, x, factor, precision, minfigs, strict_nfigs=False):
        if isIterable(x):
            l = [self.xformat(xx, factor, precision, minfigs, strict_nfigs=strict_nfigs)
                 for xx in x]
            return f'({", ".join(l)})'
        if isinstance(x, str):
            return x
        xf = si_format(x * factor, precision=precision, space='')
        if strict_nfigs:
            if minfigs is not None:
                nfigs = len(xf.split('.')[0])
                if nfigs < minfigs:
                    xf = '0' * (minfigs - nfigs) + xf
        return xf

    def paramStr(self, k, **kwargs):
        val = getattr(self, k)
        if val is None:
            return None
        xf = self.xformat(
            val,
            self.inputs()[k].get('factor', 1.),
            self.inputs()[k].get('precision', 0),
            self.inputs()[k].get('minfigs', None),
            **kwargs)
        return f"{xf}{self.inputs()[k].get('unit', '')}"

    def pdict(self, sf='{key}={value}', **kwargs):
        d = {k: self.paramStr(k, **kwargs) for k in self.inputs().keys()}
        return {k: sf.format(key=k, value=v) for k, v in d.items() if v is not None}

    @property
    def meta(self):
        return {k: getattr(self, k) for k in self.inputs().keys()}

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        for k in self.inputs().keys():
            if getattr(self, k) != getattr(other, k):
                return False
        return True

    def __repr__(self):
        return f'{self.__class__.__name__}({", ".join(self.pdict().values())})'

    @property
    def desc(self):
        return ', '.join(self.pdict(sf='{key} = {value}').values())

    def slugify(self, s):
        for pair in self.fcode_replace_pairs:
            s = s.replace(*pair)
        return s

    @property
    def filecodes(self):
        d = self.pdict(sf='{key}_{value}', strict_nfigs=True)
        return {k: self.slugify(v) for k, v in d.items()}

    def checkInt(self, key, value):
        if not isinstance(value, int):
            raise TypeError(f'Invalid {self.inputs()[key]["desc"]} (must be an integer)')
        return value

    def checkFloat(self, key, value):
        if isinstance(value, int):
            value = float(value)
        if not isinstance(value, float):
            raise TypeError(f'Invalid {self.inputs()[key]["desc"]} (must be float typed)')
        return value

    def checkStrictlyPositive(self, key, value):
        if value <= 0:
            raise ValueError(f'Invalid {key} (must be strictly positive)')

    def checkPositiveOrNull(self, key, value):
        if value < 0:
            raise ValueError(f'Invalid {key} (must be positive or null)')

    def checkStrictlyNegative(self, key, value):
        if value >= 0:
            raise ValueError(f'Invalid {key} (must be strictly negative)')

    def checkNegativeOrNull(self, key, value):
        if value > 0:
            d = self.inputs()[key]
            raise ValueError(f'Invalid {key} {d["unit"]} (must be negative or null)')

    def checkBounded(self, key, value, bounds):
        if value < bounds[0] or value > bounds[1]:
            d = self.inputs()[key]
            f, u = d.get("factor", 1), d["unit"]
            bounds_str = f'[{bounds[0] * f}; {bounds[1] * f}] {u}'
            raise ValueError(f'Invalid {d["desc"]}: {value * f} {u} (must be within {bounds_str})')
