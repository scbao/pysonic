#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-03 11:53:04
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-01 16:27:32

import pickle
import abc
import inspect
import numpy as np

from ..utils import logger, debug


class Model(metaclass=abc.ABCMeta):
    ''' Generic model interface. '''

    @property
    @abc.abstractmethod
    def tscale(self):
        ''' Relevant temporal scale of the model. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def pprint(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def filecode(self, *args):
        raise NotImplementedError

    def getDesc(self):
        return inspect.getdoc(self).splitlines()[0]

    @property
    @abc.abstractmethod
    def getPltScheme(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def getPltVars(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def checkInputs(self, *args):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def simulate(self, *args, **kwargs):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def meta(self, *args):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def createQueue(self, *args):
        raise NotImplementedError

    def runAndSave(self, outdir, *args):
        ''' Simulate system and save results in a PKL file. '''

        # If no amplitude provided, perform titration to find it
        if None in args:
            iA = args.index(None)
            new_args = [x for x in args if x is not None]
            Athr = self.titrate(*new_args)
            if np.isnan(Athr):
                logger.error('Could not find threshold excitation amplitude')
                return None
            new_args.insert(iA, Athr)
            args = new_args

        # Simulate model, save inf file and return file path
        data, tcomp = self.simulate(*args)
        meta = self.meta(*args)
        meta['tcomp'] = tcomp
        outpath = '{}/{}.pkl'.format(outdir, self.filecode(*args))
        with open(outpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': data}, fh)
        logger.debug('simulation data exported to "%s"', outpath)
        return outpath
