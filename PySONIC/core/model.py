#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-03 11:53:04
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-09 21:35:40

import os
import pickle
import abc
import inspect
import numpy as np

from .batches import createQueue
from ..utils import logger, loadData


class Model(metaclass=abc.ABCMeta):
    ''' Generic model interface. '''

    @property
    @abc.abstractmethod
    def tscale(self):
        ''' Relevant temporal scale of the model. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def simkey(self):
        ''' Keyword used to characterize simulations made with the model. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def params(self):
        ''' Gather all model parameters in a dictionary '''
        def toAvoid(p):
            return (p.startswith('__') and p.endswith('__')) or p.startswith('_abc_')
        class_attrs = inspect.getmembers(self.__class__, lambda a: not(inspect.isroutine(a)))
        inst_attrs = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)))
        class_attrs = [a for a in class_attrs if not toAvoid(a[0])]
        inst_attrs = [a for a in inst_attrs if not toAvoid(a[0]) and a not in class_attrs]
        params_dict = {a[0]: a[1] for a in class_attrs + inst_attrs}
        return params_dict

    @property
    @abc.abstractmethod
    def filecodes(self, *args):
        raise NotImplementedError

    def filecode(self, *args):
        codes = self.filecodes(*args).values()
        return '_'.join([x for x in codes if x is not None])

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
    def meta(self, *args):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def simulate(self, *args, **kwargs):
        raise NotImplementedError

    def simQueue(self, *args):
        ''' Create a simulation queue from a combination of simulation parameters. '''
        return createQueue(*args)

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

        # Simulate model, save file and return file path
        data, tcomp = self.simulate(*args)
        meta = self.meta(*args)
        meta['tcomp'] = tcomp
        fpath = '{}/{}.pkl'.format(outdir, self.filecode(*args))
        with open(fpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': data}, fh)
        logger.debug('simulation data exported to "%s"', fpath)
        return fpath

    def load(self, outdir, *args):
        ''' Load output data for a specific parameters combination. '''

        # Get file path from simulation parameters
        fpath = '{}/{}.pkl'.format(outdir, self.filecode(*args))

        # If output file does not exist, run simulation to generate it
        if not os.path.isfile(fpath):
            self.runAndSave(outdir, *args)

        # Return data and meta-data
        return loadData(fpath)
