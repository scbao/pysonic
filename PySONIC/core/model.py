# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-03 11:53:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-26 19:15:46

import os
from inspect import signature
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
        ''' String representation. '''
        raise NotImplementedError

    def params(self):
        ''' Return a dictionary of all model parameters (class and instance attributes) '''
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
    def inputs(self):
        ''' Return an informative dictionary on input variables used to simulate the model. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def filecodes(self, *args):
        ''' Return a dictionary of string-encoded inputs used for file naming. '''
        raise NotImplementedError

    def filecode(self, *args):
        ''' Generate file code given a specific combination of model input parameters. '''

        # If meta dictionary was passed, generate inputs list from it
        if len(args) == 1 and isinstance(args[0], dict):
            meta = args[0]
            meta.pop('tcomp', None)
            nparams = len(signature(self.meta).parameters)
            args = list(meta.values())[-nparams:]

        # Create file code by joining string-encoded inputs with underscores
        codes = self.filecodes(*args).values()
        return '_'.join([x for x in codes if x is not None])

    @property
    @abc.abstractmethod
    def getPltVars(self, *args, **kwargs):
        ''' Return a dictionary with information about all plot variables related to the model. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def getPltScheme(self):
        ''' Return a dictionary model plot variables grouped by context. '''
        raise NotImplementedError

    def simQueue(self, *args, outputdir=None):
        ''' Create a simulation queue from a combination of simulation parameters. '''
        queue = createQueue(*args)
        if outputdir is not None:
            for item in queue:
                item.insert(0, outputdir)
        return queue

    @property
    @abc.abstractmethod
    def checkInputs(self, *args):
        ''' Check the validity of simulation input parameters. '''
        raise NotImplementedError

    # @property
    # @abc.abstractmethod
    # def derivatives(self, *args, **kwargs):
    #     ''' Compute ODE derivatives for a specific set of ODE states and external parameters. '''
    #     raise NotImplementedError

    @property
    @abc.abstractmethod
    def simulate(self, *args, **kwargs):
        ''' Simulate the model's differential system for specific input parameters
            and return output data in a dataframe. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def meta(self, *args):
        ''' Return an informative dictionary about model and simulation parameters. '''
        raise NotImplementedError

    def checkAmplitude(self, args):
        ''' If no (None) amplitude provided in the list of input parameters,
            perform a titration to find the threshold amplitude and add it to the list.
        '''
        if None in args:
            iA = args.index(None)
            new_args = [x for x in args if x is not None]
            Athr = self.titrate(*new_args)
            if np.isnan(Athr):
                logger.error('Could not find threshold excitation amplitude')
                return None
            new_args.insert(iA, Athr)
            args = new_args
        return args

    def runAndSave(self, outdir, *args):
        ''' Simulate the model for specific parameters and save the results
            in a specific output directory. '''
        args = self.checkAmplitude(args)
        data, tcomp = self.simulate(*args)
        meta = self.meta(*args)
        meta['tcomp'] = tcomp
        fpath = '{}/{}.pkl'.format(outdir, self.filecode(*args))
        with open(fpath, 'wb') as fh:
            pickle.dump({'meta': meta, 'data': data}, fh)
        logger.debug('simulation data exported to "%s"', fpath)
        return fpath

    def getOutput(self, outdir, *args):
        ''' Get simulation output data for a specific parameters combination, by looking
            for an output file into a specific directory.

            If a corresponding output file is not found in the specified directory, the model
            is first run and results are saved in the output file.
        '''
        fpath = '{}/{}.pkl'.format(outdir, self.filecode(*args))
        if not os.path.isfile(fpath):
            self.runAndSave(outdir, *args)
        return loadData(fpath)
