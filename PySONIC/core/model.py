# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-03 11:53:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-07-15 20:35:46

import os
from functools import wraps
from inspect import signature, getdoc
import pickle
import abc
import inspect
import numpy as np

from .batches import Batch
from ..utils import logger, loadData, timer, si_format, plural


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

    @classmethod
    def description(cls):
        return getdoc(cls).split('\n', 1)[0].strip()

    @staticmethod
    @abc.abstractmethod
    def inputs():
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

    @classmethod
    @abc.abstractmethod
    def getPltVars(self, *args, **kwargs):
        ''' Return a dictionary with information about all plot variables related to the model. '''
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def getPltScheme(self):
        ''' Return a dictionary model plot variables grouped by context. '''
        raise NotImplementedError

    @staticmethod
    def checkOutputDir(queue, outputdir):
        ''' Check if an outputdir is provided in input arguments, and if so, add it as
            the first element of each item in the returned queue.
        '''
        if outputdir is not None:
            for item in queue:
                item.insert(0, outputdir)
        else:
            if len(queue) > 5:
                logger.warning('Running more than 5 simulations without file saving')
        return queue

    @classmethod
    def simQueue(cls, *args, outputdir=None):
        return cls.checkOutputDir(Batch.createQueue(*args), outputdir)

    @staticmethod
    @abc.abstractmethod
    def checkInputs(self, *args):
        ''' Check the validity of simulation input parameters. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def derivatives(self, *args, **kwargs):
        ''' Compute ODE derivatives for a specific set of ODE states and external parameters. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def simulate(self, *args, **kwargs):
        ''' Simulate the model's differential system for specific input parameters
            and return output data in a dataframe. '''
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def meta(self, *args):
        ''' Return an informative dictionary about model and simulation parameters. '''
        raise NotImplementedError

    @staticmethod
    def addMeta(simfunc):
        ''' Add an informative dictionary about model and simulation parameters to simulation output '''

        @wraps(simfunc)
        def wrapper(self, *args, **kwargs):
            data, tcomp = timer(simfunc)(self, *args, **kwargs)
            logger.debug('completed in %ss', si_format(tcomp, 1))

            # Add keyword arguments from simfunc signature if not provided
            bound_args = inspect.signature(simfunc).bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            target_args = dict(bound_args.arguments)

            # Try to retrieve meta information
            try:
                meta_params = [target_args[k] for k in inspect.signature(self.meta).parameters.keys()]
                meta = self.meta(*meta_params)
            except KeyError:
                meta = {}

            # Add computation time to it
            meta['tcomp'] = tcomp

            # Return data with meta dict
            return data, meta

        return wrapper

    @staticmethod
    def logNSpikes(simfunc):
        ''' Log number of detected spikes on charge profile of simulation output. '''
        @wraps(simfunc)
        def wrapper(self, *args, **kwargs):
            data, meta = simfunc(self, *args, **kwargs)
            nspikes = self.getNSpikes(data)
            logger.debug('{} spike{} detected'.format(nspikes, plural(nspikes)))
            return data, meta

        return wrapper

    @staticmethod
    def checkTitrate(argname):
        ''' If no (None) amplitude provided in the list of input parameters,
            perform a titration to find the threshold amplitude and add it to the list.
        '''

        def wrapper_with_args(simfunc):

            @wraps(simfunc)
            def wrapper(self, *args, **kwargs):
                func_args = list(inspect.signature(simfunc).parameters.keys())[1:]
                iarg = func_args.index(argname)
                if args[iarg] is None:
                    new_args = [x for x in args if x is not None]
                    new_args = list(args[:])
                    del new_args[iarg]
                    Athr = self.titrate(*new_args)
                    if np.isnan(Athr):
                        logger.error('Could not find threshold excitation amplitude')
                        return None
                    new_args.insert(iarg, Athr)
                    args = new_args
                return simfunc(self, *args, **kwargs)

            return wrapper

        return wrapper_with_args

    def simAndSave(self, outdir, *args):
        ''' Simulate the model and save the results in a specific output directory. '''
        data, meta = self.simulate(*args)
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
            self.simAndSave(outdir, *args)
        return loadData(fpath)
