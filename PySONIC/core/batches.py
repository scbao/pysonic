# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2020-04-09 15:06:53

''' Utility functions used in simulations '''

import os
import abc
import csv
import logging
import lockfile
import numpy as np
import pandas as pd
import multiprocess as mp

from ..utils import logger, si_format


class Consumer(mp.Process):
    ''' Generic consumer process, taking tasks from a queue and outputing results in
        another queue.
    '''

    def __init__(self, queue_in, queue_out):
        mp.Process.__init__(self)
        self.queue_in = queue_in
        self.queue_out = queue_out
        logger.debug('Starting %s', self.name)

    def run(self):
        while True:
            nextTask = self.queue_in.get()
            if nextTask is None:
                logger.debug('Exiting %s', self.name)
                self.queue_in.task_done()
                break
            answer = nextTask()
            self.queue_in.task_done()
            self.queue_out.put(answer)
        return


class Worker:
    ''' Generic worker class calling a specific function with a given set of parameters. '''

    def __init__(self, wid, func, args, kwargs, loglevel):
        ''' Worker constructor.

            :param wid: worker ID
            :param func: function object
            :param args: list of method arguments
            :param kwargs: dictionary of optional method arguments
            :param loglevel: logging level
        '''
        self.id = wid
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.loglevel = loglevel

    def __call__(self):
        ''' Caller to the function with specific parameters. '''
        logger.setLevel(self.loglevel)
        return self.id, self.func(*self.args, **self.kwargs)


class Batch:
    ''' Generic interface to run batches of function calls. '''

    def __init__(self, func, queue):
        ''' Batch constructor.

            :param func: function object
            :param queue: list of list of function parameters
        '''
        self.func = func
        self.queue = queue

    def __call__(self, *args, **kwargs):
        ''' Call the internal run method. '''
        return self.run(*args, **kwargs)

    def getNConsumers(self):
        ''' Determine number of consumers based on queue length and number of available CPUs. '''
        return min(mp.cpu_count(), len(self.queue))

    def start(self):
        ''' Create tasks and results queues, and start consumers. '''
        mp.freeze_support()
        self.tasks = mp.JoinableQueue()
        self.results = mp.Queue()
        self.consumers = [Consumer(self.tasks, self.results) for i in range(self.getNConsumers())]
        for c in self.consumers:
            c.start()

    @staticmethod
    def resolve(params):
        if isinstance(params, list):
            args = params
            kwargs = {}
        elif isinstance(params, tuple):
            args, kwargs = params
        return args, kwargs

    def assign(self, loglevel):
        ''' Assign tasks to workers. '''
        for i, params in enumerate(self.queue):
            args, kwargs = self.resolve(params)
            worker = Worker(i, self.func, args, kwargs, loglevel)
            self.tasks.put(worker, block=False)

    def join(self):
        ''' Put all tasks to None and join the queue. '''
        for i in range(len(self.consumers)):
            self.tasks.put(None, block=False)
        self.tasks.join()

    def get(self):
        ''' Extract and re-order results. '''
        outputs, idxs = [], []
        for i in range(len(self.queue)):
            wid, out = self.results.get()
            outputs.append(out)
            idxs.append(wid)
        return [x for _, x in sorted(zip(idxs, outputs))]

    def stop(self):
        ''' Close tasks and results queues. '''
        self.tasks.close()
        self.results.close()

    def run(self, mpi=False, loglevel=logging.INFO):
        ''' Run batch with or without multiprocessing. '''
        if mpi:
            self.start()
            self.assign(loglevel)
            self.join()
            outputs = self.get()
            self.stop()
        else:
            outputs = []
            for params in self.queue:
                args, kwargs = self.resolve(params)
                outputs.append(self.func(*args, **kwargs))
        return outputs

    @staticmethod
    def createQueue(*dims):
        ''' Create a serialized 2D array of all parameter combinations for a series of individual
            parameter sweeps.

            :param dims: list of lists (or 1D arrays) of input parameters
            :return: list of parameters (list) for each simulation
        '''
        ndims = len(dims)
        dims_in = [dims[1], dims[0]]
        inds_out = [1, 0]
        if ndims > 2:
            dims_in += dims[2:]
            inds_out += list(range(2, ndims))
        queue = np.stack(np.meshgrid(*dims_in), -1).reshape(-1, ndims)
        queue = queue[:, inds_out]
        return queue.tolist()


class LogBatch(metaclass=abc.ABCMeta):
    ''' Generic interface to a simulation batch in with real-time input:output caching
        in a specific log file.
    '''

    delimiter = '\t'  # csv delimiter
    rtol = 1e-9
    atol = 1e-16

    def __init__(self, inputs, root='.'):
        ''' Construtor.

            :param inputs: array of batch inputs
            :param root: root for IO operations
        '''
        self.inputs = inputs
        self.root = root
        self.fpath = self.filepath()
        self.lock = lockfile.FileLock(self.fpath)

    @property
    def root(self):
        return self._root

    @root.setter
    def root(self, value):
        if not os.path.isdir(value):
            raise ValueError(f'{value} is not a valid directory')
        self._root = value

    @property
    @abc.abstractmethod
    def in_key(self):
        ''' Input key. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_keys(self):
        ''' Output keys. '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def suffix(self):
        ''' filename suffix '''
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def unit(self):
        ''' Input unit. '''
        raise NotImplementedError

    @property
    def in_label(self):
        ''' Input label. '''
        return f'{self.in_key} ({self.unit})'

    def rangecode(self, x, label, unit):
        ''' String describing a batch input range. '''
        bounds_str = si_format([x.min(), x.max()], 1, space="")
        return '{0}{1}{3}-{2}{3}_{4}'.format(label.replace(' ', '_'), *bounds_str, unit, x.size)

    @property
    def inputscode(self):
        ''' String describing the batch inputs. '''
        return self.rangecode(self.inputs, self.in_key, self.unit)

    @abc.abstractmethod
    def corecode(self):
        ''' String describing the batch core components. '''
        raise NotImplementedError

    def filecode(self):
        ''' String fully describing the batch. '''
        return f'{self.corecode()}_{self.inputscode}_{self.suffix}_results'

    def filename(self):
        ''' Batch associated filename. '''
        return f'{self.filecode()}.csv'

    def filepath(self):
        ''' Batch associated filepath. '''
        return os.path.join(self.root, self.filename())

    def createLogFile(self):
        ''' Create batch log file if it does not exist. '''
        if not os.path.isfile(self.fpath):
            logger.debug(f'creating batch log file: "{self.fpath}"')
            self.writeLabels()
        else:
            logger.debug(f'existing batch log file: "{self.fpath}"')

    def writeLabels(self):
        ''' Write the column labels of the batch log file. '''
        with open(self.fpath, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.delimiter)
            writer.writerow([self.in_label, *self.out_keys])

    def writeEntry(self, entry, outputs):
        ''' Write a new input:ouputs entry in the batch log file. '''
        self.lock.acquire()
        with open(self.fpath, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=self.delimiter)
            writer.writerow([entry, *outputs])
        self.lock.release()

    def getLogData(self):
        ''' Retrieve the batch log file data (inputs and outputs) as a dataframe. '''
        return pd.read_csv(self.fpath, sep=self.delimiter).sort_values(self.in_label)

    def getInput(self):
        ''' Retrieve the logged batch inputs as an array. '''
        return self.getLogData()[self.in_label].values

    def getOutput(self):
        ''' Retrieve the logged batch outputs as an array (if 1 key) or dataframe (if several). '''
        if len(self.out_keys) == 1:
            return self.getLogData()[self.out_keys[0]].values
        else:
            return pd.DataFrame({k: self.getLogData()[k].values for k in self.out_keys})

    def isEntry(self, value):
        ''' Check if a given input is logged in the batch log file. '''
        inputs = self.getInput()
        if len(inputs) == 0:
            return False
        imatches = np.where(np.isclose(inputs, value, rtol=self.rtol, atol=self.atol))[0]
        if len(imatches) == 0:
            return False
        return True

    @abc.abstractmethod
    def compute(self, x):
        ''' Compute the necessary output(s) for a given input. '''
        raise NotImplementedError

    def computeAndLog(self, x):
        ''' Compute output(s) and log new entry only if input is not already in the log file. '''
        if not self.isEntry(x):
            logger.debug(f'entry not found: "{x}"')
            self.writeEntry(x, self.compute(x))
        else:
            logger.debug(f'existing entry: "{x}"')
        return 0

    def run(self, mpi=False):
        ''' Run the batch and return the output(s). '''
        self.createLogFile()
        batch = Batch(self.computeAndLog, [[x] for x in self.inputs])
        batch.run(mpi=mpi, loglevel=logger.level)
        # for x in self.inputs:
        #     self.computeAndLog(x)
        return self.getOutput()
