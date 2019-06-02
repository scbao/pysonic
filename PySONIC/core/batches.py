# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-02 14:20:06

''' Utility functions used in simulations '''

import os
import lockfile
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd

from ..utils import logger


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

    def __init__(self, wid, func, params, loglevel):
        ''' Worker constructor.

            :param wid: worker ID
            :param func: function object
            :param params: list of method parameters
            :param loglevel: logging level
        '''
        self.id = wid
        self.func = func
        self.params = params
        self.loglevel = loglevel

    def __call__(self):
        ''' Caller to the function with specific parameters. '''
        logger.setLevel(self.loglevel)
        return self.id, self.func(*self.params)


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

    def assign(self, loglevel):
        ''' Assign tasks to workers. '''
        for i, params in enumerate(self.queue):
            worker = Worker(i, self.func, params, loglevel)
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
            outputs = [self.func(*params) for params in self.queue]
        return outputs


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


def xlslog(filepath, logentry, sheetname='Data'):
    ''' Append log data on a new row to specific sheet of excel workbook, using a lockfile
        to avoid read/write errors between concurrent processes.

        :param filepath: absolute or relative path to the Excel workbook
        :param logentry: log entry (dictionary) to add to log file
        :param sheetname: name of the Excel spreadsheet to which data is appended
        :return: boolean indicating success (1) or failure (0) of operation
    '''

    # Parse log dataframe from Excel file if it exists, otherwise create new one
    if not os.path.isfile(filepath):
        df = pd.DataFrame(columns=logentry.keys())
    else:
        df = pd.read_excel(filepath, sheet_name=sheetname)

    # Add log entry to log dataframe
    df = df.append(pd.Series(logentry), ignore_index=True)

    # Write log dataframe to Excel file
    try:
        lock = lockfile.FileLock(filepath)
        lock.acquire()
        writer = pd.ExcelWriter(filepath)
        df.to_excel(writer, sheet_name=sheetname, index=False)
        writer.save()
        lock.release()
        return 1
    except PermissionError:
        # If file cannot be accessed for writing because already opened
        logger.warning('Cannot write to "%s". Close the file and type "Y"', filepath)
        user_str = input()
        if user_str in ['y', 'Y']:
            return xlslog(filepath, logentry, sheetname)
        else:
            return 0
