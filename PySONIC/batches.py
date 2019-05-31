# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-08-22 14:33:04
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-31 15:09:27

''' Utility functions used in simulations '''

import os
import lockfile
import logging
import multiprocessing as mp
import numpy as np
import pandas as pd

from .utils import logger


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


class Worker():
    ''' Generic worker class calling a specific object's method with a given set of parameters. '''

    def __init__(self, wid, obj, method_str, params, loglevel):
        ''' Worker constructor.

            :param wid: worker ID
            :param obj: object containing the method to call
            :param method_str: name of the method to call
            :param params: list of method parameters
            :param loglevel: logging level
        '''
        self.id = wid
        self.obj = obj
        self.method_str = method_str
        self.params = params
        self.loglevel = loglevel

    def __call__(self):
        ''' Caller to the specific object method. '''
        logger.setLevel(self.loglevel)
        return self.id, getattr(self.obj, self.method_str)(*self.params)


def createQueue(dims):
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


# def runAndSave(self, outdir, Fdrive, Adrive, Qm):
#     data, tcomp = self.simulate(Fdrive, Adrive, Qm)
#     meta = self.meta(Fdrive, Adrive, Qm)
#     meta['tcomp'] = tcomp
#     simcode = self.filecode(Fdrive, Adrive, Qm)
#     outpath = '{}/{}.pkl'.format(outdir, simcode)
#     with open(outpath, 'wb') as fh:
#         pickle.dump({'meta': meta, 'data': data}, fh)
#     logger.debug('simulation data exported to "%s"', outpath)
#     return outpath


def runBatch(obj, method_str, queue, extra_params=[], mpi=False,
             loglevel=logging.INFO):
    ''' Run batch of simulations of a given object for various combinations of stimulation parameters.

        :param queue: array of all stimulation parameters combinations
        :param mpi: boolean stating whether or not to use multiprocessing
    '''
    nsims = len(queue)

    if mpi:
        mp.freeze_support()

        tasks = mp.JoinableQueue()
        results = mp.Queue()
        nconsumers = min(mp.cpu_count(), nsims)
        consumers = [Consumer(tasks, results) for i in range(nconsumers)]
        for w in consumers:
            w.start()

    # Run simulations
    outputs = []
    for i, stim_params in enumerate(queue):
        params = extra_params + stim_params
        if mpi:
            worker = Worker(i, obj, method_str, params, loglevel)
            tasks.put(worker, block=False)
        else:
            outputs.append(getattr(obj, method_str)(*params))

    if mpi:
        for i in range(nconsumers):
            tasks.put(None, block=False)
        tasks.join()
        idxs = []
        for i in range(nsims):
            wid, out = results.get()
            outputs.append(out)
            idxs.append(wid)
        outputs = [x for _, x in sorted(zip(idxs, outputs))]

        # Close tasks and results queues
        tasks.close()
        results.close()

    return outputs


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
