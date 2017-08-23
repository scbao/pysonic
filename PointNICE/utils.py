#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-09-19 22:30:46
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-22 18:39:31

""" Definition of generic utility functions used in other modules """

from enum import Enum
from functools import partial
import os
import shutil
import logging
import tkinter as tk
from tkinter import filedialog
from openpyxl import load_workbook
import numpy as np
import yaml
import matplotlib.pyplot as plt


# Get package logger
logger = logging.getLogger('PointNICE')


class PmCompMethod(Enum):
    """ Enum: types of computation method for the intermolecular pressure """
    direct = 1
    predict = 2


def LoadParamsFile(filename):
    """ Load a dictionary of parameters for the BLS model from an external yaml file.

        :param filename: name of the input file
        :return: parameters dictionary
    """

    logger.info('Loading parameters from "%s"', filename)
    with open(filename, 'r') as f:
        stream = f.read()
    params = yaml.load(stream)
    return ParseNestedDict(params)


LoadParams = partial(LoadParamsFile, filename=os.path.split(__file__)[0] + '/params.yaml')


def getLookupDir():
    """ Return the location of the directory holding lookups files.

        :return: absolute path to the directory
    """
    this_dir, _ = os.path.split(__file__)
    return this_dir + '/lookups'


def ParseNestedDict(dict_in):
    """ Loop through a nested dictionary object and convert all string fields
        to floats.
    """
    for key, value in dict_in.items():
        if isinstance(value, dict):  # If value itself is dictionary
            dict_in[key] = ParseNestedDict(value)
        elif isinstance(dict_in[key], str):
            dict_in[key] = float(dict_in[key])
    return dict_in


def OpenFilesDialog(filetype, dirname=''):
    """ Open a FileOpenDialogBox to select one or multiple file.

        The default directory and file type are given.

        :param dirname: default directory
        :param filetype: default file type
        :return: tuple of full paths to the chosen filenames
    """
    root = tk.Tk()
    root.withdraw()
    filenames = filedialog.askopenfilenames(filetypes=[(filetype + " files", '.' + filetype)],
                                            initialdir=dirname)
    if filenames:
        par_dir = os.path.abspath(os.path.join(filenames[0], os.pardir))
    else:
        par_dir = None
    return (filenames, par_dir)


def SaveFigDialog(dirname, filename):
    """ Open a FileSaveDialogBox to set the directory and name
        of the figure to be saved.

        The default directory and filename are given, and the
        default extension is ".pdf"

        :param dirname: default directory
        :param filename: default filename
        :return: full path to the chosen filename
    """
    root = tk.Tk()
    root.withdraw()
    filename_out = filedialog.asksaveasfilename(defaultextension=".pdf", initialdir=dirname,
                                                initialfile=filename)
    return filename_out


def xlslog(filename, sheetname, data):
    """ Append log data on a new row to specific sheet of excel workbook.

        :param filename: absolute or relative path to the Excel workbook
        :param sheetname: name of the Excel spreadsheet to which data is appended
        :param data: data structure to be added to specific columns on a new row
        :return: boolean indicating success (1) or failure (0) of operation
    """

    try:
        wb = load_workbook(filename)
        ws = wb[sheetname]
        keys = data.keys()
        i = 1
        row_data = {}
        for k in keys:
            row_data[k] = data[k]
            i += 1
        ws.append(row_data)
        wb.save(filename)
        return 1
    except PermissionError:
        # If file cannot be accessed for writing because already opened
        logger.error('Cannot write to "%s". Close the file and type "Y"', filename)
        user_str = input()
        if user_str in ['y', 'Y']:
            return xlslog(filename, sheetname, data)
        else:
            return 0


def ImportExcelCol(filename, sheetname, colstr, startrow):
    """ Load a specific column of an excel workbook as a numpy array.

        :param filename: absolute or relative path to the Excel workbook
        :param sheetname: name of the Excel spreadsheet to which data is appended
        :param colstr: string of the column to import
        :param startrow: index of the first row to consider
        :return: 1D numpy array with the column data
    """

    wb = load_workbook(filename, read_only=True)
    ws = wb.get_sheet_by_name(sheetname)
    range_start_str = colstr + str(startrow)
    range_stop_str = colstr + str(ws.max_row)
    tmp = np.array([[i.value for i in j] for j in ws[range_start_str:range_stop_str]])
    return tmp[:, 0]


def ConstructMatrix(serialized_inputA, serialized_inputB, serialized_output):
    """ Construct a 2D output matrix from serialized input.

        :param serialized_inputA: serialized input variable A
        :param serialized_inputB: serialized input variable B
        :param serialized_output: serialized output variable
        :return: 4-tuple with vectors of unique values of A (m) and B (n),
            output variable 2D matrix (m,n) and number of holes in the matrix
    """

    As = np.unique(serialized_inputA)
    Bs = np.unique(serialized_inputB)
    nA = As.size
    nB = Bs.size

    output = np.zeros((nA, nB))
    output[:] = np.NAN
    nholes = 0
    for i in range(nA):
        iA = np.where(serialized_inputA == As[i])
        for j in range(nB):
            iB = np.where(serialized_inputB == Bs[j])
            iMatch = np.intersect1d(iA, iB)
            if iMatch.size == 0:
                nholes += 1
            elif iMatch.size > 1:
                logger.warning('Identical serialized inputs with values (%f, %f)', As[i], Bs[j])
            else:
                iMatch = iMatch[0]
                output[i, j] = serialized_output[iMatch]
    return (As, Bs, output, nholes)


def rmse(x1, x2):
    """ Compute the root mean square error between two 1D arrays """
    return np.sqrt(((x1 - x2) ** 2).mean())


def rsquared(x1, x2):
    ''' compute the R-squared coefficient between two 1D arrays '''
    residuals = x1 - x2
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((x1 - np.mean(x1))**2)
    return 1 - (ss_res / ss_tot)


def DownSample(t_dense, y, nsparse):
    """ Decimate periodic signals to a specified number of samples."""

    if(y.ndim) > 1:
        nsignals = y.shape[0]
    else:
        nsignals = 1
        y = np.array([y])

    # determine time step and period of input signal
    T = t_dense[-1] - t_dense[0]
    dt_dense = t_dense[1] - t_dense[0]

    # resample time vector linearly
    t_ds = np.linspace(t_dense[0], t_dense[-1], nsparse)

    # create MAV window
    nmav = int(0.03 * T / dt_dense)
    if nmav % 2 == 0:
        nmav += 1
    mav = np.ones(nmav) / nmav

    # determine signals padding
    npad = int((nmav - 1) / 2)

    # determine indexes of sampling on convolved signals
    ids = np.round(np.linspace(0, t_dense.size - 1, nsparse)).astype(int)

    y_ds = np.empty((nsignals, nsparse))

    # loop through signals
    for i in range(nsignals):
        # pad, convolve and resample
        pad_left = y[i, -(npad + 2):-2]
        pad_right = y[i, 1:npad + 1]
        y_ext = np.concatenate((pad_left, y[i, :], pad_right), axis=0)
        y_mav = np.convolve(y_ext, mav, mode='valid')
        y_ds[i, :] = y_mav[ids]

    if nsignals == 1:
        y_ds = y_ds[0, :]

    return (t_ds, y_ds)


def Pressure2Intensity(p, rho, c):
    """ Return the spatial peak, pulse average acoustic intensity (ISPPA)
        associated with the specified pressure amplitude.

        :param p: pressure amplitude (Pa)
        :param rho: medium density (kg/m3)
        :param c: speed of sound in medium (m/s)
        :return: spatial peak, pulse average acoustic intensity (W/m2)
    """
    return p**2 / (2 * rho * c)


def Intensity2Pressure(I, rho, c):
    """ Return the pressure amplitude associated with the specified
        spatial peak, pulse average acoustic intensity (ISPPA).

        :param I: spatial peak, pulse average acoustic intensity (W/m2)
        :param rho: medium density (kg/m3)
        :param c: speed of sound in medium (m/s)
        :return: pressure amplitude (Pa)
    """
    return np.sqrt(2 * rho * c * I)


def detectPeaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                kpsh=False, valley=False, ax=None):
    """ Detect peaks in data based on their amplitude and inter-peak distance. """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    return ind


def detectPeaksTime(t, y, mph, mtd):
    """ Extension of the detectPeaks function to detect peaks in data based on their
        amplitude and time difference, with a non-uniform time vector.

        :param t: time vector (not necessarily uniform)
        :param y: signal
        :param mph: minimal peak height
        :param mtd: minimal time difference
        :return: array of peak indexes
    """

    # Detect peaks on signal with no restriction on inter-peak distance
    raw_indexes = detectPeaks(y, mph, mpd=1)

    if raw_indexes.size > 0:

        # Filter relevant peaks with temporal distance
        n_raw = raw_indexes.size
        filtered_indexes = np.array([raw_indexes[0]])
        for i in range(1, n_raw):
            i1 = filtered_indexes[-1]
            i2 = raw_indexes[i]
            if t[i2] - t[i1] < mtd:
                if y[i2] > y[i1]:
                    filtered_indexes[-1] = i2
            else:
                filtered_indexes = np.append(filtered_indexes, i2)

        # Return peak indexes
        return filtered_indexes
    else:
        return None


def detectSpikes(t, Qm, min_amp, min_dt):
    ''' Detect spikes on a charge density signal, and
        return their number, latency and rate.

        :param t: time vector (s)
        :param Qm: charge density vector (C/m2)
        :param min_amp: minimal charge amplitude to detect spikes (C/m2)
        :param min_dt: minimal time interval between 2 spikes (s)
        :return: 3-tuple with number of spikes, latency (s) and spike rate (sp/s)
    '''
    i_spikes = detectPeaksTime(t, Qm, min_amp, min_dt)
    if i_spikes is not None:
        latency = t[i_spikes[0]]  # s
        n_spikes = i_spikes.size
        if n_spikes > 1:
            first_to_last_spike = t[i_spikes[-1]] - t[i_spikes[0]]  # s
            spike_rate = n_spikes / first_to_last_spike  # spikes/s
        else:
            spike_rate = 'N/A'
    else:
        latency = 'N/A'
        spike_rate = 'N/A'
        n_spikes = 0
    return (n_spikes, latency, spike_rate)


class InteractiveLegend(object):
    """ Class defining an interactive matplotlib legend, where lines visibility can
    be toggled by simply clicking on the corresponding legend label. Other graphic
    objects can also be associated to the toggle of a specific line

    Adapted from:
    http://stackoverflow.com/questions/31410043/hiding-lines-after-showing-a-pyplot-figure
    """

    def __init__(self, legend, aliases):
        self.legend = legend
        self.fig = legend.axes.figure
        self.lookup_artist, self.lookup_handle = self._build_lookups(legend)
        self._setup_connections()
        self.handles_aliases = aliases
        self.update()

    def _setup_connections(self):
        for artist in self.legend.texts + self.legend.legendHandles:
            artist.set_picker(10)  # 10 points tolerance

        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

    def _build_lookups(self, legend):
        ''' Method of the InteractiveLegend class building
            the legend lookups. '''

        labels = [t.get_text() for t in legend.texts]
        handles = legend.legendHandles
        label2handle = dict(zip(labels, handles))
        handle2text = dict(zip(handles, legend.texts))

        lookup_artist = {}
        lookup_handle = {}
        for artist in legend.axes.get_children():
            if artist.get_label() in labels:
                handle = label2handle[artist.get_label()]
                lookup_handle[artist] = handle
                lookup_artist[handle] = artist
                lookup_artist[handle2text[handle]] = artist

        lookup_handle.update(zip(handles, handles))
        lookup_handle.update(zip(legend.texts, handles))

        return lookup_artist, lookup_handle

    def on_pick(self, event):
        handle = event.artist
        if handle in self.lookup_artist:
            artist = self.lookup_artist[handle]
            artist.set_visible(not artist.get_visible())
            self.update()

    def update(self):
        for artist in self.lookup_artist.values():
            handle = self.lookup_handle[artist]
            if artist.get_visible():
                handle.set_visible(True)
                if artist in self.handles_aliases:
                    for al in self.handles_aliases[artist]:
                        al.set_visible(True)
            else:
                handle.set_visible(False)
                if artist in self.handles_aliases:
                    for al in self.handles_aliases[artist]:
                        al.set_visible(False)
        self.fig.canvas.draw()


    def show(self):
        ''' showing the interactive legend '''

        plt.show()


def find_nearest(array, value):
    ''' Find nearest element in 1D array. '''

    idx = (np.abs(array - value)).argmin()
    return (idx, array[idx])


def rescale(x, lb, ub, lb_new=0, ub_new=1):
    ''' Rescale a value to a specific interval by linear transformation. '''

    xnorm = (x - lb) / (ub - lb)
    return xnorm * (ub_new - lb_new) + lb_new


def printPct(pct, precision):
    print(('{:.' + str(precision) + 'f}%').format(pct), end='', flush=True)
    print('\r' * (precision + 3), end='')


def LennardJones(x, beta, alpha, C, m, n):
    """ Generic expression of a Lennard-Jones function, adapted for the context of
        symmetric deflection (distance = 2x).

        :param x: deflection (i.e. half-distance)
        :param beta: x-shifting factor
        :param alpha: x-scaling factor
        :param C: y-scaling factor
        :param m: exponent of the repulsion term
        :param n: exponent of the attraction term
        :return: Lennard-Jones potential at given distance (2x)
    """
    return C * (np.power((alpha / (2 * x + beta)), m) - np.power((alpha / (2 * x + beta)), n))



def getPatchesLoc(t, states):
    ''' Determine the location of stimulus patches.

        :param t: simulation time vector (s).
        :param states: a vector of stimulation state (ON/OFF) at each instant in time.
        :return: 3-tuple with number of patches, timing of STIM-ON an STIM-OFF instants.
    '''

    # Compute states derivatives and identify bounds indexes of pulses
    dstates = np.diff(states)
    ipatch_on = np.insert(np.where(dstates > 0.0)[0] + 1, 0, 0)
    ipatch_off = np.where(dstates < 0.0)[0]

    # Get time instants for pulses ON and OFF
    npatches = ipatch_on.size
    tpatch_on = t[ipatch_on]
    tpatch_off = t[ipatch_off]

    # return 3-tuple with #patches, pulse ON and pulse OFF instants
    return (npatches, tpatch_on, tpatch_off)


def CheckBatchLog(batch_type):
    ''' Determine batch directory, and add a log file to the directory if it is absent.

        :param batch_type: name of the log file to search for
        :return: 2-tuple with full paths to batch directory and log file
    '''

    # Get batch directory from user
    root = tk.Tk()
    root.withdraw()
    batch_dir = filedialog.askdirectory()
    assert batch_dir, 'No batch directory chosen'

    # Check presence of log file in batch directory
    logdst = batch_dir + '/log.xlsx'
    log_in_dir = os.path.isfile(logdst)

    # If no log file, copy template in directory
    if not log_in_dir:

        # Determine log template from batch type
        if batch_type == 'mech':
            logfile = 'log_mech.xlsx'
        elif batch_type == 'elec':
            logfile = 'log_elec.xlsx'
        else:
            raise ValueError('Unknown batch type', batch_type)
        this_dir, _ = os.path.split(__file__)
        # par_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
        logsrc = this_dir + '/templates/' + logfile

        # Copy template
        shutil.copy2(logsrc, logdst)

    return (batch_dir, logdst)
