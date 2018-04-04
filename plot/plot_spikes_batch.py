# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2018-04-04 11:49:07
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-04-04 12:06:32

''' Plot detected spikes on charge profiles. '''

import sys
import os
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt

from PointNICE.utils import logger, OpenFilesDialog, InputError
from PointNICE.solvers import findPeaks
from PointNICE.constants import *

# Set logging level
logger.setLevel(logging.INFO)

# Set plot parameters
fs = 15  # font size
lw = 2  # linewidth

# Select data files
pkl_filepaths, pkl_dir = OpenFilesDialog('pkl')
if not pkl_filepaths:
    logger.error('No input file')
    sys.exit(1)

try:
    for fpath in pkl_filepaths:

        # Load charge profile from file
        fname = os.path.basename(fpath)
        logger.info('Loading data from "%s" file', fname)
        with open(fpath, 'rb') as fh:
            frame = pickle.load(fh)
        df = frame['data']
        t = df['t'].values
        Qm = df['Qm'].values
        dt = t[1] - t[0]
        indexes = np.arange(t.size)
        mpd = int(np.ceil(SPIKE_MIN_DT / dt))

        ipeaks, prominences, widths, ibounds = findPeaks(Qm, mph=SPIKE_MIN_QAMP, mpd=mpd,
                                                         mpp=SPIKE_MIN_QPROM)
        if ipeaks is not None:
            widths *= dt
            tleftbounds = np.interp(ibounds[:, 0], indexes, t)
            trightbounds = np.interp(ibounds[:, 1], indexes, t)

        # Plot results
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title(os.path.splitext(fname)[0], fontsize=fs)
        ax.set_xlabel('time (ms)', fontsize=fs)
        ax.set_ylabel('charge\ (nC/cm2)', fontsize=fs)
        ax.plot(t * 1e3, Qm * 1e5, color='C0', label='trace', linewidth=lw)
        if ipeaks is not None:
            ax.scatter(t[ipeaks] * 1e3, Qm[ipeaks] * 1e5 + 3, color='k', label='peaks', marker='v')
            for i in range(len(ipeaks)):
                ax.plot(np.array([t[ipeaks[i]]] * 2) * 1e3,
                        np.array([Qm[ipeaks[i]], Qm[ipeaks[i]] - prominences[i]]) * 1e5,
                        color='C1', label='prominences' if i == 0 else '')
                ax.plot(np.array([tleftbounds[i], trightbounds[i]]) * 1e3,
                        np.array([Qm[ipeaks[i]] - 0.5 * prominences[i]] * 2) * 1e5, color='C2',
                        label='widths' if i == 0 else '')
        ax.legend()

    plt.show()

except InputError as err:
    logger.error(err)
    sys.exit(1)