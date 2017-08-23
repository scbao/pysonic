#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-01 16:35:43
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-06-07 16:14:54

""" Compute RMSE between charge profiles of NBLS output. """

import sys
import os
import pickle
import ntpath
import numpy as np

sys.path.append('C:/Users/admin/Google Drive/PhD/NICE model/NICEPython')
from NICE.utils import OpenFilesDialog, rmse, find_nearest
from NICE.constants import *

# Define options
pkl_root = "../../Output/test NBLS/"
t_offset = 10e-3  # s

# Select data files (PKL)
abs_root = os.path.abspath(pkl_root)
pkl_filepaths = OpenFilesDialog(abs_root, 'pkl')

# Check dialog output
if not pkl_filepaths:
    print('error: no input file')
elif len(pkl_filepaths) > 2:
    print('error: cannot compare more than 2 methods')
else:
    # Load data from file 1
    pkl_filename = ntpath.basename(pkl_filepaths[0])
    print('Loading data from "' + pkl_filename + '"')
    pkl_file = open(pkl_filepaths[0], 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    t1 = data['t']
    tstim1 = data['tstim']
    f1 = data['Fdrive']
    A1 = data['Adrive']
    Q1 = data['Qm'] * 1e2  # nC/cm2
    states1 = data['states']

    # Load data from file 2
    pkl_filename = ntpath.basename(pkl_filepaths[1])
    print('Loading data from "' + pkl_filename + '"')
    pkl_file = open(pkl_filepaths[1], 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    t2 = data['t']
    tstim2 = data['tstim']
    f2 = data['Fdrive']
    A2 = data['Adrive']
    Q2 = data['Qm'] * 1e2  # nC/cm2
    states2 = data['states']

    if tstim1 != tstim2 or f1 != f2 or A1 != A2:
        print('error: different stimulation conditions')
    else:
        print('comparing charge profiles')

        T = 1 / f1
        ttot = tstim1  # + toffset
        ncycles = int(ttot / T)
        tcycles1 = np.empty(ncycles)
        tcycles2 = np.empty(ncycles)
        Qcycles1 = np.empty(ncycles)
        Qcycles2 = np.empty(ncycles)
        i1 = 1
        i2 = 1
        icycle = 0
        while icycle < ncycles:
            tcycles1[icycle] = t1[i1]
            tcycles2[icycle] = t2[i2]
            Qcycles1[icycle] = Q1[i1]
            Qcycles2[icycle] = Q2[i2]
            if states1[i1 + 1] == 0:
                i1 += NPC_HH
            else:
                i1 += NPC_FULL
            if states2[i2 + 1] == 0:
                i2 += NPC_HH
            else:
                i2 += NPC_FULL
            icycle += 1

        t_error = rmse(tcycles1, tcycles2)
        print('method 1: rmse = %f us' % (t_error * 1e6))
        Q_error = rmse(Qcycles1, Qcycles2)
        print('method 1: rmse = %f nC/cm2' % Q_error)

        # determining optimal slices
        tslice = 5e-4
        ttot = tstim1 + t_offset
        slices = np.arange(0.0, ttot, tslice)
        nslices = slices.size
        print('%u comparison instants' % nslices)

        # determining real slices
        icomp1 = []
        icomp2 = []
        for i in range(nslices):
            (islice1, tslice1) = find_nearest(t1, slices[i])
            (islice2, tslice2) = find_nearest(t2, slices[i])
            icomp1.append(islice1)
            icomp2.append(islice2)

        # Comparing charge values
        Qcomp1 = Q1[icomp1]
        Qcomp2 = Q2[icomp2]
        Q_error = rmse(Qcomp1, Qcomp2)
        print('method 2: rmse = %f nC/cm2' % Q_error)
