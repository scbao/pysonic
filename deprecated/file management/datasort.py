#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2017-07-18 11:54:28
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-07 11:27:58

''' Re-organize data files '''

import os
import numpy as np

data_root_in = 'C:/Users/admin/Desktop/DATA2/TC/PRF1.50kHz'
data_root_out = 'C:/Users/admin/Desktop/DATA'

neurons = ['TC']
freqs = np.arange(850, 1001, 50)
tmpdir = 'tmp6'

amps = np.arange(50, 601, 50)
PRFs = [1.5]
DFs = np.arange(0.1, 0.91, 0.1)
durs = np.arange(10, 101, 10)

nfiles = len(neurons) * len(freqs) * len(amps) * len(PRFs) * len(durs) * len(DFs)
ifile = 0
for neuron in neurons:
    for f in freqs:
        filedir_in = '{}/{}'.format(data_root_in, tmpdir)
        filedir_out = '{}/{}/{:.0f}kHz'.format(data_root_out, neuron, f)
        # print(filedir_in)
        # print(filedir_out)
        for A in amps:
            for PRF in PRFs:
                for t in durs:
                    # ifile += 1
                    # CWcode = 'sim_{}_CW_32nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms'.format(neuron, f, A, t)
                    # filepath_in = '{}/{}_effective.pkl'.format(filedir_in, CWcode)
                    # filepath_out = '{}/CW/{}_effective.pkl'.format(filedir_out, CWcode)
                    # print('renaming file {}/{}'.format(ifile, nfiles))
                    # print(filepath_in)
                    # print(filepath_out)
                    # if os.path.isfile(filepath_in):
                    #     os.rename(filepath_in, filepath_out)
                    for DF in DFs:
                        ifile += 1
                        PWcode = 'sim_{}_PW_32nm_{:.0f}kHz_{:.0f}kPa_{:.0f}ms_PRF{:.2f}kHz_DF{:.2f}'.format(neuron, f, A, t, PRF, DF)
                        filepath_in = '{}/{}_effective.pkl'.format(filedir_in, PWcode)
                        filepath_out = '{}/PRF{:.2f}kHz/{}_effective.pkl'.format(filedir_out, PRF, PWcode)
                        print('renaming file {}/{}'.format(ifile, nfiles))
                        print(filepath_in)
                        print(filepath_out)
                        if os.path.isfile(filepath_in):
                            os.rename(filepath_in, filepath_out)
