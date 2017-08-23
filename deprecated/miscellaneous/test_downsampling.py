#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-08 21:26:06
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-06-08 11:29:54

""" Test different signal downsampling strategies """

import sys
import pickle
import ntpath
import time
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('C:/Users/admin/Google Drive/PhD/NICE model/NICEPython')
import NICE.core as nblscore
from NICE.utils import OpenFilesDialog, DownSample

# Define options
pkl_root = "../Output/test HH/"
plt_root = "../Output/test HH/"
plt_show = 0
plt_save = 1
plt_askbeforesave = 0

t_unit = 'us'
t_factor = 1e6
ind_tmax = -1


# Select data files (PKL)
pkl_filepaths = OpenFilesDialog(pkl_root, 'pkl')

# Check dialog output
if not pkl_filepaths:
    print('error: no input file')
else:
    # Loop through data files
    for pkl_filepath in pkl_filepaths:

        # Get code from file name
        pkl_filename = ntpath.basename(pkl_filepath)
        filecode = pkl_filename[0:-4]

        # Load data
        print('Loading data from "' + pkl_filename + '"')
        pkl_file = open(pkl_filepath, 'rb')
        data = pickle.load(pkl_file)

        # Extract variables
        print('Extracting variables')
        Adrive = data['Adrive']
        Fdrive = data['Fdrive']
        phi = data['phi']
        dt = data['dt']
        t = data['t']
        U = data['U']
        Z = data['Z']
        Vm = data['Vm']
        a = data['a']
        d = data['d']
        params = data['params']
        geom = {"a": a, "d": d}

        npc = int(1 / (Fdrive * dt))
        print(npc, t.size)

        # Create dummy BLS instance to use functions
        nbls = nblscore.NeuronalBilayerSonophore(geom, params, Fdrive, False)

        # Compute membrane capacitance density
        print("computing capacitance")
        # Cm = np.array([nbls.Capct(ZZ) for ZZ in Z])
        Cm = Z

        # Filter 1: N-moving average
        t0 = time.time()
        N = int(0.03 / (dt * Fdrive))
        if N % 2 == 0:
            N += 1
        npad = int((N - 1) / 2)
        Cm_begin_padding = Cm[-(npad + 2):-2]
        Cm_end_padding = Cm[1:npad + 1]
        Cm_ext = np.concatenate((Cm_begin_padding, Cm, Cm_end_padding), axis=0)
        mav = np.ones(N) / N
        Cm_filtMAV = np.convolve(Cm_ext, mav, mode='valid')
        print('Moving-average method: ' + '{:.2e}'.format(time.time() - t0) + ' s')

        # Filter 2: lowpass Butterworth
        t0 = time.time()
        fc = Fdrive * 20
        nyq = 0.5 / dt  # Nyquist frequency
        fcn = fc / nyq
        btw_order = 5  # Butterworth filter order
        filtb, filta = signal.butter(btw_order, fcn)
        Cm_filtBW = signal.filtfilt(filtb, filta, Cm)
        print('Butterworth method: ' + '{:.2e}'.format(time.time() - t0) + ' s')

        # Filter 3: FFT lowpass cutoff
        t0 = time.time()
        Cm_ft = np.fft.rfft(Cm)
        W = np.fft.rfftfreq(Cm.size, d=dt)
        cut_Cm_ft = Cm_ft.copy()
        cut_Cm_ft[(W > fc)] = 0
        Cm_filtFFT = np.fft.irfft(cut_Cm_ft, n=Cm.size)
        print('FFT cutoff method: ' + '{:.2e}'.format(time.time() - t0) + ' s')
        Cm_IFFT = np.fft.irfft(Cm_ft, n=Cm.size)

        # Extending for 2 periods
        t_ext = np.concatenate((t, t + t[-1]), axis=0)
        Cm_ext = np.concatenate((Cm, Cm), axis=0)
        Cm_filtBW_ext = np.concatenate((Cm_filtBW, Cm_filtBW), axis=0)
        Cm_filtMAV_ext = np.concatenate((Cm_filtMAV, Cm_filtMAV), axis=0)
        Cm_filtFFT_ext = np.concatenate((Cm_filtFFT, Cm_filtFFT), axis=0)
        Cm_IFFT_ext = np.concatenate((Cm_IFFT, Cm_IFFT), axis=0)

        # Down-sampling
        npc_ds = 40
        t_ds = np.linspace(t[0], t[-1], npc_ds)
        print(t[0], t_ds[0], t[-1], t_ds[-1])
        Cm_ds = signal.resample(Cm, npc_ds, t=None, axis=0, window=None)
        Cm_filtMAV_ds = signal.resample(Cm_filtMAV, npc_ds, t=None, axis=0, window=None)
        Cm_filtBW_ds = signal.resample(Cm_filtBW, npc_ds, t=None, axis=0, window=None)
        Cm_filtFFT_ds = signal.resample(Cm_filtFFT, npc_ds, t=None, axis=0, window=None)
        Cm_IFFT_ds = signal.resample(Cm_IFFT, npc_ds, t=None, axis=0, window=None)

        i_ds_custom = np.round(np.linspace(0, t.size - 1, npc_ds)).astype(int)
        Cm_filtMAV_ds_custom = Cm_filtMAV[i_ds_custom]

        (t_ds2, Cm_filtMAV_ds_custom2) = DownSample(t, Cm, 0.025 / Fdrive)

        # Extending DS signals for 2 periods
        t_ds_ext = np.concatenate((t_ds, t_ds + t_ds[-1]), axis=0)

        print(t_ext[0], t_ds_ext[0], t_ext[-1], t_ds_ext[-1])

        Cm_ds_ext = np.concatenate((Cm_ds, Cm_ds), axis=0)
        Cm_filtBW_ds_ext = np.concatenate((Cm_filtBW_ds, Cm_filtBW_ds), axis=0)
        Cm_filtMAV_ds_ext = np.concatenate((Cm_filtMAV_ds, Cm_filtMAV_ds), axis=0)
        Cm_filtFFT_ds_ext = np.concatenate((Cm_filtFFT_ds, Cm_filtFFT_ds), axis=0)
        Cm_IFFT_ds_ext = np.concatenate((Cm_IFFT_ds, Cm_IFFT_ds), axis=0)

        Cm_filtMAV_ds_custom_ext = np.concatenate((Cm_filtMAV_ds_custom, Cm_filtMAV_ds_custom),
                                                  axis=0)

        t_ds2_ext = np.concatenate((t_ds2, t_ds2 + t_ds2[-1]), axis=0)
        Cm_filtMAV_ds_custom2_ext = np.concatenate((Cm_filtMAV_ds_custom2, Cm_filtMAV_ds_custom2),
                                                   axis=0)

        # Plots

        # fig, axes = plt.subplots(4,1)

        # ax = axes[0]
        # ax.set_title(str(npc) + ' samples per cycle', fontsize=28)
        # ax.plot(t_ext*t_factor, Cm_ext*1e2, color='black', linewidth=2, label='original')
        # ax.plot(t_ext*t_factor, Cm_IFFT_ext*1e2, color='magenta', linewidth=2,
        #         label='IFFT reconstructed')
        # ax.set_ylabel('$C_m \ (uF/cm^2)$', fontsize=28)
        # ax.legend(loc=0, fontsize=28)

        # ax = axes[1]
        # ax.plot(t_ext*t_factor, Cm_ext*1e2, color='black', linewidth=2, label='original')
        # ax.plot(t_ext*t_factor, Cm_filtMAV_ext*1e2, color='green', linewidth=2,
        #         label=str(N)+'-moving average')
        # ax.set_ylabel('$C_m \ (uF/cm^2)$', fontsize=28)
        # ax.legend(loc=0, fontsize=28)

        # ax = axes[2]
        # ax.plot(t_ext*t_factor, Cm_ext*1e2, color='black', linewidth=2, label='original')
        # ax.plot(t_ext*t_factor, Cm_filtBW_ext*1e2, color='red', linewidth=2,
        #     label='{:.2f}'.format(fc*1e-6) + 'MHz lowpass BW')
        # ax.set_ylabel('$C_m \ (uF/cm^2)$', fontsize=28)
        # ax.legend(loc=0, fontsize=28)

        # ax = axes[3]
        # ax.plot(t_ext*t_factor, Cm_ext*1e2, color='black', linewidth=2, label='original')
        # ax.plot(t_ext*t_factor, Cm_filtFFT_ext*1e2, color='blue', linewidth=2,
        #     label='{:.2f}'.format(fc*1e-6) + 'MHz lowpass FFT')
        # ax.set_ylabel('$C_m \ (uF/cm^2)$', fontsize=28)
        # ax.set_xlabel('time (' + t_unit + ')')
        # ax.legend(loc=0, fontsize=28)

        fig, ax = plt.subplots()
        ax.set_title('Downsampled signals (' + str(npc_ds) + ' samples per cycle)', fontsize=28)
        ax.plot(t_ext * t_factor, Cm_ext * 1e2, color='gold', linewidth=2, label='original')
        ax.plot(t_ds_ext * t_factor, Cm_ds_ext * 1e2,
                color='black', linewidth=2, label='original DS')
        # ax.plot(t_ds_ext*t_factor, Cm_filtMAV_ds_ext*1e2, color='green', linewidth=2,
        #         label='MAV DS')
        # ax.plot(t_ds_ext*t_factor, Cm_filtBW_ds_ext*1e2, color='red', linewidth=2, label='BW DS')
        # ax.plot(t_ds_ext*t_factor, Cm_filtFFT_ds_ext*1e2, color='blue', linewidth=2,
        #         label='FFT DS')
        ax.plot(t_ds_ext * t_factor, Cm_filtMAV_ds_custom_ext * 1e2, color='magenta', linewidth=2,
                label='MAV DS custom')
        ax.plot(t_ds2_ext * t_factor, Cm_filtMAV_ds_custom2_ext * 1e2, color='cyan', linewidth=2,
                label='MAV DS custom 2')
        ax.set_xlabel('time (' + t_unit + ')')
        ax.set_ylabel('$C_m \ (uF/cm^2)$', fontsize=28)
        ax.legend()

        # fig, ax = plt.subplots()
        # Cm_ext = np.concatenate((Cm, Cm), axis=0)
        # Cm_ifft_ext = np.concatenate((cut_Cm, cut_Cm), axis=0)
        # Cm_mav_ext = np.concatenate((Cm_mav, Cm_mav), axis=0)
        # Cm_filtbw_ext = np.concatenate((Cm_filtbw, Cm_filtbw), axis=0)
        # ax.plot(Cm_ext*1e2, color='black', linewidth=3, label='original repeated')
        # ax.plot(Cm_mav_ext*1e2, color='green', linewidth=3, label='MAV repeated')
        # ax.plot(Cm_filtbw_ext*1e2, color='red', linewidth=3, label='BW repeated')
        # ax.plot(Cm_ifft_ext*1e2, color='blue', linewidth=3, label='FFT repeated')
        # ax.legend()

        # Plot FFT of capacitance
        # fig, ax = plt.subplots()
        # ax.set_xscale('log')
        # ax.plot(W, np.absolute(Cm_ft), color='black', linewidth=3, label='FFT mod')
        # ax.plot([fc, fc], [0, np.amax(np.absolute(Cm_ft))], color='red', linewidth=3,
        #         label='cutoff')
        # ax.set_xlabel('frequency (Hz)')
        # ax.legend()

plt.show()
