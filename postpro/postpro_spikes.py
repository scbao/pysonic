#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-27 09:50:55
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-28 14:18:29

""" Test influence of acoustic intensity and duration on number of spikes. """

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

from PointNICE.utils import ImportExcelCol, ConstructMatrix, Pressure2Intensity


# Define options
plot2d_bool = 0
plot3d_show = 1
plot3d_save = 0
plt_root = "../Output/effective spikes 2D/"
plt_save_ext = '.png'

# Import data
xls_file = "../../Output/effective spikes 2D/nbls_log_spikes.xlsx"
sheet = 'Data'
f_all = ImportExcelCol(xls_file, sheet, 'E', 2) * 1e3  # Hz
A_all = ImportExcelCol(xls_file, sheet, 'F', 2) * 1e3  # Pa
T_all = ImportExcelCol(xls_file, sheet, 'G', 2) * 1e-3  # s
N_all = ImportExcelCol(xls_file, sheet, 'Q', 2)  # number of spikes

freqs = np.unique(f_all)

for Fdrive in freqs:

    # Select data
    A = A_all[f_all == Fdrive]
    T = T_all[f_all == Fdrive]
    N = N_all[f_all == Fdrive]

    # Reshape serialized data into 2 dimensions
    (durations, amps, nspikes, nholes) = ConstructMatrix(T, A, N)
    nspikes2 = nspikes.conj().T  # conjugate tranpose of nspikes matrix (for surface plot)

    # Convert to appropriate units
    intensities = Pressure2Intensity(amplitudes) * 1e-4  # W/cm2
    durations = durations * 1e3  # ms

    nDurations = durations.size
    nIntensities = intensities.size

    Tmax = np.amax(durations)
    Tmin = np.amin(durations)
    Imax = np.amax(intensities)
    Imin = np.amin(intensities)
    print(str(nholes) + " hole(s) in reconstructed matrix")

    mymap = cm.get_cmap('jet')

    if plot2d_bool == 1:

        # Plot spikes vs. intensity (with duration color code)
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xlabel("$I \ (W/cm^2)$", fontsize=28)
        ax.set_ylabel("$\#\ spikes$", fontsize=28)
        for i in range(nIntensities):
            ax.plot(intensities, nspikes[i, :], c=mymap((durations[i] - Tmin) / (Tmax - Tmin)),
                    label='t = ' + str(durations[i]) + ' ms')
        sm_duration = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Tmin, Tmax))
        sm_duration._A = []
        cbar = plt.colorbar(sm_duration)
        cbar.ax.set_ylabel('$duration \ (ms)$', fontsize=28)

        # Plot spikes vs. duration (with intensity color code)
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xlabel("$duration \ (ms)$", fontsize=28)
        ax.set_ylabel("$\#\ spikes$", fontsize=28)
        for j in range(nDurations):
            ax.plot(durations, nspikes[:, j], c=mymap((intensities[j] - Imin) / (Imax - Imin)),
                    label='I = ' + str(intensities[j]) + ' W/cm2')
        sm_int = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(Imin, Imax))
        sm_int._A = []
        cbar = plt.colorbar(sm_int)
        cbar.ax.set_ylabel("$I \ (W/cm^2)$", fontsize=28)


    if plot3d_show == 1 and nholes == 0:

        # 3D surface plot: nspikes = f(duration, intensity)
        X, Y = np.meshgrid(durations, intensities)
        fig = plt.figure(figsize=(12, 9))
        ax = fig.gca(projection=Axes3D.name)
        ax.plot_surface(X, Y, nspikes2, rstride=1, cstride=1, cmap=mymap, linewidth=0,
                        antialiased=False)
        ax.set_xlabel("$duration \ (ms)$", fontsize=24, labelpad=20)
        ax.set_ylabel("$intensity \ (W/cm^2)$", fontsize=24, labelpad=20)
        ax.set_zlabel("$\#\ spikes$", fontsize=24, labelpad=20)
        csetx = ax.contour(X, Y, nspikes2, zdir='x', offset=150, cmap=cm.coolwarm)
        csety = ax.contour(X, Y, nspikes2, zdir='y', offset=0.8, cmap=cm.coolwarm)
        ax.view_init(33, -126)
        ax.set_xticks([0, 50, 100, 150])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_zticks([0, 20, 40, 60, 80])
        for item in ax.get_yticklabels():
            item.set_fontsize(24)
        for item in ax.get_xticklabels():
            item.set_fontsize(24)
        for item in ax.get_zticklabels():
            item.set_fontsize(24)

        # Save figure if needed
        if plot3d_save == 1:
            plt_filename = '{}spikes_{:.0f}KHz{}'.format(plt_root, Fdrive * 1e-3, plt_save_ext)
            plt.savefig(plt_filename)
            print('Saving figure to "' + plt_root + '"')
            plt.close()

    plt.show()
