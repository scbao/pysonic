#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-10-07 10:22:24
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-08-21 16:09:21

""" Analysis of the system geometric variables and interplaying forces at
stake in a static quasi-steady NICE system. """

import time
import numpy as np
import matplotlib.pyplot as plt

from PySONIC import BilayerSonophore
from PySONIC.utils import PmCompMethod


plt_bool = 1

# Initialization: create a BLS instance
a = 32e-9  # in-plane radius (m)
Fdrive = 0.0  # dummy stimulation frequency
Cm0 = 1e-2  # membrane resting capacitance (F/m2)
Qm0 = -71.9e-5  # membrane resting charge density (C/m2)
bls = BilayerSonophore(a, Fdrive, Cm0, Qm0)


# Input 1: leaflet deflections
ZMin = -0.45 * bls.Delta
ZMax = 2 * bls.a
nZ = 3000
Z = np.linspace(ZMin, ZMax, nZ)
Zlb = -0.5 * bls.Delta
Zub = bls.a

# Input 2: acoustic perturbations
PacMax = 9.5e4
nPac1 = 5
nPac2 = 100
Pac1 = np.linspace(-PacMax, PacMax, nPac1)
Pac2 = np.linspace(-PacMax, PacMax, nPac2)

# Input 3: membrane charge densities
QmMin = bls.Qm0
QmMax = 50.0e-5
nQm = 7
Qm = np.linspace(QmMin, QmMax, nQm)


# Outputs
R = np.empty(nZ)
Cm = np.empty(nZ)
Pm_apex = np.empty(nZ)
Pm_avg = np.empty(nZ)
Pm_avg_predict = np.empty(nZ)
Pg = np.empty(nZ)
Pec = np.empty(nZ)
Pel = np.empty(nZ)
P0 = np.ones(nZ) * bls.P0
Pnet = np.empty(nZ)
Pqs = np.empty((nZ, nPac1))
Pecdense = np.empty((nZ, nQm))
Pnetdense = np.empty((nZ, nQm))
Zeq = np.empty(nPac1)
Zeq_dense = np.empty(nPac2)

t0 = time.time()

# Check net QS pressure at Z = 0
Peq0 = bls.PtotQS(0.0, bls.ng0, bls.Qm0, 0.0, PmCompMethod.direct)
print('Net QS pressure at Z = 0.0 without perturbation: ' + '{:.2e}'.format(Peq0) + ' Pa')

# Loop through the deflection vector
for i in range(nZ):

    # 1-dimensional output vectors
    R[i] = bls.curvrad(Z[i])
    Cm[i] = bls.Capct(Z[i])
    Pm_apex[i] = bls.PMlocal(0.0, Z[i], R[i])
    Pm_avg[i] = bls.PMavg(Z[i], R[i], bls.surface(Z[i]))
    Pm_avg_predict[i] = bls.PMavgpred(Z[i])
    Pel[i] = bls.PEtot(Z[i], R[i])
    Pg[i] = bls.gasmol2Pa(bls.ng0, bls.volume(Z[i]))
    Pec[i] = bls.Pelec(Z[i], bls.Qm0)
    Pnet[i] = bls.PtotQS(Z[i], bls.ng0, bls.Qm0, 0.0, PmCompMethod.direct)

    # loop through the acoustic perturbation vector an compute 2-dimensional
    # balance pressure output vector
    for j in range(nPac1):
        Pqs[i, j] = bls.PtotQS(Z[i], bls.ng0, bls.Qm0, Pac1[j], PmCompMethod.direct)

    for j in range(nQm):
        Pecdense[i, j] = bls.Pelec(Z[i], Qm[j])
        Pnetdense[i, j] = bls.PtotQS(Z[i], bls.ng0, Qm[j], 0.0, PmCompMethod.direct)

# Compute min local intermolecular pressure
Pm_apex_min = np.amin(Pm_apex)
iPm_apex_min = np.argmin(Pm_apex)
print("min local intermolecular resultant pressure = %.2e Pa for z = %.2f nm" %
      (Pm_apex_min, Z[iPm_apex_min] * 1e9))

for j in range(nPac1):
    Zeq[j] = bls.balancedefQS(bls.ng0, bls.Qm0, Pac1[j], PmCompMethod.direct)
for j in range(nPac2):
    Zeq_dense[j] = bls.balancedefQS(bls.ng0, bls.Qm0, Pac2[j], PmCompMethod.direct)


t1 = time.time()
print("computation completed in " + '{:.2f}'.format(t1 - t0) + " s")


if plt_bool == 1:

    # 1: Intermolecular pressures
    fig1, ax = plt.subplots()
    fig1.canvas.set_window_title("1: integrated vs. predicted average intermolecular pressure")
    ax.set_xlabel('Z $(nm)$', fontsize=18)
    ax.set_ylabel('Pressures $(MPa)$', fontsize=18)
    ax.grid(True)
    ax.plot([Zlb * 1e9, Zlb * 1e9], [np.amin(Pm_avg) * 1e-6, np.amax(Pm_avg) * 1e-6], '--',
            color="blue", label="$-\Delta /2$")
    ax.plot([Zub * 1e9, Zub * 1e9], [np.amin(Pm_avg) * 1e-6, np.amax(Pm_avg) * 1e-6], '--',
            color="red", label="$a$")
    ax.plot(Z * 1e9, Pm_avg * 1e-6, '-', label="$P_{M, avg}$", color="green", linewidth=2.0)
    ax.plot(Z * 1e9, Pm_avg_predict * 1e-6, '-', label="$P_{M, avg-predict}$", color="red",
            linewidth=2.0)
    ax.set_xlim(ZMin * 1e9 - 5, ZMax * 1e9)
    ax.legend(fontsize=24)


    # 2: Capacitance and electric pressure
    fig2, ax = plt.subplots()
    fig2.canvas.set_window_title("2: Capacitance and electric equivalent pressure")
    ax.set_xlabel('Z $(nm)$', fontsize=18)
    ax.set_ylabel('$C_m \ (uF/cm^2)$', fontsize=18)
    ax.plot(Z * 1e9, Cm * 1e2, '-', label="$C_{m}$", color="black", linewidth=2.0)
    ax.set_xlim(ZMin * 1e9 - 5, ZMax * 1e9)
    ax2 = ax.twinx()
    ax2.set_ylabel('$P_{EC}\ (MPa)$', fontsize=18, color='magenta')
    ax2.plot(Z * 1e9, Pec * 1e-6, '-', label="$P_{EC}$", color="magenta", linewidth=2.0)

    # tmp: electric pressure for varying membrane charge densities
    figtmp, ax = plt.subplots()
    figtmp.canvas.set_window_title("electric pressure for varying membrane charges")
    ax.set_xlabel('Z $(nm)$', fontsize=18)
    ax.set_ylabel('$P_{EC} \ (MPa)$', fontsize=18)
    for j in range(nQm):
        lbl = "$Q_m$ = " + '{:.2f}'.format(Qm[j] * 1e5) + " nC/cm2"
        ax.plot(Z * 1e9, Pecdense[:, j] * 1e-6, '-', label=lbl, linewidth=2.0)
    ax.set_xlim(ZMin * 1e9 - 5, ZMax * 1e9)
    ax.legend()


    # tmp: net pressure for varying membrane potentials
    figtmp, ax = plt.subplots()
    figtmp.canvas.set_window_title("net pressure for varying membrane charges")
    ax.set_xlabel('Z $(nm)$', fontsize=18)
    ax.set_ylabel('$P_{net} \ (MPa)$', fontsize=18)
    for j in range(nQm):
        lbl = "$Q_m$ = " + '{:.2f}'.format(Qm[j] * 1e5) + " nC/cm2"
        ax.plot(Z * 1e9, Pnetdense[:, j] * 1e-6, '-', label=lbl, linewidth=2.0)
    ax.set_xlim(ZMin * 1e9 - 5, ZMax * 1e9)
    ax.legend()


    # 3: Net pressure without perturbation
    fig3, ax = plt.subplots()
    fig3.canvas.set_window_title("3: Net QS pressure without perturbation")
    ax.set_xlabel('Z $(nm)$', fontsize=18)
    ax.set_ylabel('Pressures $(kPa)$', fontsize=18)
    # ax.grid(True)
    # ax.plot([Zlb * 1e9, Zlb * 1e9], [np.amin(Pec) * 1e-3, np.amax(Pm_avg) * 1e-3], '--',
    #         color="blue", label="$-\Delta / 2$")
    # ax.plot([Zub * 1e9, Zub * 1e9], [np.amin(Pec) * 1e-3, np.amax(Pm_avg) * 1e-3], '--',
    #         color="red", label="$a$")
    ax.plot(Z * 1e9, Pg * 1e-3, '-', label="$P_{gas}$", linewidth=3.0, color='C0')
    ax.plot(Z * 1e9, -P0 * 1e-3, '-', label="$-P_{0}$", linewidth=3.0, color='C1')
    ax.plot(Z * 1e9, Pm_avg * 1e-3, '-', label="$P_{mol}$", linewidth=3.0, color='C2')
    ax.plot(Z * 1e9, Pec * 1e-3, '-', label="$P_{elec}$", linewidth=3.0, color='C3')
    ax.plot(Z * 1e9, Pel * 1e-3, '-', label="$P_{elastic}$", linewidth=3.0, color='C4')
    # ax.plot(Z * 1e9, (Pg - P0 + Pm_avg + Pec + Pel) * 1e-3, '--', label="$P_{net}$", linewidth=2.0,
            # color='black')
    # ax.plot(Z * 1e9, (Pg - P0 + Pm_avg + Pec - Pnet) * 1e-6, '--', label="$P_{net} diff$",
    #         linewidth=2.0, color="blue")
    ax.set_xlim(ZMin * 1e9 - 5, 30)
    ax.set_ylim(-1500, 2000)
    ax.legend(fontsize=24)
    # ax.grid(True)


    # 4: QS pressure for different perturbations
    fig4, ax = plt.subplots()
    fig4.canvas.set_window_title("4: Net QS pressure for different acoustic perturbations")
    ax.set_xlabel('Z $(nm)$', fontsize=18)
    ax.set_ylabel('Pressures $(MPa)$', fontsize=18)
    ax.grid(True)
    ax.plot([Zlb * 1e9, Zlb * 1e9], [np.amin(Pqs[:, 0]) * 1e-6, np.amax(Pqs[:, nPac1 - 1]) * 1e-6],
            '--', color="blue", label="$-\Delta/2$")
    ax.plot([Zub * 1e9, Zub * 1e9], [np.amin(Pqs[:, 0]) * 1e-6, np.amax(Pqs[:, nPac1 - 1]) * 1e-6],
            '--', color="red", label="$a$")
    ax.set_xlim(ZMin * 1e9 - 5, ZMax * 1e9)
    for j in range(nPac1):
        lbl = "$P_{A}$ = %.2f MPa" % (Pac1[j] * 1e-6)
        ax.plot(Z * 1e9, Pqs[:, j] * 1e-6, '-', label=lbl, linewidth=2.0)
        ax.plot([Zeq[j] * 1e9, Zeq[j] * 1e9], [np.amin(Pqs[:, nPac1 - 1]) * 1e-6,
                                               np.amax(Pqs[:, 0]) * 1e-6], '--', color="black")
    ax.legend(fontsize=24)

    # 5: QS balance deflection for different acoustic perturbations
    fig5, ax = plt.subplots()
    fig5.canvas.set_window_title("5: QS balance deflection for different acoustic perturbations ")
    ax.set_xlabel('Perturbation $(MPa)$', fontsize=18)
    ax.set_ylabel('Z $(nm)$', fontsize=18)
    ax.plot([np.amin(Pac2) * 1e-6, np.amax(Pac2) * 1e-6], [Zlb * 1e9, Zlb * 1e9], '--',
            color="blue", label="$-\Delta / 2$")
    ax.plot([np.amin(Pac2) * 1e-6, np.amax(Pac2) * 1e-6], [Zub * 1e9, Zub * 1e9], '--',
            color="red", label="$a$")
    ax.plot([-bls.P0 * 1e-6, -bls.P0 * 1e-6],
            [np.amin(Zeq_dense) * 1e9, np.amax(Zeq_dense) * 1e9], '--', color="black",
            label="$-P_0$")
    ax.plot(Pac2 * 1e-6, Zeq_dense * 1e9, '-', label="$Z_{eq}$", linewidth=2.0)
    ax.set_xlim(-0.12, 0.12)
    ax.set_ylim(ZMin * 1e9 - 5, bls.a * 1e9 + 5)
    ax.legend(fontsize=24)

    plt.show()
