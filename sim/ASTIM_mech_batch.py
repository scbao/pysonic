#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-21 10:46:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2017-08-24 14:13:24

""" Run batch simulations of the NICE mechanical model with imposed charge densities """

import time
import logging
import pickle
import numpy as np

import PointNICE
from PointNICE.utils import LoadParams
from PointNICE.solvers import xlslog, checkBatchLog

# Set logging options
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%d/%m/%Y %H:%M:%S:')
logger = logging.getLogger('PointNICE')
logger.setLevel(logging.DEBUG)

# Select output directory
try:
    (batch_dir, log_filepath) = checkBatchLog('mech')
except AssertionError as err:
    logger.error(err)
    quit()

# Define naming and logging settings
sim_str = 'sim_{:.0f}nm_{:.0f}kHz_{:.0f}kPa_{:.1f}nCcm2_mech'
sim_log = 'simulation %u/%u (a = %.1f nm, d = %.1f um, f = %.2f kHz, A = %.2f kPa, Q = %.1f nC/cm2)'


logger.info("Starting BLS simulation batch")

# Load NICE parameters
params = LoadParams()
biomech = params['biomech']
ac_imp = biomech['rhoL'] * biomech['c']  # Rayl

# Set geometry of NBLS structure
a = 32e-9  # in-plane radius (m)
d = 0.0e-6  # embedding tissue thickness (m)
geom = {"a": a, "d": d}
Cm0 = 1e-2  # membrane resting capacitance (F/m2)
Qm0 = -54.0e-5  # membrane resting charge density (C/m2)

# Set stimulation parameters
freqs = [6.9e5]  # Hz
amps = [1.43e3]  # Pa
charges = np.linspace(0.0, 80.0, 81) * 1e-5  # C/m2

# Run simulations
nsims = len(freqs) * len(amps) * len(charges)
simcount = 0
for Fdrive in freqs:
    try:
        bls = PointNICE.BilayerSonophore(geom, params, Fdrive, Cm0, Qm0)
        # Create SolverUS instance (compression modulus of embedding tissue depends on frequency)

        for Adrive in amps:
            for Qm in charges:

                simcount += 1

                # Get date and time info
                date_str = time.strftime("%Y.%m.%d")
                daytime_str = time.strftime("%H:%M:%S")

                # Log to console
                logger.info(sim_log, simcount, nsims, a * 1e9, d * 1e6, Fdrive * 1e-3,
                            Adrive * 1e-3, Qm * 1e5)

                # Run simulation
                tstart = time.time()
                (t, y, states) = bls.runMech(Fdrive, Adrive, Qm)
                (Z, ng) = y
                U = np.insert(np.diff(Z) / np.diff(t), 0, 0.0)
                tcomp = time.time() - tstart
                logger.info('completed in %.2f seconds', tcomp)

                # Export data to PKL file
                simcode = sim_str.format(a * 1e9, Fdrive * 1e-3, Adrive * 1e-3, Qm * 1e5)
                datafile_name = batch_dir + '/' + simcode + ".pkl"
                data = {'a': a,
                        'd': d,
                        'params': params,
                        'Fdrive': Fdrive,
                        'Adrive': Adrive,
                        'phi': np.pi,
                        'Qm': Qm,
                        't': t,
                        'states': states,
                        'U': U,
                        'Z': Z,
                        'ng': ng}

                with open(datafile_name, 'wb') as fh:
                    pickle.dump(data, fh)

                # Compute key output metrics
                Zmax = np.amax(Z)
                Zmin = np.amin(Z)
                Zabs_max = np.amax(np.abs([Zmin, Zmax]))
                eAmax = bls.arealstrain(Zabs_max)
                Tmax = bls.TEtot(Zabs_max)
                Pmmax = bls.PMavgpred(Zmin)
                ngmax = np.amax(ng)
                dUdtmax = np.amax(np.abs(np.diff(U) / np.diff(t)**2))

                # Export key metrics to log xls file
                log = {
                    'A': date_str,
                    'B': daytime_str,
                    'C': a * 1e9,
                    'D': d * 1e6,
                    'E': Fdrive * 1e-3,
                    'F': Adrive * 1e-3,
                    'G': Qm * 1e5,
                    'H': t.size,
                    'I': tcomp,
                    'J': bls.kA + bls.kA_tissue,
                    'K': Zmax * 1e9,
                    'L': eAmax,
                    'M': Tmax * 1e3,
                    'N': (ngmax - bls.ng0) / bls.ng0,
                    'O': Pmmax * 1e-3,
                    'P': dUdtmax
                }

                success = xlslog(log_filepath, 'Data', log)
                if success == 1:
                    logger.info('log exported to "%s"', log_filepath)
                else:
                    logger.error('log export to "%s" aborted', log_filepath)

    except AssertionError as err:
        logger.error(err)
