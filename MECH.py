#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-21 10:46:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2018-09-23 16:12:32

''' Run simulations of the NICE mechanical model. '''

import os
import logging
import numpy as np
from argparse import ArgumentParser

from PySONIC.core import BilayerSonophore
from PySONIC.utils import logger, selectDirDialog, checkBatchLog
from PySONIC.neurons import CorticalRS
from PySONIC.batches import createQueue, runBatch
from PySONIC.plt import plotBatch

# Default parameters
defaults = dict(
    Cm0=CorticalRS().Cm0 * 1e2,  # uF/m2
    Qm0=CorticalRS().Vm0,  # nC/m2
    diams=[32.0],  # nm
    embeddings=[0.],  # um
    freqs=[500.0],  # kHz
    amps=[100.0],  # kPa
    charges=[0.]  # nC/cm2
)


def runMechBatch(outdir, logpath, bls, stim_params, mpi=False):
    ''' Run batch simulations of the mechanical system with imposed values of charge density.

        :param outdir: full path to output directory
        :param logpath: full path log file
        :param bls: BilayerSonophore object
        :param stim_params: dictionary containing sweeps for all stimulation parameters
        :param mpi: boolean stating whether or not to use multiprocessing
        :return: list of full paths to the output files
    '''

    # Checking validity of stimulation parameters
    mandatory_params = ['freqs', 'amps', 'charges']
    for mparam in mandatory_params:
        if mparam not in stim_params:
            raise ValueError('Missing stimulation parameter field: "{}"'.format(mparam))

    logger.info("Starting mechanical simulation batch")

    # Unpack stimulation parameters
    freqs = np.array(stim_params['freqs'])
    amps = np.array(stim_params['amps'])
    charges = np.array(stim_params['charges'])

    # Generate simulations queue and run batch
    queue = createQueue((freqs, amps, charges))
    return runBatch(bls, 'runAndSave', queue, extra_params=[outdir, logpath], mpi=mpi)


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('--mpi', default=False, action='store_true', help='Use multiprocessing')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-p', '--plot', default=False, action='store_true', help='Plot results')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')

    # Stimulation parameters
    ap.add_argument('-a', '--diams', nargs='+', type=float, help='Sonophore diameter (nm)')
    ap.add_argument('--Cm0', type=float, default=defaults['Cm0'],
                    help='Resting membrane capacitance (uF/cm2)')
    ap.add_argument('--Qm0', type=float, default=defaults['Qm0'],
                    help='Resting membrane charge density (nC/cm2)')
    ap.add_argument('-d', '--embeddings', nargs='+', type=float, help='Embedding depth (um)')
    ap.add_argument('-f', '--freqs', nargs='+', type=float, help='US frequency (kHz)')
    ap.add_argument('-A', '--amps', nargs='+', type=float, help='Acoustic pressure amplitude (kPa)')
    ap.add_argument('-Q', '--charges', nargs='+', type=float, help='Membrane charge density (nC/cm2)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    loglevel = logging.DEBUG if args['verbose'] is True else logging.INFO
    logger.setLevel(loglevel)
    outdir = args['outputdir'] if 'outputdir' in args else selectDirDialog()
    mpi = args['mpi']
    plot = args['plot']
    Cm0 = args['Cm0'] * 1e-2  # F/m2
    Qm0 = args['Qm0'] * 1e-5  # C/m2
    diams = np.array(args.get('diams', defaults['diams'])) * 1e-9  # m
    embeddings = np.array(args.get('embeddings', defaults['embeddings'])) * 1e-6  # m
    stim_params = dict(
        freqs=np.array(args.get('freqs', defaults['freqs'])) * 1e3,  # Hz
        amps=np.array(args.get('amps', defaults['amps'])) * 1e3,  # Pa
        charges=np.array(args.get('charges', defaults['charges'])) * 1e-5  # C/m2
    )

    # Run MECH batch
    logpath, _ = checkBatchLog(outdir, 'MECH')
    pkl_filepaths = []
    for a in diams:
        for d in embeddings:
            bls = BilayerSonophore(a, Cm0, Qm0, embedding_depth=d)
            pkl_filepaths += runMechBatch(outdir, logpath, bls, stim_params, mpi=mpi)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    if plot:
        plotBatch(pkl_dir, pkl_filepaths)


if __name__ == '__main__':
    main()
