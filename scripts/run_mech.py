#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2016-11-21 10:46:56
# @Email: theo.lemaire@epfl.ch
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-31 17:05:15

''' Run simulations of the NICE mechanical model. '''

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from PySONIC.core import BilayerSonophore, runBatch
from PySONIC.utils import logger, selectDirDialog, parseUSAmps
from PySONIC.neurons import CorticalRS
from PySONIC.plt import plotBatch

# Default parameters
defaults = dict(
    Cm0=CorticalRS().Cm0 * 1e2,  # uF/m2
    Qm0=CorticalRS().Vm0,  # nC/m2
    radius=[32.0],  # nm
    embedding=[0.],  # um
    freq=[500.0],  # kHz
    amp=[100.0],  # kPa
    charge=[0.]  # nC/cm2
)


def runMechBatch(outdir, bls, stim_params, mpi=False):
    ''' Run batch simulations of the mechanical system with imposed values of charge density.

        :param outdir: full path to output directory
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
    queue = bls.createQueue(freqs, amps, charges)
    for item in queue:
        item.insert(0, outdir)
    return runBatch(bls.runAndSave, queue, mpi=mpi)


def main():
    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('--mpi', default=False, action='store_true', help='Use multiprocessing')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-p', '--plot', type=str, nargs='+', help='Variables to plot')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')

    # Stimulation parameters
    ap.add_argument('-a', '--radius', nargs='+', type=float, help='Sonophore radius (nm)')
    ap.add_argument('--Cm0', type=float, default=defaults['Cm0'],
                    help='Resting membrane capacitance (uF/cm2)')
    ap.add_argument('--Qm0', type=float, default=defaults['Qm0'],
                    help='Resting membrane charge density (nC/cm2)')
    ap.add_argument('-d', '--embedding', nargs='+', type=float, help='Embedding depth (um)')
    ap.add_argument('-f', '--freq', nargs='+', type=float, help='US frequency (kHz)')
    ap.add_argument('-A', '--amp', nargs='+', type=float, help='Acoustic pressure amplitude (kPa)')
    ap.add_argument('--Arange', type=str, nargs='+', help='Amplitude range [scale min max n] (kPa)')
    ap.add_argument('-I', '--intensity', nargs='+', type=float, help='Acoustic intensity (W/cm2)')
    ap.add_argument('--Irange', type=str, nargs='+',
                    help='Intensity range [scale min max n] (W/cm2)')
    ap.add_argument('-Q', '--charge', nargs='+', type=float,
                    help='Membrane charge density (nC/cm2)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    loglevel = logging.DEBUG if args['verbose'] is True else logging.INFO
    logger.setLevel(loglevel)
    outdir = args['outputdir'] if 'outputdir' in args else selectDirDialog()
    if outdir == '':
        logger.error('No output directory selected')
        quit()
    mpi = args['mpi']
    Cm0 = args['Cm0'] * 1e-2  # F/m2
    Qm0 = args['Qm0'] * 1e-5  # C/m2
    radii = np.array(args.get('radius', defaults['radius'])) * 1e-9  # m
    embeddings = np.array(args.get('embedding', defaults['embedding'])) * 1e-6  # m

    try:
        amps = parseUSAmps(args, defaults)
    except ValueError as err:
        logger.error(err)
        quit()

    stim_params = dict(
        freqs=np.array(args.get('freq', defaults['freq'])) * 1e3,  # Hz
        amps=amps,  # Pa
        charges=np.array(args.get('charge', defaults['charge'])) * 1e-5  # C/m2
    )

    # Run MECH batch
    pkl_filepaths = []
    for a in radii:
        for d in embeddings:
            bls = BilayerSonophore(a, Cm0, Qm0, embedding_depth=d)
            pkl_filepaths += runMechBatch(outdir, bls, stim_params, mpi=mpi)
    pkl_dir, _ = os.path.split(pkl_filepaths[0])

    # Plot resulting profiles
    if 'plot' in args:
        if args['plot'] == ['all']:
            pltscheme = None
        else:
            pltscheme = {x: [x] for x in args['plot']}
        plotBatch(pkl_filepaths, pltscheme=pltscheme)
        plt.show()


if __name__ == '__main__':
    main()
