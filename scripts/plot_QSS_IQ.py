# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-03 09:45:29

''' Phase-plane analysis of neuron behavior under quasi-steady state approximation. '''

import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging

from PySONIC.utils import logger, selectDirDialog, parseUSAmps, addUSAmps, getInDict
from PySONIC.neurons import getNeuronsDict
from PySONIC.plt import plotQSSvars, plotQSSVarVsAmp, plotEqChargeVsAmp


def main():

    ap = ArgumentParser()

    # Runtime options
    ap.add_argument('-c', '--cmap', type=str, default='viridis', help='Colormap name')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save output figures')
    ap.add_argument('--comp', default=False, action='store_true',
                    help='Compare with simulations')
    ap.add_argument('--vars', type=str, nargs='+', default=None, help='Variables to plot')
    ap.add_argument('--mpi', default=False, action='store_true', help='Use multiprocessing')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')
    ap.add_argument('-i', '--inputdir', type=str, default=None, help='Input directory')

    # Stimulation parameters
    ap.add_argument('-n', '--neurons', type=str, nargs='+', default=None, help='Neuron types')
    ap.add_argument('-a', '--radius', type=float, default=32., help='Sonophore radius (nm)')
    ap.add_argument('-f', '--freq', type=float, default=500., help='US frequency (kHz)')
    ap.add_argument('-A', '--amp', nargs='+', type=float, help='Acoustic pressure amplitude (kPa)')
    ap.add_argument('--Arange', type=str, nargs='+', help='Amplitude range [scale min max n] (kPa)')
    ap.add_argument('-I', '--intensity', nargs='+', type=float, help='Acoustic intensity (W/cm2)')
    ap.add_argument('--Irange', type=str, nargs='+',
                    help='Intensity range [scale min max n] (W/cm2)')
    ap.add_argument('--tstim', type=float, default=1000., help='Stimulus duration (ms)')
    ap.add_argument('--toffset', type=float, default=0., help='Offset duration (ms)')
    ap.add_argument('--PRF', type=float, default=100., help='Pulse-repetition-frequency (Hz)')
    ap.add_argument('--DC', type=float, nargs='+', default=None, help='Duty cycle (%)')

    # Parse arguments
    args = {key: value for key, value in vars(ap.parse_args()).items() if value is not None}
    loglevel = logging.DEBUG if args['verbose'] is True else logging.INFO
    logger.setLevel(loglevel)
    mpi = args['mpi']
    comp = args['comp']
    save = args['save']
    cmap = args['cmap']
    pltvars = args.get('vars', ['dQdt'])
    Ascale = args.get('Arange', ['lin'])[0]

    if comp:
        indir = getInDict(args, 'inputdir', selectDirDialog)
        if indir == '':
            logger.error('no input directory')
            return
    if save:
        outdir = getInDict(args, 'outputdir', selectDirDialog)
        if outdir == '':
            logger.error('no output directory')
            return

    neurons = [getNeuronsDict()[n]() for n in args.get('neurons', ['RS', 'LTS'])]
    a = args['radius'] * 1e-9  # m
    Fdrive = args['freq'] * 1e3  # Hz
    amps = parseUSAmps(args, np.linspace(1., 600., 3) * 1e3)  # Pa
    tstim = args['tstim'] * 1e-3  # s
    toffset = args['toffset'] * 1e-3  # s
    PRF = args['PRF']  # Hz
    DCs = np.array(args.get('DC', [100.])) * 1e-2  # (-)

    figs = []

    # Plot iNet vs Q for different amplitudes for each neuron and DC
    for i, neuron in enumerate(neurons):
        for DC in DCs:
            if amps.size == 1:
                figs.append(
                    plotQSSvars(neuron, a, Fdrive, amps[0]))
            else:
                for var in pltvars:
                    figs.append(plotQSSVarVsAmp(
                        neuron, a, Fdrive, var, amps=amps, DC=DC, cmap=cmap, zscale=Ascale))

        # Plot equilibrium charge as a function of amplitude for each neuron
        if amps.size > 1 and 'dQdt' in pltvars:
            figs.append(
                plotEqChargeVsAmp(
                    neuron, a, Fdrive, amps=amps, tstim=tstim, PRF=PRF, DCs=DCs, toffset=toffset,
                    xscale=Ascale, compdir=indir, mpi=mpi, loglevel=loglevel))

    if save:
        for fig in figs:
            s = fig.canvas.get_window_title()
            s = s.replace('(', '- ').replace('/', '_').replace(')', '')
            figname = '{}.png'.format(s)
            fig.savefig(os.path.join(outdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
