# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-04-05 18:46:36

''' Phase-plane analysis of neuron behavior under quasi-steady state approximation. '''

import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging

from PySONIC.utils import logger, selectDirDialog
from PySONIC.neurons import getNeuronsDict
from PySONIC.plt import plotQSSVarVsAmp, plotEqChargeVsAmp


def main():

    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-n', '--neurons', type=str, nargs='+', default=None, help='Neuron types')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')
    ap.add_argument('-c', '--cmap', type=str, default='viridis', help='Colormap name')
    ap.add_argument('-v', '--verbose', default=False, action='store_true',
                    help='Increase verbosity')
    ap.add_argument('-s', '--save', default=False, action='store_true',
                    help='Save output figures')
    ap.add_argument('--titrate', default=False, action='store_true',
                    help='Titrate excitation threshold')
    ap.add_argument('--tstim', type=float, default=500.,
                    help='Stimulus duration for titration (ms)')
    ap.add_argument('--PRF', type=float, default=100.,
                    help='Pulse-repetition-frequency for titration (Hz)')
    ap.add_argument('--DC', type=float, nargs='+', default=None, help='Duty cycle (%)')
    ap.add_argument('--Qi', type=str, default=None,
                    help='Initial membrane charge density for phase-plane analysis (nC/cm2)')
    ap.add_argument('--Ascale', type=str, default='log',
                    help='Scale type for acoustic amplitude ("lin" or "log")')
    ap.add_argument('-p', '--plotdetails', default=False, action='store_true',
                    help='Plot details')

    # Parse arguments
    args = ap.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    neurons = ['RS', 'LTS'] if args.neurons is None else args.neurons
    neurons = [getNeuronsDict()[n]() for n in neurons]

    if args.Qi == 'Qm0':
        Qi = np.array([neuron.Qm0() for neuron in neurons])  # C/m2
    elif args.Qi is None:
        Qi = [None] * len(neurons)
    else:
        Qi = np.ones(len(neurons)) * float(args.Qi) * 1e-5  # C/m2

    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    Arange = (1., 50.)  # kPa
    nA = 300
    amps = {
        'lin': np.linspace(Arange[0], Arange[1], nA),
        'log': np.logspace(np.log10(Arange[0]), np.log10(Arange[1]), 100)
    }[args.Ascale] * 1e3  # Pa
    tstim = args.tstim * 1e-3  # s
    PRF = args.PRF  # Hz
    DCs = [100.] if args.DC is None else args.DC  # %
    DCs = np.array(DCs) * 1e-2  # (-)

    figs = []

    if args.plotdetails:
        # Plot iNet vs Q for different amplitudes for each neuron
        logger.info('plotting iNet QSS profiles')
        for i, neuron in enumerate(neurons):
            for DC in DCs:
                figs.append(plotQSSVarVsAmp(
                    neuron, a, Fdrive, 'iNet', amps=amps, DC=DC, Qi=Qi[i], zscale=args.Ascale))

    # Plot equilibrium charge as a function of amplitude for each neuron
    logger.info('plotting equilibrium charges')
    figs.append(
        plotEqChargeVsAmp(
            neurons, a, Fdrive, amps=amps, tstim=tstim, PRF=PRF, DCs=DCs,
            Qi=Qi, xscale=args.Ascale, titrate=args.titrate))

    if args.save:
        outputdir = args.outputdir if args.outputdir is not None else selectDirDialog()
        if outputdir == '':
            logger.error('no output directory')
        else:
            for fig in figs:
                s = fig.canvas.get_window_title()
                s = s.replace('(', '- ').replace('/', '_').replace(')', '')
                figname = '{}.png'.format(s)
                fig.savefig(os.path.join(outputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()