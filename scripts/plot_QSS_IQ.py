# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-05-15 16:32:25

''' Phase-plane analysis of neuron behavior under quasi-steady state approximation. '''

import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging

from PySONIC.utils import logger, selectDirDialog
from PySONIC.neurons import getNeuronsDict
from PySONIC.plt import plotQSSvars, plotQSSVarVsAmp, plotEqChargeVsAmp


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
    ap.add_argument('-A', '--amp', type=float, default=None, help='Amplitude (kPa or mA/m2)')
    ap.add_argument('--tstim', type=float, default=500.,
                    help='Stimulus duration for titration (ms)')
    ap.add_argument('--PRF', type=float, default=100.,
                    help='Pulse-repetition-frequency for titration (Hz)')
    ap.add_argument('--DC', type=float, nargs='+', default=None, help='Duty cycle (%)')
    ap.add_argument('--Ascale', type=str, default='lin',
                    help='Scale type for acoustic amplitude ("lin" or "log")')
    ap.add_argument('--stim', type=str, default='US', help='Stimulation type ("US" or "elec")')
    ap.add_argument('--vars', type=str, nargs='+', default=None, help='Variables to plot')

    # Parse arguments
    args = ap.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    neurons = ['RS', 'LTS'] if args.neurons is None else args.neurons
    neurons = [getNeuronsDict()[n]() for n in neurons]

    # US parameters
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    Arange = (1., 60.)  # kPa
    nA = 100
    US_amps = {
        'lin': np.linspace(Arange[0], Arange[1], nA),
        'log': np.logspace(np.log10(Arange[0]), np.log10(Arange[1]), nA)
    }[args.Ascale] * 1e3  # Pa

    # E-STIM parameters
    Irange = (-20., 20.)  # mA/m2
    nI = 100
    Iinjs = np.linspace(Irange[0], Irange[1], nI)  # mA/m2

    # Pulsing parameters
    tstim = args.tstim * 1e-3  # s
    PRF = args.PRF  # Hz
    DCs = [100.] if args.DC is None else args.DC  # %
    DCs = np.array(DCs) * 1e-2  # (-)

    if args.stim == 'US':
        amps = US_amps if args.amp is None else np.array([args.amp * 1e3])
        cmap = args.cmap
    else:
        a = None
        Fdrive = None
        amps = Iinjs if args.amp is None else np.array([args.amp])
        cmap = 'RdBu_r'

    if args.vars is None:
        args.vars = ['dQdt']

    figs = []

    # Plot iNet vs Q for different amplitudes for each neuron and DC
    for i, neuron in enumerate(neurons):
        for DC in DCs:
            if amps.size == 1:
                figs.append(
                    plotQSSvars(neuron, a, Fdrive, amps[0]))
            else:
                for var in args.vars:
                    figs.append(plotQSSVarVsAmp(
                        neuron, a, Fdrive, var, amps=amps, DC=DC, cmap=cmap, zscale=args.Ascale))

    # Plot equilibrium charge as a function of amplitude for each neuron
    if amps.size > 1 and 'dQdt' in args.vars:
        figs.append(
            plotEqChargeVsAmp(
                neurons, a, Fdrive, amps=amps, tstim=tstim, PRF=PRF, DCs=DCs,
                xscale=args.Ascale, titrate=args.titrate))

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
