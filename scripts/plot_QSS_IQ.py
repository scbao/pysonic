# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2018-09-28 16:13:34
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-03-26 20:16:42

''' Phase-plane analysis of neuron behavior under quasi-steady state approximation. '''

import os
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import logging

from PySONIC.utils import logger, selectDirDialog
from PySONIC.neurons import getNeuronsDict
from PySONIC.plt import plotQSSVarVsAmp


def main():

    ap = ArgumentParser()

    # Stimulation parameters
    ap.add_argument('-n', '--neuron', type=str, default='STN', help='Neuron type')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')
    ap.add_argument('-c', '--cmap', type=str, default='viridis', help='Colormap name')
    ap.add_argument('-v', '--verbose', default=False, action='store_true', help='Increase verbosity')
    ap.add_argument('-s', '--save', default=False, action='store_true', help='Save output figures')

    # Parse arguments
    args = ap.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    neuron = getNeuronsDict()[args.neuron]()
    a = 32e-9  # m
    Fdrive = 500e3  # Hz
    amps = np.logspace(np.log10(1.), np.log10(50.), 30) * 1e3  # Pa
    fig = plotQSSVarVsAmp(neuron, a, Fdrive, 'iNet', amps=amps, zscale='log')

    if args.save:
        outputdir = args.outputdir if args.outputdir is not None else selectDirDialog()
        if outputdir == '':
            logger.error('no output directory')
        else:
            s = fig.canvas.get_window_title()
            s = s.replace('(', '- ').replace('/', '_').replace(')', '')
            figname = '{}.png'.format(s)
            fig.savefig(os.path.join(outputdir, figname), transparent=True)
    else:
        plt.show()


if __name__ == '__main__':
    main()
