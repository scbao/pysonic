# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2019-03-18 18:06:20
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-12 12:20:32

import os
import logging
from argparse import ArgumentParser

from PySONIC.neurons import getPointNeuron
from PySONIC.utils import logger, selectDirDialog
from PySONIC.core import NmodlGenerator


def main():
    ap = ArgumentParser()
    ap.add_argument('-n', '--neuron', type=str, default='RS', help='Neuron name (string)')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')

    logger.setLevel(logging.INFO)
    args = ap.parse_args()
    try:
        pneuron = getPointNeuron(args.neuron)
    except ValueError as err:
        logger.error(err)
        return
    outdir = args.outputdir if args.outputdir is not None else selectDirDialog()
    if outdir == '':
        logger.error('No output directory selected')
        quit()
    outfile = '{}.mod'.format(args.neuron)
    outpath = os.path.join(outdir, outfile)

    gen = NmodlGenerator(pneuron)
    logger.info('generating %s neuron MOD file in "%s"', pneuron.name, outdir)
    gen.print(outpath)


if __name__ == '__main__':
    main()
