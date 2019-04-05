# -*- coding: utf-8 -*-
# @Author: Theo
# @Date:   2019-03-18 18:06:20
# @Last Modified by:   Theo
# @Last Modified time: 2019-03-18 21:18:38

import os
import logging
from argparse import ArgumentParser

from PySONIC.neurons import getNeuronsDict
from PySONIC.utils import logger, selectDirDialog
from PySONIC.core import NmodlGenerator


def main():
    ap = ArgumentParser()
    ap.add_argument('-n', '--neuron', type=str, default='RS', help='Neuron name (string)')
    ap.add_argument('-o', '--outputdir', type=str, default=None, help='Output directory')

    logger.setLevel(logging.INFO)
    args = ap.parse_args()
    neuron = getNeuronsDict()[args.neuron]()
    outdir = args.outputdir if args.outputdir is not None else selectDirDialog()
    if outdir == '':
        logger.error('No output directory selected')
        quit()
    outfile = '{}.mod'.format(args.neuron)
    outpath = os.path.join(outdir, outfile)

    gen = NmodlGenerator(neuron)
    logger.info('generating %s neuron MOD file in "%s"', neuron.name, outdir)
    gen.print(outpath)



if __name__ == '__main__':
    main()