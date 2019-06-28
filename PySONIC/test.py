# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Email: theo.lemaire@epfl.ch
# @Date:   2019-06-28 11:55:16
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2019-06-28 12:18:47

import os
import time
import cProfile
import pstats
import inspect
import matplotlib.pyplot as plt

from .utils import logger
from .parsers import TestParser


class TestBase:

    prefix = 'test_'

    def execute(self, func_str, globals, locals, is_profiled):
        ''' Execute function with or without profiling. '''
        if is_profiled:
            pfile = 'tmp.stats'
            cProfile.runctx(func_str, globals, locals, pfile)
            stats = pstats.Stats(pfile)
            os.remove(pfile)
            stats.strip_dirs()
            stats.sort_stats('cumulative')
            stats.print_stats()
        else:
            eval(func_str, globals, locals)

    def main(self):

        # Build list of candidate testsets
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        testsets = {}
        n = len(self.prefix)
        for name, obj in methods:
            if name[:n] == self.prefix:
                testsets[name[n:]] = obj

        # Parse command line arguments
        parser = TestParser(list(testsets.keys()))
        parser.addHideOutput()
        args = parser.parse()
        logger.setLevel(args['loglevel'])
        if args['profile'] and args['subset'] == 'all':
            logger.error('profiling can only be run on individual tests')
            return

        # Run appropriate tests
        t0 = time.time()
        for s in args['subset']:
            testsets[s](args['profile'])
        tcomp = time.time() - t0
        logger.info('tests completed in %.0f s', tcomp)

        if not args['hide']:
            plt.show()
