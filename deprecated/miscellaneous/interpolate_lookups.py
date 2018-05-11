import os
import time
import pickle
import logging
import numpy as np
from scipy.interpolate import interp2d
from PointNICE.utils import logger, getLookupDir, InputError, itrpLookupsFreq

# Set logging level
logger.setLevel(logging.INFO)

# Check lookup file existence
neuron = 'RS'
a = 32e-9
Fdrive = 600e3
Adrive = 100e3
lookup_file = '{}_lookups_a{:.1f}nm.pkl'.format(neuron, a * 1e9)
lookup_path = '{}/{}'.format(getLookupDir(), lookup_file)
if not os.path.isfile(lookup_path):
    raise InputError('Missing lookup file: "{}"'.format(lookup_file))

# Load lookups dictionary
with open(lookup_path, 'rb') as fh:
    lookups3D = pickle.load(fh)

# Retrieve 1D inputs from lookups dictionary
freqs = lookups3D.pop('f')
amps = lookups3D.pop('A')
charges = lookups3D.pop('Q')

t0 = time.time()
lookups2D = itrpLookupsFreq(lookups3D, freqs, Fdrive)
logger.info('3D -> 2D projection in %.3f ms', (time.time() - t0) * 1e3)

t0 = time.time()
lookups1D = {key: np.squeeze(interp2d(amps, charges, lookups2D[key].T)(Adrive, charges))
             for key in lookups2D.keys()}
lookups1D['ng0'] = np.squeeze(interp2d(amps, charges, lookups2D['ng'].T)(0.0, charges))
logger.info('2D -> 1D projection in %.3f ms', (time.time() - t0) * 1e3)
