import os
import glob
import re
import pickle
import numpy as np

from sonic.utils import getNeuronsDict

# Get list of implemented neurons names
neurons = list(getNeuronsDict().keys())

# Define root directory and filename regular expression
root = 'C:/Users/admin/Google Drive/PhD/NICE model/sonic/sonic/lookups'
rgxp = re.compile('([A-Za-z]*)_lookups_a([0-9.]*)nm_f([0-9.]*)kHz.pkl')

# For each neuron
for neuron in neurons:

    print('--------- Aggregating lookups for {} neuron -------'.format(neuron))

    # Get filepaths from all lookups file in directory
    neuron_root = '{}/{}'.format(root, neuron)
    filepaths = glob.glob('{}/*.pkl'.format(neuron_root))

    # Create empty frequencies list and empty directory to hold aggregated coefficients
    freqs = []
    agg_coeffs = {}
    ifile = 0

    # Loop through each lookup file
    for fp in filepaths:

        # Get information from filename (a, f)
        filedir, filename = os.path.split(fp)
        mo = rgxp.fullmatch(filename)
        if mo:
            name = mo.group(1)
            a = float(mo.group(2)) * 1e-9  # nm
            f = float(mo.group(3)) * 1e3  # Hz
        else:
            print('error: lookup file does not match regular expression pattern')
            quit()

        # Add lookup frequency to list
        freqs.append(f)

        print('f =', f * 1e-3, 'kHz')

        # Open file and get coefficients dictionary
        with open(fp, 'rb') as fh:
            coeffs = pickle.load(fh)

        # If first file: initialization steps
        if ifile == 0:
            # Get names of output coefficients
            coeffs_keys = [ck for ck in coeffs.keys() if ck not in ['A', 'Q']]

            # Save input coefficients vectors separately
            A = coeffs['A']
            Q = coeffs['Q']
            print('nQ = ', len(Q))

            # Initialize aggregating dictionary of output coefficients with empty lists
            agg_coeffs = {ck: [] for ck in coeffs_keys}

        # Append current coefficients to corresponding list in aggregating dictionary
        for ck in coeffs_keys:
            agg_coeffs[ck].append(coeffs[ck])

        # Increment file index
        ifile += 1

    # Transform lists of 2D arrays into 3D arrays inside aggregating dictionary
    for ck in agg_coeffs.keys():
        # shape = agg_coeffs[ck][0].shape
        # for tmp in agg_coeffs[ck]:
        #     if tmp.shape != shape:
        #         print('dimensions error:', shape, tmp.shape)
        #         quit()
        print(ck, len(agg_coeffs[ck]), [tmp.shape for tmp in agg_coeffs[ck]])
        agg_coeffs[ck] = np.array(agg_coeffs[ck])

    # Add the 3 input vectors to the dictionary
    agg_coeffs['f'] = np.array(freqs)
    agg_coeffs['A'] = A
    agg_coeffs['Q'] = Q

    # Print out all coefficients names and dimensions
    for ck in agg_coeffs.keys():
        print(ck, agg_coeffs[ck].shape)

    # Save aggregated lookups in file
    filepath_out = '{}/{}_lookups_a{:.1f}nm.pkl'.format(root, neuron, a * 1e9)
    print(filepath_out)
    # with open(filepath_out, 'wb') as fh:
    #     pickle.dump(agg_coeffs, fh)
