import os
import pickle
import numpy as np

# Define frequencies
# freqs = np.append(np.arange(50, 1001, 50), 690.0) * 1e3
freqs = np.arange(50, 1001, 50) * 1e3

# Locate lookup files
for Fdrive in freqs:
    lookup_file_in = 'Tcell_lookups_a32.0nm_f{:.1f}kHz.pkl'.format(Fdrive * 1e-3)
    lookup_file_out = 'LeechT_lookups_a32.0nm_f{:.1f}kHz.pkl'.format(Fdrive * 1e-3)
    lookup_path_in = 'lookups/LeechT/{}'.format(lookup_file_in)
    lookup_path_out = 'lookups/LeechT/{}'.format(lookup_file_out)

    # Load coefficients
    assert os.path.isfile(lookup_path_in), 'Error: no lookup file for these stimulation parameters'
    print('modifying dict keys in "{}"'.format(lookup_path_in))

    # Load file
    with open(lookup_path_in, 'rb') as fh:
        coeffs = pickle.load(fh)

    print('keys:')
    print(coeffs.keys())

    # m-gate
    coeffs['alpham'] = coeffs['miTm_Na']
    coeffs['betam'] = coeffs['invTm_Na'] - coeffs['alpham']
    coeffs.pop('miTm_Na')
    coeffs.pop('invTm_Na')

    # h-gate
    coeffs['alphah'] = coeffs['hiTh_Na']
    coeffs['betah'] = coeffs['invTh_Na'] - coeffs['alpham']
    coeffs.pop('hiTh_Na')
    coeffs.pop('invTh_Na')

    # n-gate
    coeffs['alphan'] = coeffs['miTm_K']
    coeffs['betan'] = coeffs['invTm_K'] - coeffs['alphan']
    coeffs.pop('miTm_K')
    coeffs.pop('invTm_K')

    # s-gate
    coeffs['alphas'] = coeffs['miTm_Ca']
    coeffs['betas'] = coeffs['invTm_Ca'] - coeffs['alphas']
    coeffs.pop('miTm_Ca')
    coeffs.pop('invTm_Ca')


    print('new keys:')
    print(coeffs.keys())

    # Save new dict in file
    with open(lookup_path_out, 'wb') as fh:
        pickle.dump(coeffs, fh)
