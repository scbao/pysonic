import numpy as np
from pyDOE import lhs


def lh2DWithCorners(n, x1_range, x2_range, crtrn):
    ''' This function generates a 2D Latin Hypercube distribution vector, scaled up
        to the input domain range, and containing the 4 corners of the domain.

        :param n: number of samples to generate (including the 4 corners)
        :param x1_range: range of the 1st input variable
        :param x2_range: range of the 2nd input variable
        :param crtrn: criterion for Latin Hypercube sampling
        :return: 2xn array of generated samples
    '''

    lh = lhs(2, samples=(n - 4), criterion=crtrn)
    corners = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    lhc = np.vstack((lh, corners))
    lhc[:, 0] = lhc[:, 0] * (x1_range[1] - x1_range[0]) + x1_range[0]
    lhc[:, 1] = lhc[:, 1] * (x2_range[1] - x2_range[0]) + x2_range[0]
    return lhc
