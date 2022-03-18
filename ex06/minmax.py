#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.array using the min-max standardization.
    Args:
    x: has to be an numpy.array, a vector.
    Return:
    x’ as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception.
    """
    if isinstance(x, np.ndarray) and x.size != 0 and x.shape[1] == 1:
        normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
        return np.ravel(normalized)
    return


if __name__ == "__main__":
    X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
    print(minmax(X))
    # Output: array([0.58333333, 1., 0.33333333, 0.77777778, 0.91666667, 0.66666667, 0.])

    Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
    print(minmax(Y))
    # Output: array([0.63636364, 1., 0.18181818, 0.72727273, 0.93939394, 0.6969697 , 0.])
