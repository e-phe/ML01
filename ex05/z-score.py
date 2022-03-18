#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np

def zscore(x):
    """Computes the normalized version of a non-empty numpy.array using the z-score standardization.
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
        mean = sum(x) / x.shape[0]
        std = np.sqrt(np.square(x - mean).sum() / x.shape[0])
        x = (x - mean) / std
        return np.ravel(x)
    return

if __name__ == "__main__":
    X = np.array([[0],[ 15],[ -9],[ 7],[ 12],[ 3],[ -21]])
    print(zscore(X))
    # Output: array([-0.08620324, 1.2068453, -0.86203236, 0.51721942, 0.94823559, 0.17240647, -1.89647119])

    Y = np.array([[2],[ 14],[ -13],[ 5],[ 12],[ 4],[ -19]])
    print(zscore(Y))
    # Output: array([ 0.11267619, 1.16432067, -1.20187941, 0.37558731, 0.98904659, 0.28795027, -1.72770165])
