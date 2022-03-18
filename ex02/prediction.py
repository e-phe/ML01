#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of shape m * 1.
    theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of shape m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta shapes are not appropriate.
    None if x or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        isinstance(x, np.ndarray)
        and x.size != 0
        and isinstance(theta, np.ndarray)
        and theta.size != 0
    ):
        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        return x @ theta
    return
