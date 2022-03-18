#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for loop.
    The three arrays must have compatible shapes.
    Args:
    x: has to be a numpy.array, a matrix of shape m * 1.
    y: has to be a numpy.array, a vector of shape m * 1.
    theta: has to be a numpy.array, a 2 * 1 vector.
    Return:
    The gradient as a numpy.array, a vector of shape 2 * 1.
    None if x, y, or theta is an empty numpy.array.
    None if x, y and theta do not have compatible shapes.
    None if x, y or theta is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if (
        isinstance(x, np.ndarray)
        and x.size != 0
        and x.shape[1] == 1
        and isinstance(y, np.ndarray)
        and y.size != 0
        and x.shape == y.shape
        and isinstance(theta, np.ndarray)
        and theta.size != 0
        and theta.shape == (2, 1)
    ):
        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        return np.dot(x.transpose(), x @ theta - y) / x.shape[0]
    return


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])

    theta1 = np.array([[2], [0.7]])
    print(gradient(x, y, theta1))
    # Output: array([[-19.0342574],[-586.66875564]])
    [[-19.0342574], [-586.66875564]]

    theta2 = np.array([[1], [-0.4]])
    print(gradient(x, y, theta2))
    # Output: array([[-57.86823748],[-2230.12297889]])
