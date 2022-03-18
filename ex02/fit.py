#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
    y: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
    theta: has to be a numpy.array, a vector of shape 2 * 1.
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of shape 2 * 1.
    None if there is a matching shape problem.
    None if x, y, theta, alpha or max_iter is not of the expected type.
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
        and isinstance(alpha, float)
        and alpha > 0
        and isinstance(max_iter, int)
    ):
        x = np.insert(x, 0, values=1.0, axis=1).astype(float)
        for _ in range(max_iter):
            gradient = np.dot(x.transpose(), x @ theta - y) / x.shape[0]
            theta = theta - alpha * gradient
        return theta
    return


from prediction import predict_

if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    theta = np.array([[1], [1]])

    theta1 = fit_(x, y, theta, alpha=5e-6, max_iter=15000)
    print(theta1)
    # Output: array([[1.40709365], [1.1150909]])

    print(predict_(x, theta1))
    # Output: array([[15.3408728], [25.38243697], [36.59126492], [55.95130097], [65.53471499]])
