#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import numpy as np


class MyLinearRegression:
    """
    Description:
        My personal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        if (
            isinstance(thetas, list)
            and thetas
            and len(thetas) == 2
            and isinstance(alpha, float)
            and alpha > 0
            and isinstance(max_iter, int)
        ):
            self.alpha = alpha
            self.max_iter = max_iter
            self.thetas = np.reshape(thetas, (2, 1))
        else:
            return

    def fit_(self, x, y):
        if (
            isinstance(x, np.ndarray)
            and x.size != 0
            and x.shape[1] == 1
            and isinstance(y, np.ndarray)
            and y.size != 0
            and x.shape == y.shape
        ):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            for _ in range(self.max_iter):
                gradient = np.dot(x.transpose(), x @ self.thetas - y) / x.shape[0]
                self.thetas = self.thetas - self.alpha * gradient
            return self.thetas
        return

    def predict_(self, x):
        if isinstance(x, np.ndarray) and x.size != 0:
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            return x @ self.thetas
        return

    def loss_elem_(self, y, y_hat):
        if (
            isinstance(y, np.ndarray)
            and y.size != 0
            and isinstance(y_hat, np.ndarray)
            and y_hat.size != 0
            and y.shape == y_hat.shape
            and y.shape[1] == 1
        ):
            return np.array(
                [(y_hat[i] - y[i]) * (y_hat[i] - y[i]) for i in range(y.shape[0])]
            )
        return

    def loss_(self, y, y_hat):
        if (
            isinstance(y, np.ndarray)
            and y.size != 0
            and isinstance(y_hat, np.ndarray)
            and y_hat.size != 0
            and y.shape == y_hat.shape
            and y.shape[1] == 1
        ):
            return np.square(y_hat - y).sum() / (2 * y.shape[0])
        return


if __name__ == "__main__":
    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLinearRegression([2, 0.7])

    print(lr1.predict_(x))
    # Output: array([[10.74695094],[17.05055804],[24.08691674],[36.24020866],[42.25621131]])

    print(lr1.loss_elem_(lr1.predict_(x), y))
    # Output: array([[710.45867381],[364.68645485],[469.96221651],[108.97553412],[299.37111101]])

    print(lr1.loss_(lr1.predict_(x), y))
    # Output: 195.34539903032385

    lr2 = MyLinearRegression([1, 1], 5e-8, 1500000)
    lr2.fit_(x, y)
    print(lr2.thetas)
    # Output: array([[1.40709365],[1.1150909]])

    print(lr2.predict_(x))
    # Output: array([[15.3408728],[25.38243697],[36.59126492],[55.95130097],[65.53471499]])

    print(lr2.loss_elem_(lr2.predict_(x), y))
    # Output: array([[486.66604863],[115.88278416],[84.16711596],[85.96919719],[35.71448348]])

    print(lr2.loss_(lr2.predict_(x), y))
    # Output: 80.83996294128525
