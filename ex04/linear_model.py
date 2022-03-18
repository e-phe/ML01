#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os


class MyLinearRegression:
    def __init__(self, thetas=np.array([[0.0], [0.0]])):
        if (
            isinstance(thetas, np.ndarray)
            and thetas.size != 0
            and thetas.shape == (2, 1)
        ):
            self.alpha = 0.01
            self.max_iter = 10000
            self.thetas = thetas
        else:
            return

    def fit_(self, x, y):
        if check_matrix(x) and check_matrix(y) and x.shape == y.shape:
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            for _ in range(self.max_iter):
                gradient = np.dot(np.transpose(x), x @ self.thetas - y) / x.shape[0]
                self.thetas = self.thetas - self.alpha * gradient

    def predict_(self, x):
        if check_matrix(x):
            x = np.insert(x, 0, values=1.0, axis=1).astype(float)
            return x @ self.thetas

    def plot_(self, x, y, predict):
        if (
            check_matrix(x)
            and check_matrix(y)
            and check_matrix(predict)
            and x.shape == y.shape
            and x.shape == predict.shape
        ):
            plt.grid()
            plt.plot(x, predict, c="g", linestyle="dashed", label="S predict (pills)")
            plt.scatter(x, predict, c="g", marker="P", label="S predict (pills)")
            plt.scatter(x, y, c="c", label="S true (pills)")
            plt.legend()
            plt.xlabel("Quantity of blue pill (in micrograms)")
            plt.ylabel("Space driving score")
            plt.show()

    def loss_plot_(self, x, y):
        if check_matrix(x) and check_matrix(y) and x.shape == y.shape:
            x_inter = np.arange(70, 110, 0.5)
            y_inter = np.arange(-5, -12.6, -0.2)

            X, Y = np.meshgrid(x_inter, y_inter)

            thetas = np.c_[np.ravel(X), np.ravel(Y)]

            # all results predicted
            y_hats = thetas.dot(np.c_[np.ones(x.shape[0]), x].T)
            # all results subtracted
            y_subs = y_hats - np.repeat(y.T, repeats=y_hats.shape[0], axis=0)
            # square the result
            y_squared = np.square(y_subs)
            # sum them
            y_add = np.sum(y_squared, axis=1)
            # divide by the number of values
            y_cost = y_add / x.shape[1]

            y_cost[y_cost > 500] = 500

            # costs for each combination
            Z = y_cost.reshape(X.shape)

            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            surf = ax.plot_surface(
                X,
                Y,
                Z,
                rstride=1,
                cstride=1,
                cmap=cm.RdBu,
                linewidth=0,
                antialiased=False,
            )

            ax.zaxis.set_major_locator(ticker.LinearLocator(10))
            ax.zaxis.set_major_formatter(ticker.FormatStrFormatter("%.02f"))

            fig.colorbar(surf, shrink=0.5, aspect=5)

            ax.set_xlabel("theta0")
            ax.set_ylabel("theta1")
            ax.set_zlabel("cost")

            plt.show()

    def mse_(self, y, predict):
        if check_matrix(y) and check_matrix(predict) and y.shape == predict.shape:
            return np.square(predict - y).sum() / y.shape[0]


def check_matrix(matrix):
    if isinstance(matrix, np.ndarray) and matrix.size != 0 and matrix.shape[1] == 1:
        return True
    exit("Error matrix")


if __name__ == "__main__":
    try:
        if os.stat("../resources/are_blue_pills_magics.csv").st_size > 0:
            data = np.loadtxt(
                "../resources/are_blue_pills_magics.csv",
                dtype=float,
                delimiter=",",
                skiprows=1,
            )
        else:
            exit("FileNotFoundError")
    except:
        exit("FileNotFoundError")

    x = data[:, [1]]
    y = data[:, [2]]

    mlr = MyLinearRegression()
    mlr.fit_(x, y)
    predict = mlr.predict_(x)
    mlr.plot_(x, y, predict)
    mlr.loss_plot_(x, y)
    print(mlr.mse_(y, predict))
