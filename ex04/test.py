#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from linear_model import MyLinearRegression as MyLR

if __name__ == "__main__":
    data = pd.read_csv("../resources/are_blue_pills_magics.csv")
    Xpill = np.array(data["Micrograms"]).reshape(-1, 1)
    Yscore = np.array(data["Score"]).reshape(-1, 1)

    linear_model1 = MyLR(np.array([[89.0], [-8]]))
    Y_model1 = linear_model1.predict_(Xpill)
    print(linear_model1.mse_(Yscore, Y_model1))
    # Output:
    # 57.60304285714282
    print(linear_model1.mse_(Yscore, Y_model1) == mean_squared_error(Yscore, Y_model1))

    linear_model2 = MyLR(np.array([[89.0], [-6]]))
    Y_model2 = linear_model2.predict_(Xpill)
    print(linear_model2.mse_(Yscore, Y_model2))
    # Output:
    # 232.16344285714285
    print(linear_model2.mse_(Yscore, Y_model2) == mean_squared_error(Yscore, Y_model2))
