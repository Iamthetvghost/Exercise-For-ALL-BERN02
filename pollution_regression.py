# -*- coding: utf-8 -*-
"""
Created on Tue Sep  18:09:07 2026

@author: Elena
"""

import pandas as pd
from function_local_regression import local_regression


data = pd.read_csv("pollution_cleaneddata.csv")

x = data["POOR"].values
y = data["MORT"].values

k = 60
x_0 = [10, 18, 25]

pred, se = local_regression(y, x, k, x_0)

for i in range(len(x_0)):
    print("POOR =", x_0[i])
    print("Prediction =", pred[i])
    print("SE =", se[i])
