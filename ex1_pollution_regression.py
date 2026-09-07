# -*- coding: utf-8 -*-
"""
Created on Tue Sep  18:09:07 2026

@author: Elena
"""

import numpy as np
import pandas as pd
from function_local_regression import local_regression


data = pd.read_csv("pollution_cleaneddata.csv")

x = data["POOR"].values
y = data["MORT"].values


cv_errors = []

for k in range(3, len(x)):
    errors = []

    for i in range(len(x)):
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)

        pred_i, _ = local_regression(
            y_train,
            x_train,
            k,
            [x[i]]
        )

        errors.append((y[i] - pred_i[0]) ** 2)

    cv_error = np.mean(errors)
    cv_errors.append(cv_error)


best_k = np.argmin(cv_errors) + 3

print("Best k =", best_k)
print("Cross-validation error =", min(cv_errors))


k = best_k
x_0 = [10, 18, 25]

pred, se = local_regression(y, x, k, x_0)

for i in range(len(x_0)):
    print("POOR =", x_0[i])
    print("Prediction =", pred[i])
    print("SE =", se[i])
