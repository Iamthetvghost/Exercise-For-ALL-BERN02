# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 04:43:57 2026

@author: Elena
"""

import numpy as np

def local_regression(y, x, k, x_0):

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if k < 3:
        raise ValueError("k must be at least 3.")

    if k > len(x):
        raise ValueError("k cannot be larger than the number of observations.")

    pred = []
    se = []

    for x0 in x_0:

        distances = np.abs(x - x0)
        indices = np.argsort(distances)[:k]

        x_local = x[indices]
        y_local = y[indices]

        x_mean = np.mean(x_local)
        y_mean = np.mean(y_local)

        b1 = np.sum((x_local - x_mean) * (y_local - y_mean)) / np.sum((x_local - x_mean)**2)

        b0 = y_mean - b1 * x_mean

        prediction = b0 + b1 * x0
        pred.append(prediction)

        residuals = y_local - (b0 + b1 * x_local)

        residual_variance = np.mean(residuals**2)

        prediction_variance = residual_variance * (1 / k +(x0 - x_mean)**2 /np.sum((x_local - x_mean)**2))

        prediction_se = np.sqrt(prediction_variance)
        se.append(prediction_se)

    return np.array(pred), np.array(se)