# -*- coding: utf-8 -*-
"""
Created on Tue Sep 1 2026

@author: Elena (Yaxuan Fu)
"""


import numpy as np


def local_regression(y, x, k, x_0):

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_0 = np.asarray(x_0, dtype=float)

    # Check input lengths
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    if k < 3:
        raise ValueError("k must be at least 3.")

    if k > len(x):
        raise ValueError("k cannot be larger than the number of observations.")

    pred = []
    se = []

    for x0 in x_0:

        # Calculate distances from x0
        distances = np.abs(x - x0)

        # Select k nearest observations
        indices = np.argsort(distances)[:k]

        x_local = x[indices]
        y_local = y[indices]
        distances_local = distances[indices]

        # Tricube weights: w_i = [1 - (d_i / d_max)^3]^3
        max_dist = np.max(distances_local)

        if max_dist == 0:
            weights = np.ones(k)
        else:
            u = distances_local / max_dist
            weights = (1 - u**3)**3


        X = np.column_stack((np.ones(k), x_local))

        sqrt_weights = np.sqrt(weights)

        X_weighted = X * sqrt_weights[:, np.newaxis]
        y_weighted = y_local * sqrt_weights

        beta = np.linalg.lstsq(
            X_weighted,
            y_weighted,
            rcond=None
        )[0]

        beta0 = beta[0]
        beta1 = beta[1]

        prediction = beta0 + beta1 * x0

        pred.append(prediction)

        residuals = y_local - X @ beta

        degrees_of_freedom = k - 2

        residual_variance = (
            np.sum(residuals**2) / degrees_of_freedom
        )


        x_mean = np.mean(x_local)

        SSX = np.sum((x_local - x_mean)**2)

        prediction_variance = (
            residual_variance
            * (
                1 / k
                + (x0 - x_mean)**2 / SSX
            )
        )

        prediction_se = np.sqrt(prediction_variance)

        se.append(prediction_se)
