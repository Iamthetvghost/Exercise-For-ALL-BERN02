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

    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
        
    if k > len(x):
        raise ValueError("k cannot be larger than the number of observations.")    

    if k < 3:
        raise ValueError("k must be at least 3.")  # 局部线性回归至少需要估计两个参数

    pred = []
    se = []

    for x0 in x_0:

        distances = np.abs(x - x0)

        indices = np.argsort(distances)[:k]

        x_local = x[indices]
        y_local = y[indices]

        X = np.column_stack((np.ones(k), x_local))

        beta = np.linalg.lstsq(X, y_local, rcond=None)[0] 
        # np.linalg.lstsq()直接用来计算最小二乘解 Used directly to calculate the least squares solution

        beta0 = beta[0]
        beta1 = beta[1]

        prediction = beta0 + beta1 * x0

        pred.append(prediction)

        residuals = y_local - X @ beta #残差

        degrees_of_freedom = k - 2 #自由度

        residual_variance = np.sum(residuals**2) / degrees_of_freedom #方差

        x0_vector = np.array([1.0, x0])

        covariance_matrix = (
            residual_variance * np.linalg.inv(X.T @ X)
        )   # Cov(beta) = s^2 (X^T X)^(-1)  #协方差矩阵

        prediction_variance = (
            x0_vector @ covariance_matrix @ x0_vector
        )

        prediction_se = np.sqrt(prediction_variance) #标准差

        se.append(prediction_se)

    return np.array(pred), np.array(se)
