# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 01:06:04 2026

@author: Elena
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln


# Read the data
data = pd.read_csv("bird_count.csv")

year = data["yr"].values
count = data["count"].values


# Center the year variable for numerical stability
year_c = year - np.mean(year)


# Define the negative log-likelihood
#
# Model:
# Y_i ~ Poisson(lambda_i)
# log(lambda_i) = beta0 + beta1 * year_c

def neg_log_likelihood(beta, year, count):
    
    beta0, beta1 = beta
    
    # Poisson mean
    lam = np.exp(beta0 + beta1 * year)
    
    # Log-likelihood
    log_likelihood = np.sum(
        count * np.log(lam)
        - lam
        - gammaln(count + 1)
    )
    
    # Minimize negative log-likelihood
    return -log_likelihood


# Estimate the parameters using maximum likelihood

initial_beta = np.array([1.0, 0.0])

result = minimize(
    neg_log_likelihood,
    initial_beta,
    args=(year_c, count)
)


# Check optimization
print("Optimization successful:", result.success)


# Estimated parameters
beta0, beta1 = result.x

print("Estimated parameters:")
print("beta0 =", beta0)
print("beta1 =", beta1)


# Calculate fitted Poisson means

lambda_hat = np.exp(beta0 + beta1 * year_c)

print("\nFitted mean counts:")
print(lambda_hat)


# Generate three hypothetical samples

np.random.seed(123)

sample_1 = np.random.poisson(lambda_hat)
sample_2 = np.random.poisson(lambda_hat)
sample_3 = np.random.poisson(lambda_hat)


# Combine the three samples

samples = pd.DataFrame({
    "sample": np.repeat([1, 2, 3], len(year)),
    "yr": np.tile(year, 3),
    "count": np.concatenate([
        sample_1,
        sample_2,
        sample_3
    ])
})


print("\nGenerated samples:")
print(samples)


# Save the samples as a CSV file

samples.to_csv("poisson_samples.csv", index=False)

print("\nSamples saved as poisson_samples.csv")
