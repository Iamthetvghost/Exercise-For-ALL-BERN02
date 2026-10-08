# Sleep Duration Prediction — Model Selection Competition

## Introduction

This exercise is a Kaggle competition based on a dataset describing sleep quality.

The response variable is `Sleep_Duration`, and the goal is to predict sleep duration from the available descriptors.

**Kaggle username:** Yaxuan Fu

The model selection methods required for this exercise are:
- best subset selection, forward selection, or backward selection
- shrinkage methods

I compare several models using cross-validation and use **mean squared error (MSE)** as the main model selection criterion. After selecting the final model, it is fitted on the full training data and used to predict the hold-out test set for submission to Kaggle.

The final model is selected based on both its cross-validated performance and its predictive performance on the Kaggle hold-out data.


## 1. Data
### 1.1 Load the data
The training data contain the response variable `Sleep_Duration`, while the test data contain the predictors for the hold-out observations.
The `ID` column is treated as an identifier rather than a predictor.

### 1.2 Prepare the predictors
The response variable is `Sleep_Duration`.
The `ID` column is excluded from the predictors because it is an identifier.
Categorical variables are one-hot encoded, while numerical variables are standardized when required by the model.
`handle_unknown="ignore"` allows the preprocessing pipeline to handle categories that may occur in a validation or test set but not in the corresponding training split.



## 2. Model evaluation
The Kaggle competition evaluates predictions using mean residual sum of squares, i.e. mean squared error (MSE).
Therefore, MSE is used as the main criterion for comparing the models.
Lower MSE indicates better predictive performance.

### 2.1 Linear regression baseline
I first fit an ordinary linear regression model as a baseline.
This model assumes an additive linear relationship between the predictors and sleep duration.

### 2.2 Forward selection
Forward selection starts with no predictors and adds predictors sequentially according to their contribution to predictive performance.
Here the selection is performed using the numerical predictors. The selected variables are then evaluated using 5-fold cross-validation.

### 2.3 Backward selection
Backward selection starts with all numerical predictors and removes predictors sequentially.
The resulting subset is evaluated using the same 5-fold cross-validation procedure.

### 2.4 Best subset selection
Best subset selection evaluates different combinations of the numerical predictors.
For each subset size, all possible combinations are considered and evaluated using cross-validation. The subset with the lowest cross-validated MSE is selected.

### 2.5 Ridge regression
Ridge regression is a shrinkage method that penalizes large regression coefficients.
The regularization strength is selected by cross-validation.

### 2.6 Lasso regression
Lasso is another shrinkage method. In addition to shrinking coefficients, it can set some coefficients exactly to zero, providing a form of variable selection.
The regularization strength is selected using cross-validation.

### 2.7 Compare the required model-selection approaches
| Model | CV_MSE |
|---|---:|
| Lasso | 0.096360 |
| Ridge | 0.103551 |
| Linear Regression | 0.124837 |
| Best Subset | 0.170414 |
| Backward Selection | 0.172205 |
| Forward Selection | 0.172809 |

### Model selection results

Among the linear and shrinkage-based models, Lasso gives the lowest cross-validated MSE.

However, the relatively high MSE compared with the nonlinear models tested below suggests that a purely linear relationship may not be sufficient to describe the data.



## 3. Nonlinear predictive models
After evaluating the required model-selection approaches, I also compare several tree-based models.
These models can capture nonlinear relationships and interactions between predictors that are not represented by ordinary linear regression or shrinkage models.

### 3.1 HistGradientBoosting
HistGradientBoosting CV MSE: 0.034627
CV standard deviation: 0.018192

### 3.2 Random Forest
| max_features | min_samples_leaf | CV_MSE | CV_SD |
|---|---:|---:|---:|
| 0.5 | 2 | 0.026446 | 0.004247 |
| 1.0 | 2 | 0.026750 | 0.014962 |
| sqrt | 1 | 0.029543 | 0.005217 |
| sqrt | 2 | 0.038813 | 0.008092 |
| sqrt | 4 | 0.052382 | 0.010436 |

### 3.3 Extra Trees regression
Extra Trees is another tree ensemble method. It introduces additional randomization when constructing the individual trees.
The best-performing configuration from the model comparison uses all available features at each split and a minimum leaf size of 2.

### 3.4 Overall model comparison
| Model | CV_MSE |
|---|---:|
| Extra Trees | 0.015682 |
| Random Forest | 0.026446 |
| HistGradientBoosting | 0.034627 |
| Lasso | 0.096360 |
| Ridge | 0.103551 |
| Linear Regression | 0.124837 |
| Best Subset | 0.170414 |
| Backward Selection | 0.172205 |
| Forward Selection | 0.172809 |

The cross-validation results show a clear improvement when moving from linear and shrinkage models to nonlinear tree-based models.
Among the required model-selection approaches, Lasso gives the lowest cross-validated MSE. However, Extra Trees achieves a substantially lower cross-validated MSE than both Lasso and the other models considered.
This suggests that nonlinear relationships and interactions between the descriptors are important for predicting sleep duration.
Therefore, Extra Trees is selected as the final predictive model.
The final model is fitted using the complete training dataset before generating predictions for the Kaggle hold-out data.


## Summary
I initially selected Lasso regression as my final model because it was the best-performing shrinkage method among the linear models considered. My first Kaggle submission was therefore based on the Lasso model and achieved a public leaderboard MSE of **0.05511**.
After seeing the Kaggle result, I explored several nonlinear predictive models, including HistGradientBoosting, Random Forest, and Extra Trees. These models performed substantially better in cross-validation, with Extra Trees achieving the lowest cross-validated MSE of approximately **0.0155**.
I therefore selected Extra Trees as the final model and submitted its predictions to Kaggle. The Extra Trees submission achieved a public leaderboard MSE of **0.01596**, which was a substantial improvement over the Lasso submission.
This comparison suggests that nonlinear relationships and interactions between the sleep-related descriptors are important for predicting sleep duration, and that the Extra Trees model was better suited to this dataset than the linear shrinkage model.
