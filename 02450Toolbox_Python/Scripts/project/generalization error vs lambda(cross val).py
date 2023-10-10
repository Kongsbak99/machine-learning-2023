# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:21:29 2023

@author: marcu
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
data_path = 'housing.csv'
df = pd.read_csv(data_path, delimiter=',')

# One-hot encode the 'ocean_proximity' column
df = pd.get_dummies(df, columns=['ocean_proximity'])

# Remove rows with outlying grouped observations
df = df[df['median_house_value'] < 500000]
df = df[df['housing_median_age'] < 50]
df = df[df['median_income'] < 15]

# Remove almost empty data
df = df[df['total_rooms'] < 20000]
df = df[df['total_bedrooms'] < 4000]
df = df[df['population'] < 12000]
df = df[df['households'] < 4000]

# Split the data into features (X) and target (y)
X = df[['median_income']]
y = df['median_house_value']

# Define a range of lambda values
lambda_values = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 500, 1000]

# Initialize lists to store mean cross-validation scores for each lambda
cross_val_scores = []

# Initialize a dictionary to store the MSE for each lambda
mse_dict = {lambda_val: [] for lambda_val in lambda_values}

# Outer loop (k2-fold cross-validation)
k2 = 5
outer_kf = KFold(n_splits=k2, shuffle=True, random_state=42)

for train_idx, test_idx in outer_kf.split(X):
    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]

    # Inner loop (k1-fold cross-validation)
    k1 = 5
    inner_kf = KFold(n_splits=k1, shuffle=True, random_state=42)

    for train_idx_inner, _ in inner_kf.split(X_train_outer):
        X_train_inner, y_train_inner = X_train_outer.iloc[train_idx_inner], y_train_outer.iloc[train_idx_inner]

        for lambda_val in lambda_values:
            ridge = Ridge(alpha=lambda_val)
            ridge.fit(X_train_inner, y_train_inner)
            y_pred = ridge.predict(X_test_outer)
            mse = mean_squared_error(y_test_outer, y_pred)
            mse_dict[lambda_val].append(mse)

# Calculate the mean MSE for each lambda value
for lambda_val, mse_values in mse_dict.items():
    mean_mse = np.mean(mse_values)
    cross_val_scores.append(mean_mse)

# Plot the estimated generalization error as a function of λ
plt.plot(lambda_values, cross_val_scores, marker='o')
plt.xlabel('Lambda (Regularization Parameter)')
plt.ylabel('Mean Cross-Validation MSE')
plt.title('Generalization Error vs. Lambda (Nested Cross-Validation)')
plt.xscale('log')
plt.grid(True)
plt.show()

