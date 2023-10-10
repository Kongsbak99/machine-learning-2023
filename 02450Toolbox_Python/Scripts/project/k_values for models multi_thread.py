# -*- coding: utf-8 -*-
"""
Created on Tue Oct 3 16:27:55 2023
@author: marcu
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor
import concurrent.futures  # Import concurrent.futures module

# Load the data
data_path = '../housing.csv'
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

# Features (X), Target (y)
X = df[['median_income', 'total_rooms']]
y = df['median_house_value']

# Define a range of values for k1 and k2
k1_values = [4, 5, 6, 7]  # outer fold
k2_values = [2, 3, 4, 5]  # inner fold

# Initialize a dictionary to store the errors for each model and each combination of k1 and k2
Egen_dict = OrderedDict()

# Initialize a list to store the models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0, max_iter=1000),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

print("Starting")

# Initialize lists to store results
model_names = list(models.keys())
Egen_values = {model_name: [] for model_name in model_names}

# Loop through different values of k2 (keeping k1 constant)
for k2 in k2_values:
    print(f"Inner Folds (k2): {k2}")
    Egen_dict[k2] = OrderedDict()

    # Outer cross-validation loop (k1-fold)
    for k1 in k1_values:
        print(f"Outer Folds (k1): {k1}")
        Egen_dict[k2][k1] = OrderedDict()

        # Loop through different models
        for model_name, model in models.items():
            Egen_dict[k2][k1][model_name] = []

        def train_and_evaluate_model(model_name, X_train, y_train, X_val, y_val):
            # Train and evaluate each of the models
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)

            return mse

        # Outer cross-validation loop (k1-fold)
        outer_kf = KFold(n_splits=k1, shuffle=True, random_state=42)
        for i, (par_idx, test_idx) in enumerate(outer_kf.split(X)):
            X_par, X_test = X.iloc[par_idx], X.iloc[test_idx]
            y_par, y_test = y.iloc[par_idx], y.iloc[test_idx]

            E_s_dict = OrderedDict()

            # Inner cross-validation loop (k2-fold)
            inner_kf = KFold(n_splits=k2, shuffle=True, random_state=42)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_model = {executor.submit(train_and_evaluate_model, model_name, X_par.iloc[train_idx], y_par.iloc[train_idx], X_par.iloc[val_idx], y_par.iloc[val_idx]): model_name for train_idx, val_idx in inner_kf.split(X_par)}

                for future in concurrent.futures.as_completed(future_to_model):
                    model_name = future_to_model[future]
                    mse = future.result()

                    if model_name not in E_s_dict:
                        E_s_dict[model_name] = []

                    E_s_dict[model_name].append(mse)

            # Compute E(s)_k2 for each model
            for model_name, mse_values in E_s_dict.items():
                E_s_k2 = np.mean(mse_values)
                Egen_dict[k2][k1][model_name].append(E_s_k2)

# Create a bar chart to visualize the change in performance for each k1 value
plt.figure(figsize=(10, 6))
for model_name in model_names:
    Egen_values[model_name] = [np.mean(Egen_dict[k2][k1][model_name]) for k1 in k1_values]
    plt.plot(k1_values, Egen_values[model_name], marker='o', label=model_name)

plt.xlabel('Outer Folds (k1)')
plt.ylabel('Mean Generalization Error')
plt.title('Change in Performance for Different k1 Values')
plt.legend()
plt.grid()
plt.show()

print("finished")
