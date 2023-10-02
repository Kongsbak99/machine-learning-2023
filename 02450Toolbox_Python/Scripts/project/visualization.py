# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:46:10 2023

@author: marcu
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.stats import norm

# Load the data
data_path = '../housing.csv'
df = pd.read_csv(data_path, delimiter=',')

# Create the new features
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['rooms_per_household'] = df['total_rooms'] / df['households']

# One-hot encode the 'ocean_proximity' column
df_encoded = pd.get_dummies(df, columns=['ocean_proximity'])

# Compute the correlation matrix
corr_matrix = df_encoded.corr()

# Mask the upper triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the matplotlib figure
plt.figure(figsize=(14, 12))

# Use the 'PuBu' color palette
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

plt.title('Correlation Matrix Heatmap')
plt.show()

# Compute the average median income for each 'ocean_proximity' category
avg_median_income = df.groupby('ocean_proximity')['median_house_value'].mean()
print(avg_median_income)

# Calculate skewness for all columns
skewness_values = df.skew()
# Print the skewness values
print(skewness_values)

# Standardize the numeric columns
numeric_columns = df.select_dtypes(include=[np.number])
df_standardized = (numeric_columns - numeric_columns.mean()) / numeric_columns.std()

# Number of columns for visualization
n_cols = len(df_standardized.columns)

# Create subplots
fig, axes = plt.subplots(nrows=n_cols, figsize=(10, 4 * n_cols))

for ax, column in zip(axes, df_standardized.columns):
    sns.distplot(df_standardized[column], bins=50, kde=True, color='b', ax=ax, fit=norm)
    ax.set_title(column)

plt.tight_layout()
plt.show()
