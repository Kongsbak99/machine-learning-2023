# exercise 1.5.1
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('project/Data/housing.csv')


# Remove rows with outlying grouped observations
df = df[df['median_house_value'] < 500000]
df = df[df['housing_median_age'] < 50]
df = df[df['median_income'] < 15]

# Remove almost empty data
df = df[df['total_rooms'] < 20000]
df = df[df['total_bedrooms'] < 4000]
df = df[df['population'] < 12000]
df = df[df['households'] < 4000]


#remove missing values 
raw_data_clean = df.dropna(axis = 0) 



# Extract class names to python list,
# then encode with integers (dict)
attributeNames = raw_data_clean.columns
classNames = raw_data_clean['ocean_proximity'].unique()
classDict = dict(zip(classNames, range(len(classNames))))
classLabels = raw_data_clean['ocean_proximity']

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

N = len(df)
M = len(attributeNames)

# raw with dommies
final_raw = pd.get_dummies(raw_data_clean, columns=['ocean_proximity'])

# Separate the continuous features and the dummy variables
continuous_features = final_raw.drop(columns=[col for col in final_raw if col.startswith('ocean_proximity_')])
dummy_variables = final_raw[[col for col in final_raw if col.startswith('ocean_proximity_')]]

# Standardize the continuous features manually
means = continuous_features.mean()
stds = continuous_features.std()
continuous_features_standardized = (continuous_features - means) / stds

# Concatenate the standardized features with the dummy variables
final_raw_standardized = pd.concat([continuous_features_standardized, dummy_variables], axis=1)

