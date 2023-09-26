# exercise 1.5.1
import numpy as np
import pandas as pd



raw_data = pd.read_csv('project/Data/housing.csv')

#remove missing values 
raw_data_clean = raw_data.dropna(axis = 0) 

# Extract class names to python list,
# then encode with integers (dict)
attributeNames = raw_data_clean.columns
classNames = raw_data_clean['ocean_proximity'].unique()
classDict = dict(zip(classNames, range(len(classNames))))
classLabels = raw_data_clean['ocean_proximity']

# Extract vector y, convert to NumPy array
y = np.asarray([classDict[value] for value in classLabels])

N = len(raw_data_clean)
M = len(attributeNames)