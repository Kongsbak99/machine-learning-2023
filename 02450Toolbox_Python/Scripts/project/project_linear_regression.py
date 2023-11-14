import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from project_ETL import *

data = raw_data_clean 

print('Final data has ',data.shape[0],'rows and ',data.shape[1],' columns') 
#data = pd.get_dummies(data, columns = ['ocean_proximity'], drop_first = True, dtype=int) # create dummy variables for categorical variable
#data = data.drop(['index']) # drop index column

# #Define X & Y variables for model building
X = data.copy().drop(['median_house_value'], axis =1 ) #i.e. use all attributes as features
X_stnd = X - 1*X.mean(0)
X_stnd = X_stnd*(1/np.std(X_stnd,0))

y_cont = data['median_house_value'] 

## median_house_value as reponse variable and the rest as predictors
print('######################### LINEAR REGRESSION #########################')
#add constant to predictor variables
X_int = sm.add_constant(X_stnd) # add intercept

#fit linear regression model OLS = linear regression
linreg = sm.OLS(y_cont, X_int).fit()
print(linreg.summary())


#To use a regularization parameter (lambda) we need to use Ridge regression
print("######################### RIDGE REGRESSION #########################")
# Define the lambda (alpha) value
alpha_value = 1.0  # Adjust based on cross-validation results

# Create the Ridge regression model
ridge_reg = Ridge(alpha=alpha_value)

# Fit the model
ridge_reg.fit(X_stnd, y_cont)

#compute R squared value
r2 = ridge_reg.score(X_stnd, y_cont)

# Print summary
print(f"R^2 Value: {r2}")
print("Regularization Parameter (Alpha):", alpha_value)
print("Intercept:", ridge_reg.intercept_)
print("\nCoefficients:")
for col, coef in zip(X_stnd.columns, ridge_reg.coef_):
    print(f"{col}: {coef}")




## RANDOM FOREST 
print("######################### RANDOM FOREST #########################")
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_stnd, y_cont, test_size=0.2, random_state=42)

# Initialize the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)


# Calculate the R^2 value
r2 = rf.score(X_test, y_test)
print(f"R^2 Value: {r2}")
# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
# Print feature importances
print("\nFeature Importances:")
for col, importance in zip(X_stnd.columns, rf.feature_importances_):
    print(f"{col}: {importance}")


