import numpy as np
import pandas as pd

import statsmodels.api as sm

from project_ETL import *

data = raw_data_clean 

print('Final data has ',data.shape[0],'rows and ',data.shape[1],' columns') 
data = pd.get_dummies(data, columns = ['ocean_proximity'], drop_first = True, dtype=int) # create dummy variables for categorical variable
#data = data.drop(['index']) # drop index column

# #Define X & Y variables for model building
X = data.copy().drop(['median_house_value'], axis =1 )
y_cont = data['median_house_value'] 

## median_house_value as reponse variable and the rest as predictors
print('######################### LINEAR REGRESSION #########################')
#add constant to predictor variables
X_int = sm.add_constant(X) # add intercept

#fit linear regression model
linreg = sm.OLS(y_cont, X_int).fit()
print(linreg.summary())