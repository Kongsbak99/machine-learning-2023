import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from collections import OrderedDict




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

# Features (X), Target (y)
X = df[['median_income', 'total_rooms']]
y = df['median_house_value']

# Number of models
S = 4

# k1 and k2
k1 = 2
k2 = 5


# Store the models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0, max_iter=1000),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    #'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# Store the errors for each model
gen_error = OrderedDict()


print("Starting")
# Outer cross-validation loop (k1)
outer_kf = KFold(n_splits=k1, shuffle=True, random_state=42)
for i, (par_idx, test_idx) in enumerate(outer_kf.split(X)):
    print(str(i) + "/4")
    X_par, X_test = X.iloc[par_idx], X.iloc[test_idx]
    y_par, y_test = y.iloc[par_idx], y.iloc[test_idx]

    E_s_dict = OrderedDict()

    # Inner cross-validation loop (k2)
    inner_kf = KFold(n_splits=k2, shuffle=True, random_state=42)
    for j, (train_idx, val_idx) in enumerate(inner_kf.split(X_par)):
        X_train, X_val = X_par.iloc[train_idx], X_par.iloc[val_idx]
        y_train, y_val = y_par.iloc[train_idx], y_par.iloc[val_idx]

        # Train and evaluate each of the S models
        for model_name, model in models.items():
            #print("Training and evaluating: " + str(model_name) )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            mse = mean_squared_error(y_val, y_pred)

            if model_name not in E_s_dict:
                E_s_dict[model_name] = []

            E_s_dict[model_name].append(mse)

    # Compute for each model and select the best model (M*)
    for model_name, mse_values in E_s_dict.items():
        E_s_k2 = np.mean(mse_values)
        if model_name not in gen_error:
            gen_error[model_name] = []
        gen_error[model_name].append(E_s_k2)

# Compute the estimate of the generalization error for each model
model_names = []
Egen_values = []
for model_name, mse_values in gen_error.items():
    Egen = np.mean(mse_values)
    model_names.append(model_name)
    Egen_values.append(Egen)
    print(f"Estimated Generalization Error for {model_name}: {Egen:.4f}")

# Create a bar chart to visualize the estimated generalization errors
plt.figure(figsize=(10, 6))
plt.bar(model_names, Egen_values)
plt.xlabel('Models')
plt.ylabel('Estimated Generalization Error')
plt.title('Generalization Error Comparison for Different Models')
plt.xticks(rotation=45)
plt.show()
print("finished")