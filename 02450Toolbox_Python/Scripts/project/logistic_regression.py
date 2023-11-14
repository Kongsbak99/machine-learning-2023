# Import necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Import your ETL script
import project_ETL

# Use the variables from your ETL script
X = project_ETL.final_raw_standardized.drop(columns=[col for col in project_ETL.final_raw_standardized if col.startswith('ocean_proximity_')])
y = project_ETL.y

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
chosen_C = 1  # Replace with the chosen value from your exercises
logistic_model = LogisticRegression(C=chosen_C, max_iter=10000)
logistic_model.fit(X_train, y_train)

# Evaluate the model
predictions = logistic_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Feature Importance
coefficients = logistic_model.coef_[0]
feature_names = X.columns
feature_importance = pd.DataFrame(coefficients, index=feature_names, columns=['Importance'])
feature_importance['Absolute Importance'] = feature_importance['Importance'].abs()
feature_importance = feature_importance.sort_values(by='Absolute Importance', ascending=False)

print(feature_importance)

# Save the model, if needed
# import joblib
# joblib.dump(logistic_model, 'logistic_model.pkl')
