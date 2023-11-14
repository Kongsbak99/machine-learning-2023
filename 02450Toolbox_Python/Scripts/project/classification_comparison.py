# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from statsmodels.stats.contingency_tables import mcnemar

from project_ETL import *


# remove ISLAND for classification since cross validation is diffucult with only 2 entries
# final_raw_standardized = final_raw_standardized[final_raw_standardized['ocean_proximity'] != 'ISLAND']

# Update 'y' variable after removing the 'ISLAND' entries
# y = np.asarray([classDict[value] for value in final_raw_standardized['ocean_proximity']])




# Prepare data for classification
X = final_raw_standardized.drop(columns=["ocean_proximity_" + name for name in classNames]).to_numpy()

# Define hyperparameter grids for grid search
tree_params = {
    'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}
logistic_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100]
}

outer_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = []

# Collection predictionbs
predictions_logistic = []
predictions_tree = []
predictions_baseline = []
# Add container to collect true labels
true_labels = []
# Outer fold cross-validation
for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Inner fold cross-validation for Method 2 (Classification Tree)
    print('Inner fold cross-validation for Method 2 (Classification Tree)')
    tree_gs = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=5, scoring='accuracy')
    tree_gs.fit(X_train, y_train)
    
    tree_best_model = tree_gs.best_estimator_
    tree_test_accuracy = accuracy_score(y_test, tree_best_model.predict(X_test))
    
    # Inner fold cross-validation for Logistic Regression
    print('Inner fold cross-validation for Logistic Regression')
    logistic_gs = GridSearchCV(LogisticRegression(max_iter=100000), logistic_params, cv=5, scoring='accuracy')
    logistic_gs.fit(X_train, y_train)
    
    logistic_best_model = logistic_gs.best_estimator_
    logistic_test_accuracy = accuracy_score(y_test, logistic_best_model.predict(X_test))
    
    # Baseline: Predicting the most frequent class
    print('Baseline: Predicting the most frequent class')
    most_frequent_class = np.bincount(y_train).argmax()
    baseline_accuracy = accuracy_score(y_test, [most_frequent_class]*len(y_test))
    
    # Collect true labels
    true_labels.append(y[test_idx])

    # Collect predictions
    logistic_predictions = logistic_best_model.predict(X_test)
    tree_predictions = tree_best_model.predict(X_test)
    baseline_predictions = [most_frequent_class] * len(y_test)

    predictions_logistic.append(logistic_predictions)
    predictions_tree.append(tree_predictions)
    predictions_baseline.append(baseline_predictions)

    # Collect results
    results.append({
        'Outer fold': outer_cv.get_n_splits() - len(results),
        'Method 2 best param': tree_gs.best_params_['max_depth'],
        'Method 2 error rate': 1 - tree_test_accuracy,
        'Logistic best param': logistic_gs.best_params_['C'],
        'Logistic error rate': 1 - logistic_test_accuracy,
        'Baseline error rate': 1 - baseline_accuracy,
    })

results_df = pd.DataFrame(results)

print(results_df)

# Statistical tests
# Flatten the lists of predictions and true values
# y_true = y[test_idx]
y_true = np.concatenate(true_labels)
flat_logistic = np.concatenate(predictions_logistic)
flat_tree = np.concatenate(predictions_tree)
flat_baseline = np.concatenate(predictions_baseline)

# Function to perform McNemar's test
def perform_mcnemars_test(model1, model2):
    # Build contingency table
    tb = confusion_matrix(model1 == y_true, model2 == y_true)
    # Perform McNemar's test
    result = mcnemar(tb, exact=False, correction=True)
    return result.statistic, result.pvalue

# Compare Logistic Regression vs. Decision Tree
statistic, pvalue = perform_mcnemars_test(flat_logistic, flat_tree)
print(f'McNemar’s test statistic between Logistic Regression and Decision Tree: {statistic}, p-value: {pvalue}')

# Compare Logistic Regression vs. Baseline
statistic, pvalue = perform_mcnemars_test(flat_logistic, flat_baseline)
print(f'McNemar’s test statistic between Logistic Regression and Baseline: {statistic}, p-value: {pvalue}')

# Compare Decision Tree vs. Baseline
statistic, pvalue = perform_mcnemars_test(flat_tree, flat_baseline)
print(f'McNemar’s test statistic between Decision Tree and Baseline: {statistic}, p-value: {pvalue}')

"""
# ---- Logistic Regression ----
lambdas = [0.001, 0.01, 0.1, 1, 10, 100]
best_accuracy_lr = 0
best_lambda = 0
print("\n")
for lam in lambdas:
    log_reg = LogisticRegression(C=1/lam, max_iter=1000, multi_class='auto', solver='lbfgs')
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy_lr:
        best_accuracy_lr = accuracy
        best_lambda = lam

print(f"Best accuracy for Logistic Regression: {best_accuracy_lr} with lambda: {best_lambda}")

# ---- Classification Trees (CT) ----
depths = list(range(1, 11))
best_accuracy_ct = 0
best_depth = 0

for depth in depths:
    clf = DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > best_accuracy_ct:
        best_accuracy_ct = accuracy
        best_depth = depth

print(f"Best accuracy for Classification Trees: {best_accuracy_ct} with depth: {best_depth}")

# ---- Baseline Model (Majority Class Classifier) ----
majority_class = np.bincount(y_train).argmax()
y_pred_baseline = [majority_class] * len(y_test)
accuracy_baseline = accuracy_score(y_test, y_pred_baseline)

print(f"Accuracy for Baseline Model (Majority Class Classifier): {accuracy_baseline}")
print(" ")

"""


