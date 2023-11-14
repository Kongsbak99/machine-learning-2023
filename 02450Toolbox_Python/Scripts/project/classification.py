# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from project_ETL import *

# Prepare data for classification
X = final_raw_standardized.drop(columns=["ocean_proximity_" + name for name in classNames])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on test set
y_pred = clf.predict(X_test)

# Evaluate the classifier
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plotting the confusion matrix using seaborn
conf_mat = confusion_matrix(y_test, y_pred)
conf_df = pd.DataFrame(conf_mat, index=classNames, columns=classNames)
plt.figure(figsize=(10,7))
sns.heatmap(conf_df, annot=True, cmap="Blues", fmt='g')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
