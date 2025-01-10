# Basic Libraries
import pandas as pd
import numpy as np

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# File and Execution Time Handling
from scipy.io import arff

# Load the dataset
data, meta = arff.loadarff('phishing_data.arff')
dataset = pd.DataFrame(data)

# Step 1: General Information
print("Dataset dimensions (Rows, Columns):", dataset.shape)
print("Column names:")
print(dataset.columns)

# Step 2: Target class distribution
print("\nTarget class distribution:")
print(dataset['Result'].value_counts())

# Step 3: Check for missing values
print("\nMissing values in each column:")
print(dataset.isnull().sum())
# Remove the byte format (decode byte strings)
dataset = dataset.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Convert target column 'Result' to integer
dataset['Result'] = dataset['Result'].astype(int)

# Display the updated dataset
print("\nUpdated Dataset (first 5 rows):")
print(dataset.head())

# Features and target variable
X = dataset.drop('Result', axis=1)
y = dataset['Result']

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Check the new class distribution
print("Balanced training set class distribution:")
print(pd.Series(y_resampled).value_counts())

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)

# visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Phishing (-1)', 'Legitimate (1)'], yticklabels=['Phishing (-1)', 'Legitimate (1)'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.savefig('confusion_matrix.pdf', dpi=300, bbox_inches='tight')
plt.savefig('/Users/hakan/PhishingProject/confusion_matrix.png', dpi=300, bbox_inches='tight')

# Create and train the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred_dt = dt_model.predict(X_test)

# Evaluate performance
print("Decision Tree - Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree - Classification Report:")
print(classification_report(y_test, y_pred_dt))
print("\nDecision Tree - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Phishing (-1)', 'Legitimate (1)'], 
            yticklabels=['Phishing (-1)', 'Legitimate (1)'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Decision Tree - Confusion Matrix")
plt.savefig('decision_tree_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Create and train the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_resampled, y_resampled)

# Make predictions on the test set
y_pred_nb = nb_model.predict(X_test)

# Evaluate performance
print("Naive Bayes - Accuracy:", accuracy_score(y_test, y_pred_nb))
print("\nNaive Bayes - Classification Report:")
print(classification_report(y_test, y_pred_nb))
print("\nNaive Bayes - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# Visualize the Confusion Matrix
cm = confusion_matrix(y_test, y_pred_nb)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=['Phishing (-1)', 'Legitimate (1)'], 
            yticklabels=['Phishing (-1)', 'Legitimate (1)'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes - Confusion Matrix")
plt.savefig('naive_bayes_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()