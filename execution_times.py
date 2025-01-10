import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
import time  # For measuring execution time

# Load the dataset
data, meta = arff.loadarff('phishing_data.arff')
dataset = pd.DataFrame(data)

# Remove the byte format (decode byte strings)
dataset = dataset.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Convert target column 'Result' to integer
dataset['Result'] = dataset['Result'].astype(int)

# Features and target variable
X = dataset.drop('Result', axis=1)
y = dataset['Result']

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Logistic Regression - Measure execution time
start_time = time.time()
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_resampled, y_resampled)
logistic_end_time = time.time()
logistic_execution_time = logistic_end_time - start_time
print(f"Logistic Regression - Execution Time: {logistic_execution_time:.2f} seconds")

# Decision Tree - Measure execution time
start_time = time.time()
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_resampled, y_resampled)
dt_end_time = time.time()
dt_execution_time = dt_end_time - start_time
print(f"Decision Tree - Execution Time: {dt_execution_time:.2f} seconds")

# Naive Bayes - Measure execution time
start_time = time.time()
nb_model = GaussianNB()
nb_model.fit(X_resampled, y_resampled)
nb_end_time = time.time()
nb_execution_time = nb_end_time - start_time
print(f"Naive Bayes - Execution Time: {nb_execution_time:.2f} seconds")

# Summarize results
execution_times = {
    "Logistic Regression": logistic_execution_time,
    "Decision Tree": dt_execution_time,
    "Naive Bayes": nb_execution_time
}

print("\nExecution Times Summary:")
for model, time_taken in execution_times.items():
    print(f"{model}: {time_taken:.2f} seconds")
 