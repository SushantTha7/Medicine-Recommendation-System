import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the datasets (using on_bad_lines='skip' to handle bad lines)
train = pd.read_csv("training_data.csv", on_bad_lines='skip')
test = pd.read_csv("test_data.csv", on_bad_lines='skip')

# Inspect the data (optional)
print(train.head())
print(test.head())
print(train.info())

# Prepare the training and testing datasets
y_train = train.prognosis
x_train = train.drop('prognosis', axis=1)

y_test = test.prognosis
x_test = test.drop('prognosis', axis=1)

# Initialize RandomForestClassifier from sklearn
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf_rf = clf.fit(x_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(x_test)

# Calculate evaluation metrics
ac = accuracy_score(y_test, y_pred)
ps = precision_score(y_test, y_pred, average='weighted')
rs = recall_score(y_test, y_pred, average='weighted')
fs = f1_score(y_test, y_pred, average='weighted')

# Print evaluation results
print('Accuracy:', ac)
print('Precision:', ps)
print('Recall:', rs)
print('F1-score:', fs)

# Optional: Save the model using pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(clf_rf, f)

# Optional: Load the model from the saved pickle file (if needed)
# with open('model.pkl', 'rb') as f:
#     clf_rf = pickle.load(f)

# Custom decision tree and random forest implementation (for experimentation)
class DecisionTree:
    def __init__(self):
        self.tree = None
    
    def fit(self, X, y):
        # Implement decision tree training logic
        pass
    
    def predict(self, X):
        # Implement decision tree prediction logic
        return np.zeros(X.shape[0])  # Placeholder prediction logic

class RandomForest:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = [DecisionTree() for _ in range(n_trees)]
    
    def fit(self, X, y):
        n_samples = X.shape[0]
        for i in range(self.n_trees):
            # Create a random sample of the training data
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]
            
            # Train a decision tree on the sample
            self.trees[i].fit(X_sample, y_sample)
    
    def predict(self, X):
        # Predict using all decision trees and aggregate results via majority vote
        predictions = np.zeros((X.shape[0], self.n_trees))
        for i in range(self.n_trees):
            predictions[:, i] = self.trees[i].predict(X)
        
        # Majority vote for the final prediction
        return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=predictions)

# Example of usage for custom RandomForest (if you want to experiment with it)
# custom_rf = RandomForest(n_trees=5)
# custom_rf.fit(x_train.to_numpy(), y_train.to_numpy())
# custom_y_pred = custom_rf.predict(x_test.to_numpy())
