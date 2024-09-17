import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from random import randrange
from sklearn.metrics import accuracy_score
import pickle
import sys

# Create a sample dataset
data = sys.path.append(
    '/content/drive/MyDrive/Medicine-Recommendation-Sytem-master/medicinerecommend')

# Split the dataset into X and y
X = data.drop(['medicine'], axis=1)

y = data['medicine']


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


# placing the sets into train and test

# Save the training data to CSV
train_data = pd.concat([X_train, y_train], axis=1)
train_data.to_csv('training_data.csv', index=False)

# Save the test data to CSV
test_data = pd.concat([X_test, y_test], axis=1)
test_data.to_csv('test_data.csv', index=False)


data = X
target = y


# Define a function to calculate the Gini index
def calc_gini(data, target):
    n = len(data)
    gini = 0
    for c in np.unique(target):
        p = np.sum(target == c) / n
        gini += p * (1 - p)
    return gini

# The calc_gini function returns the Gini index, which is a measure of impurity in a dataset. The Gini index ranges from 0 to 1, where 0 represents a pure dataset (all samples belong to the same class) and 1 represents a completely impure dataset (samples are evenly distributed across all classes).

# Define a function to 
# calculate the Gini index for a split

def calc_split_gini(data, feature, target):
    n = len(data)
    left = data[feature] < data[feature].mean()
    right = ~left
    gini_left = calc_gini(data[left], target[left])
    gini_right = calc_gini(data[right], target[right])
    return gini_left * sum(left) / n + gini_right * sum(right) / n

# return the weighted sum of the Gini indices for the left and right subsets.
# This information can be used to further split the dataset into left and right subsets based on the split point determined by the mean value of the feature. By doing this, it allows the decision tree algorithm to recursively partition the data into smaller and smaller subsets until the stopping criterion is met, such as reaching the maximum depth or minimum number of samples required to create a node.


# Define a function to find the best split for a given feature
def find_best_split(data, feature, target):
    split_gini = [calc_split_gini(data, feature, target)
                  for feature in data.columns]
    best_feature = np.argmin(split_gini)
    return best_feature, split_gini[best_feature]

# it returns the index of the best feature and the Gini index for the best split.


# Define the Node class to represent a decision tree node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None
# value: the predicted value for this node if it is a leaf node. If this node is not a leaf node, value is set to None.


# Define the DecisionTree class to build and use a decision tree
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

# Fit the decision tree
    def fit(self, X, y):
        self.n_features = X.shape[1]
        self.tree = self._build_tree(X, y)

# Predict the target variable of new data X
    def predict(self, X):
        return np.array([self._traverse(x, self.tree) for x in X])

# Recursively build the decision tree by finding the best split

# The _build_tree method recursively constructs the tree by finding the best feature to split the data on and splitting the data into two subsets based on the threshold of that feature

# Build a decision Tree
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stop criteria
        if depth == self.max_depth or n_labels == 1 or n_samples < 2:
            return Node(value=np.argmax(np.bincount(y)))

        # Find the best split
        best_feature, best_gini = find_best_split(X, y, n_features)
        if best_gini == 0:
            return Node(value=np.argmax(np.bincount(y)))

        # Recursive splitting
        left = X.iloc[:, best_feature] < X.iloc[:, best_feature].mean()
        right = ~left
        left_tree = self._build_tree(X.loc[left], y[left], depth+1)
        right_tree = self._build_tree(X.loc[right], y[right], depth+1)

        return Node(best_feature, X.iloc[:, best_feature].mean(), left_tree, right_tree)
    

# The _traverse method takes an input sample and traverses the tree from the root node to a leaf node, classifying the sample based on the value stored at the leaf node.

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] < node.threshold:
            return self._traverse(x, node.left)
        else:
            return self._traverse(x, node.right)


# Sagun portion

#
def subsample(dataset, ratio):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# Make a prediction with a list of bagged trees
def bagging_predict(trees, row):
    predictions = [find_best_split(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)

# Random Forest Algorithm


def random_forest(train, test, max_depth, min_size, sample_size, n_trees, n_features):
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = DecisionTree(sample, max_depth, min_size, n_features)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return (predictions)


clr_rf = random_forest(X_train, y_train, max_depth=None, min_size=10,
                       sample_size=0.8, n_trees=100, n_features=3)  # model trained


# saving training dataset #saving for future
pickle.dump(clr_rf, open('model.pkl', 'wb'))
