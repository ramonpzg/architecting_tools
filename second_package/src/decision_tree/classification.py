import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        unique_classes = np.unique(y)

        # Base cases
        if len(unique_classes) == 1:
            # If all labels are the same, create a leaf node
            return {'class': unique_classes[0]}

        if self.max_depth is not None and depth == self.max_depth:
            # If max depth is reached, create a leaf node with the majority class
            majority_class = np.argmax(np.bincount(y))
            return {'class': majority_class}

        # Find the best split
        best_split = self._find_best_split(X, y)

        if best_split is None:
            # If no split improves purity, create a leaf node with the majority class
            majority_class = np.argmax(np.bincount(y))
            return {'class': majority_class}

        feature_index, threshold = best_split

        # Split the data
        left_indices = X[:, feature_index] <= threshold
        right_indices = ~left_indices

        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        # Return a node representing the split
        return {'feature_index': feature_index, 'threshold': threshold,
                'left': left_subtree, 'right': right_subtree}

    def _find_best_split(self, X, y):
        num_features = X.shape[1]
        best_gini = float('inf')
        best_split = None

        for feature_index in range(num_features):
            feature_values = np.unique(X[:, feature_index])
            for value in feature_values:
                left_indices = X[:, feature_index] <= value
                right_indices = ~left_indices

                gini = self._calculate_gini_index(y[left_indices], y[right_indices])

                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature_index, value)

        return best_split

    def _calculate_gini_index(self, left_labels, right_labels):
        total_size = len(left_labels) + len(right_labels)

        if total_size == 0:
            return 0

        p_left = len(left_labels) / total_size
        p_right = len(right_labels) / total_size

        gini_left = 1 - np.sum((np.bincount(left_labels) / len(left_labels))**2)
        gini_right = 1 - np.sum((np.bincount(right_labels) / len(right_labels))**2)

        gini_index = p_left * gini_left + p_right * gini_right

        return gini_index

    def predict(self, X):
        return np.array([self._predict_tree(self.tree, x) for x in X])

    def _predict_tree(self, node, x):
        if 'class' in node:
            # If it's a leaf node, return the predicted class
            return node['class']
        else:
            # Traverse the tree recursively
            if x[node['feature_index']] <= node['threshold']:
                return self._predict_tree(node['left'], x)
            else:
                return self._predict_tree(node['right'], x)

# Example usage:
# X_train and y_train are your training data
# X_test is your test data
# dt_classifier = DecisionTreeClassifier(max_depth=3)
# dt_classifier.fit(X_train, y_train)
# predictions = dt_classifier.predict(X_test)
