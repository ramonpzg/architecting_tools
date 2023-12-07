import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha  # Regularization strength
        self.max_iter = max_iter  # Maximum number of iterations for optimization
        self.tol = tol  # Tolerance to determine convergence
        self.weights = None  # Coefficients

    def fit(self, X, y):
        # Initialize coefficients with zeros
        self.weights = np.zeros(X.shape[1] + 1)

        # Add a column of ones to the feature matrix for the intercept term
        X_augmented = np.column_stack([np.ones(X.shape[0]), X])

        # Initial cost and gradient
        cost, gradient = self._cost_and_gradient(X_augmented, y, self.weights)

        # Iterative optimization using gradient descent
        for iteration in range(self.max_iter):
            # Update weights using gradient descent
            self.weights -= self.alpha * gradient

            # Calculate new cost and gradient
            new_cost, new_gradient = self._cost_and_gradient(X_augmented, y, self.weights)

            # Check for convergence
            if np.abs(new_cost - cost) < self.tol:
                break

            # Update cost and gradient for the next iteration
            cost, gradient = new_cost, new_gradient

    def _cost_and_gradient(self, X, y, weights):
        # Calculate the cost (objective function) and gradient for Lasso regression
        n_samples = X.shape[0]
        predictions = np.dot(X, weights)
        residuals = predictions - y

        # Cost (objective function) term
        cost = (1 / (2 * n_samples)) * np.sum(residuals**2)

        # L1 regularization term
        l1_term = self.alpha * np.sum(np.abs(weights[1:]))

        # Total cost
        total_cost = cost + l1_term

        # Gradient of the cost with respect to weights
        gradient = (1 / n_samples) * np.dot(X.T, residuals) + self.alpha * np.sign(weights)
        gradient[0] -= self.alpha * np.sign(weights[0])  # Exclude the intercept term

        return total_cost, gradient

    def predict(self, X):
        # Add a column of ones for the intercept term
        X_augmented = np.column_stack([np.ones(X.shape[0]), X])

        # Make predictions
        predictions = np.dot(X_augmented, self.weights)

        return predictions

# Example usage:
# X_train and y_train are your training data
# X_test is your test data
# lasso_model = LassoRegression(alpha=0.01, max_iter=1000, tol=1e-4)
# lasso_model.fit(X_train, y_train)
# predictions = lasso_model.predict(X_test)
