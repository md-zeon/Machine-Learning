import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.bias = None
        self.weights = None
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def fit(self, X, y): # X train, y train
        # X.shape => (# of samples, # of features)
        m, n = X.shape

        # STEP 1: Initialize parameters
        self.bias = 0
        self.weights = np.zeros(n) # [0, 0, 0, ..., 0] (n times)

        # Gradient Descent
        for _ in range(self.n_iterations):
            # STEP 2: Compute predictions
            y_pred = self.bias + np.dot(X, self.weights) # y_pred = bias + (X * weights)

            # STEP 3: Compute gradients
            db = (1/m) * np.sum(y_pred - y) # derivative of bias
            dw = (1/m) * np.dot(X.T, (y_pred - y)) # derivative of weights

            # STEP 4: Convergence theorem - Update parameters
            self.bias -= self.learning_rate * db
            self.weights -= self.learning_rate * dw


    def predict(self, X):
        y_pred = self.bias + np.dot(X, self.weights)
        return y_pred


X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([3, 5, 7, 9])

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
print("Predictions:", y_pred)

print("Bias:", model.bias)
print("Weights:", model.weights)
