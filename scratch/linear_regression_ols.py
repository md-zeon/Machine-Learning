import numpy as np

class LinearRegression:
    def __init__(self):
        self.bias = None
        self.weights = None

    def fit(self, X, y):
        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]  # Add bias term

        # lstsq also handles redundant or perfectly correlated features.
        theta = np.linalg.lstsq(X_b, y, rcond=None)[0]

        self.bias = theta[0]
        self.weights = theta[1:]
 
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