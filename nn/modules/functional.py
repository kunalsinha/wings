import numpy as np

def sigmoid(X):
    """
    Implements a numerically stable sigmoid function.
    """
    return np.where(X >= 0, 1 / (1 + np.exp(-X)), np.exp(X) / (1 + np.exp(X)))

def sigmoid_derivative(X):
    derivative = sigmoid(X)
    return derivative * (1 - derivative)

