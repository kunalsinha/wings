from .module import Module
from .functional import sigmoid, sigmoid_derivative
import numpy as np

class ReLU(Module):
    """
    Implements a ReLU activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        Apply ReLU function max(0, x)

        Args:
            X: input array
        """
        self.X = X
        return np.maximum(X, 0)

    def backward(self, dout):
        """
        Backprop through ReLU activation.
        """
        dout[self.X <=0] = 0
        return dout

    def __repr__(self):
        return f"ReLU()"


class Sigmoid(Module):
    """
    Implements a sigmoid activation function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, X):
        """
        Apply sigmoid activation 1 / (1 + e^(-x))
        """
        self.X = X
        self.X_sigmoid = sigmoid(self.X)
        return self.X_sigmoid

    def backward(self, dout):
        """
        Backprop through sigmoid activation.
        """
        return self.X_sigmoid * (1 - self.X_sigmoid) * dout

    def __repr__(self):
        return f"Sigmoid()"

