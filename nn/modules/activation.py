from .module import Module
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

    def _sigmoid(self):
        return np.where(self.X >=0, 
                (1 / (1 + np.exp(-self.X))),
                (np.exp(self.X) / (1 + np.exp(self.X))))

    def _sigmoid_derivative(self):
        der = self._sigmoid()
        return der * (1 - der)

    def forward(self, X):
        """
        Apply sigmoid activation 1 / (1 + e^(-x))
        """
        self.X = X
        return self._sigmoid()

    def backward(self, dout):
        """
        Backprop through sigmoid activation.
        """
        return self._sigmoid_derivative() * dout

    def __repr__(self):
        return f"Sigmoid()"

