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

    def __call__(self, X):
        return self.forward(X)

    def __repr__(self):
        return f"ReLU()"
