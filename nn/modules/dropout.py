from .module import Module
import numpy as np


class Dropout(Module):
    """
    Implement dropout layer.
    """

    def __init__(self, prob):
        """
        Args:
            prob: Dropout probability between 0 and 1
        """
        super().__init__()
        if prob < 0 or prob > 1:
            raise ValueError("Invalid dropout probability")
        self.prob = prob

    def forward(self, X):
        """
        Forward propagates through the dropout layer.
        """
        if self._mode == "train":
            self.mask = np.random.rand(*X.shape) > self.prob
            # E[X] = p*0 + (1-p)*X = (1-p)*X
            # So divide by (1-p) to ensure consistency during test time
            out = (X * self.mask) / (1 - self.prob)
        else:
            out = X
        return out

    def backward(self, dout):
        """
        Backpropagates through the dropupt layer.
        """
        return dout * self.mask
