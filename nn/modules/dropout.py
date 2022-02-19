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
        N, H = X.shape
        self.mask = np.random.rand(N, H) > self.prob
        # print(self.mask)
        return X * self.mask

    def backward(self, dout):
        """
        Backpropagates through the dropupt layer.
        """
        return dout * self.mask
