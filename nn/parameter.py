try:
    import cupy as np
except Exception:
    import numpy as np


class Parameter:
    """
    Class for storing model parameters. Provides functionality to 
    store data and gradients for the parameter.
    """

    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(self.data)
        self.v = np.zeros_like(self.data)
        self.s = np.zeros_like(self.data)

    def __repr__(self):
        return str(f"Parameter containing:\n{repr(self.data)}")

    def zero_grad(self):
        """
        Reset the gradient to zero.
        """
        self.grad.fill(0)
