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
        dout[self.X <= 0] = 0
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


class Softmax(Module):
    """
    Implements Softmax activation function. Computes gradients analytically.
    """

    def __init__(self):
        super().__init__()

    def _softmax(self):
        """
        Implements a numerically stable softmax function.
        """
        # subtract max score from other scores for each
        # example to prevent overflow
        self.X -= np.max(self.X, axis=1, keepdims=True)
        # clip min scores to prevent underflow
        self.X = self.X.clip(-700)
        exp_x = np.exp(self.X)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        return (exp_x / sum_exp_x)

    def forward(self, X):
        """
        Softmax forward pass.
        """
        self.X = X
        self.N, self.K = self.X.shape
        self.probs = self._softmax()
        return self.probs

    def backward(self, dout):
        """
        Softmax backprop. Computes gradients analytically.
        """
        A = np.sum(dout * self.probs, axis=1, keepdims=True)
        dx = self.probs * (dout - A)
        return dx


class SoftmaxJacobian(Module):
    """
    Implements Softmax activation function. Computes jacobians to calculate
    gradients.
    """

    def __init__(self):
        super().__init__()

    def _softmax(self):
        """
        Implements a numerically stable softmax function.
        """
        # subtract max score from other scores for each
        # example to prevent overflow
        self.X -= np.max(self.X, axis=1, keepdims=True)
        # clip min scores to prevent underflow
        self.X = self.X.clip(-700)
        exp_x = np.exp(self.X)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        return (exp_x / sum_exp_x)

    def forward(self, X):
        """
        Softmax forward pass.
        """
        self.X = X
        self.N, self.K = self.X.shape
        self.probs = self._softmax()
        return self.probs

    def backward(self, dout):
        """
        Softmax backprop. Computes jacobians to calculate gradients.
        """
        p = self.probs
        # reshaped probs from (N, K) to (N, 1, K)
        p = p[:, np.newaxis]
        # subtract probs for each training example from identity
        q = np.identity(self.K) - p
        # compute the jacobian
        jacobians = p.transpose(0, 2, 1) * q
        # reshaped dout from (N, K) to (N, 1, K)
        dout_3d = dout[:, np.newaxis]
        return (dout_3d @ jacobians).reshape(self.N, self.K)


class SoftmaxCG(Module):
    """
    Softmax activation function. Uses computation graph to calculate gradients.
    """

    def __init__(self):
        super().__init__()

    def _softmax(self):
        """
        Implements a numerically stable softmax function.
        """
        # subtract max score from other scores for each
        # example to prevent overflow
        self.X -= np.max(self.X, axis=1, keepdims=True)
        # clip min scores to prevent underflow
        self.X = self.X.clip(-700)
        self.exp_x = np.exp(self.X)
        self.sum_exp_x = np.sum(self.exp_x, axis=1, keepdims=True)
        return (self.exp_x / self.sum_exp_x)

    def forward(self, X):
        """
        Forward pass of the softmax function.

        Args:
            X: matrix of shape (N, K) where N is the number of training
                examples and K is the number of classes.

        Returns:
            probability matrix of same shape as X.
        """
        self.X = X
        self.N, self.K = X.shape
        self.probs = self._softmax()
        return self.probs

    def backward(self, dout):
        """
        Backprop through the softmax function.

        Args:
            dout: matrix of gradients from above with same (N, K) shape as X 
            supplied during the forward pass.
        """
        isum_exp_x = 1 / self.sum_exp_x
        dexp_x = dout * isum_exp_x
        disum_exp_x = np.sum(dout * self.exp_x, axis=1, keepdims=True)
        dsum_exp_x = disum_exp_x * (-1 / self.sum_exp_x ** 2)
        dexp_x += dsum_exp_x * np.ones((self.N, self.K))
        dx = dexp_x * self.exp_x
        return dx
