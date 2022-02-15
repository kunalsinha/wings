from .module import Module
from .functional import sigmoid, sigmoid_derivative
import numpy as np

class Loss:
    """
    Base class for all loss functions.
    """

    def __init__(self):
        self.N = None
        self.prediction = None
        self.target = None

    def _is_valid_args(self):
        pred_len = len(self.prediction)
        targ_len = len(self.target)
        return pred_len == targ_len

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

class MSELoss(Loss):
    """
    Implements a mean squared loss function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction, target):
        """
        Forward propagation to compute the mean squared loss given the 
        target and prediction.

        Args:
            target: vector of real values
            prediction: vector of real values
        """
        self.N = len(prediction)
        self.prediction = prediction
        self.target = target
        if not self._is_valid_args():
            raise ValueError("Mismatched sizes for prediction and target")
        return np.sum((prediction - target) ** 2) / (2 * self.N)

    def backward(self):
        """
        Backprop to calculate the gradients.
        """
        return (self.prediction - self.target) / self.N

class BCELoss(Loss):
    """
    Implements a binary cross entropy loss function.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def _bce_loss(self):
        return np.sum(-(self.target * np.log(self.prediction + self.eps) +
            (1 - self.target) * np.log(1 - self.prediction + self.eps)))

    def _bce_backprop(self):
        return (((1 - self.target) / (1 - self.prediction + self.eps)) - \
                (self.target / self.prediction + self.eps))

    def forward(self, prediction, target):
        """
        Forward propagation to compute the binary cross entropy loss
        given the prediction and target.
        """
        self.N = len(prediction)
        self.prediction = prediction
        self.target = target
        if not self._is_valid_args():
            raise ValueError("Mismatched sizes for prediction and target")
        return self._bce_loss() / self.N

    def backward(self):
        return self._bce_backprop() / self.N

class BCEWithLogitsLoss(BCELoss):
    """
    Implements a logits computation along with binary cross entropy loss.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-8


    def forward(self, X, target):
        """
        Forward propagation to compute logits and then the loss for the 
        given input.
        """
        self.X = X
        self.target = target
        self.prediction = sigmoid(self.X)
        self.N = len(self.prediction)
        if not self._is_valid_args():
            raise ValueError("Mismatched sizes for prediction and target")
        return self._bce_loss() / self.N

    def backward(self):
        """
        Backpropagate through the binary cross entropy and logit functions.
        """
        return (self.prediction - self.target) / self.N


class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def _softmax(self):
        # score - max(score) for each example, to prevent overflow
        self.X = self.X - np.max(self.X, axis=1, keepdims=True)
        # clip to prevent underflow
        self.X = self.X.clip(-700)
        exp_X = np.exp(self.X)
        exp_sum_X = np.sum(exp_X, axis=1, keepdims=True)
        return exp_X / exp_sum_X

    def forward(self, X, Y):
        self.prediction = X
        self.target = Y
        self.X = X
        self.Y = Y
        self.N = len(self.X)
        if not self._is_valid_args():
            raise ValueError("Mismatched sizes for prediction and target")
        # calculate class probabilities for every example
        self.probs = self._softmax()
        # calculate loss
        true_probs = self.probs[list(range(self.N)), self.Y]
        loss = -np.sum(np.log(true_probs)) / self.N
        return loss

    def backward(self):
        mask = np.zeros_like(self.X)
        mask[list(range(self.N)), self.Y] = 1
        return (self.probs - mask) * (1 / self.N)

