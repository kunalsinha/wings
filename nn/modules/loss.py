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
        loss =  np.sum((prediction - target) ** 2)
        return loss * (0.5 / self.N)

    def backward(self):
        """
        Backprop to calculate the gradients.
        """
        grad = 2 * (self.prediction - self.target)
        return grad * (0.5 / self.N)

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
        loss = self._bce_loss()
        return loss / self.N

    def backward(self):
        grad = self._bce_backprop()
        return grad / self.N

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
        loss = self._bce_loss()
        return loss / self.N

    def backward(self):
        """
        Backpropagate through the binary cross entropy and logit functions.
        """
        dyhat = self._bce_backprop()
        dz = dyhat * self.prediction * (1 - self.prediction)
        return dz / self.N
