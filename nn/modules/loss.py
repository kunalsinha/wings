from .module import Module
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

    def __init__(self, reduction='mean'):
        """
        Args:
            reduction (str): "mean", "sum"
        """
        super().__init__()
        self.reduction = reduction

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
        if self.reduction == "mean":
            loss *= (0.5 / self.N)
        return loss

    def backward(self):
        """
        Backprop to calculate the gradients.
        """
        grad = 2 * (self.prediction - self.target)
        if self.reduction == "mean":
            grad *= (0.5 / self.N)
        return grad


class BCELoss(Loss):
    """
    Implements a binary cross entropy loss function.
    """

    def __init__(self, reduction="mean"):
        """
        Args:
            reduction (str): "mean", "sum"
        """
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-8

    def forward(self, prediction, target):
        """
        Forward propagation to compute the binary cross entropy loss
        given the prediction and target.
        """
        self.N = len(prediction)
        self.prediction = prediction
        self.target = target
        #print(self.prediction)
        #print(self.target)
        if not self._is_valid_args():
            raise ValueError("Mismatched sizes for prediction and target")
        loss = - np.sum((self.target * np.log(self.prediction + self.eps) + 
                 (1 - self.target) * np.log(1 - self.prediction + self.eps)))
        if self.reduction == "mean":
            loss /= self.N
        return loss

    def backward(self):
        grad = np.sum(np.log(1 - self.prediction + self.eps) - 
                np.log(self.prediction + self.eps))
        if self.reduction == "mean":
            grad /= self.N
        return grad

class BCEWithLogitsLoss(Loss):
    """
    Implements a logits computation along with binary cross entropy loss.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-8

    def _sigmoid(self):
        return np.where(self.X >= 0, 
                1 / (1 + np.exp(-self.X)),
                np.exp(self.X) / (1 + np.exp(self.X)))

    def _bce_loss(self):
        return np.sum(-(self.target * np.log(self.prediction + self.eps) +
            (1 - self.target) * np.log(1 - self.prediction + self.eps)))

    def forward(self, X, target):
        """
        Compute logits and then the loss for the given input.
        """
        self.X = X
        self.target = target
        self.prediction = self._sigmoid()
        return self._bce_loss()

    def backward(self):
        dyhat = ((1 - self.target) / (1 - self.prediction + self.eps)) - \
                (self.target / self.prediction + self.eps)
        dz = dyhat * self.prediction * (1 - self.prediction)
        return dz
