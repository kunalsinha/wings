from .module import Module
import numpy as np

class MSE:
    """
    Implements a mean squared loss cost function.
    """

    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def forward(self, prediction, target):
        """
        Forward propagation to compute the mean squared loss given the 
        target and prediction.

        Args:
            target: vector of real values
            prediction: vector of real values
        """
        self.target = target
        self.prediction = prediction
        self.N = len(prediction)
        if self.N != len(target):
            raise ValueError("Mismatched sizes for prediction and target")
        loss =  np.sum((prediction - target) ** 2)
        if self.reduction == 'mean':
            loss *= (0.5 / self.N)
        return loss

    def backward(self):
        """
        Backprop to calculate the gradients.
        """
        grad = 2 * (self.prediction - self.target)
        if self.reduction == "mean":
            grad /= (2 * self.N)
        return grad

    def __call__(self, prediction, target):
        return self.forward(prediction, target)


class Softmax:
    """
    Implements a softmax loss function.
    """

    def __init__(self, prediction, target):
        pass
