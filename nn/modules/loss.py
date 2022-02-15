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
        self.eps = 1e-8

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
            prediction: array of shape (N, 1) where N is the number of examples.
            target: array of target values of shape (N, 1) or (N)
        """
        self.N = len(prediction)
        self.prediction = prediction
        self.target = target
        if not self._is_valid_args():
            raise ValueError("Mismatched sizes for prediction and target")
        if len(self.target.shape) == 1:
            self.target = self.target.reshape(-1, 1)
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

    def _bce_loss(self):
        """
        Numerically stable binary cross entropy loss computation.
        """
        return np.sum(-(self.target * np.log(self.prediction + self.eps) +
            (1 - self.target) * np.log(1 - self.prediction + self.eps)))

    def _bce_backprop(self):
        """
        Gradient computation in backprop of binary cross entropy loss function.
        """
        return (((1 - self.target) / (1 - self.prediction + self.eps)) - \
                (self.target / self.prediction + self.eps))

    def forward(self, prediction, target):
        """
        Forward propagation to compute the binary cross entropy loss
        given the prediction and target.

        Args:
            prediction: probability matrix of shape (N, 1) where N is the
                number of examples.
            target: target labels of shape (N) or (N, 1)
        """
        self.N = len(prediction)
        self.prediction = prediction
        self.target = target
        if not self._is_valid_args():
            raise ValueError("Mismatched sizes for prediction and target")
        if len(self.target.shape) == 1:
            self.target = self.target.reshape(-1, 1)
        return self._bce_loss() / self.N

    def backward(self):
        return self._bce_backprop() / self.N

class BCEWithLogitsLoss(BCELoss):
    """
    Implements a logits computation along with binary cross entropy loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, X, target):
        """
        Forward propagation to compute logits and then the loss for the 
        given input.

        Args:
            X: array of shape (N, 1) where N is the number of examples.
            target: array of target labels of shape (N) or (N, 1)
        """
        self.X = X
        self.target = target
        self.prediction = sigmoid(self.X)
        self.N = len(self.prediction)
        if not self._is_valid_args():
            raise ValueError("Mismatched sizes for prediction and target")
        if len(self.target.shape) == 1:
            self.target = self.target.reshape(-1, 1)
        return self._bce_loss() / self.N

    def backward(self):
        """
        Backpropagate through the binary cross entropy and logit functions.
        """
        return (self.prediction - self.target) / self.N


class CrossEntropyLoss(Loss):
    """
    Cross entropy loss function for multiclass classification. Takes raw scores
    as input and computes softmax and then the negative log loss.
    """

    def __init__(self):
        super().__init__()

    def _softmax(self):
        """
        Numerically stable softmax function.
        """
        # score - max(score) for each example, to prevent overflow
        self.X = self.X - np.max(self.X, axis=1, keepdims=True)
        # clip to prevent underflow
        self.X = self.X.clip(-700)
        exp_X = np.exp(self.X)
        exp_sum_X = np.sum(exp_X, axis=1, keepdims=True)
        return exp_X / exp_sum_X

    def forward(self, X, Y):
        """
        Cross entropy loss forward function to compute loss from the given
        scores and target labels.

        Args:
            X: matrix of shape (N, K) where N is the number of examples and 
                K is the number of classes.
            Y: matrix of target labels of shape (N) or (N, K).
        """
        # next two lines used for validating args. remove.
        self.prediction = X
        self.target = Y
        self.X = X
        self.Y = Y
        self.N = len(self.X)
        if not self._is_valid_args():
            raise ValueError("Mismatched sizes for prediction and target")
        if len(self.Y.shape) > 1:
            self.Y = np.argmax(self.Y, axis=1)
        # calculate class probabilities for every example
        self.probs = self._softmax()
        # add eps for numerical stability when computing log
        self.probs += self.eps
        # calculate loss
        true_probs = self.probs[list(range(self.N)), self.Y]
        loss = -np.sum(np.log(true_probs)) / self.N
        return loss

    def backward(self):
        """
        Backprop though cross entropy loss function.
        """
        mask = np.zeros_like(self.X)
        mask[list(range(self.N)), self.Y] = 1
        return (self.probs - mask) * (1 / self.N)


class NLLLoss(Loss):
    """
    Negative loss likelihood function for multiclass classification. Takes 
    softmax output as input and computes the resulting loss.
    """

    def __init__(self):
        super().__init__()

    def _one_hot_encoded_target(self):
        """
        Generates one hot encoded target matrix from the given target vector.
        """
        mask = np.zeros((self.N, self.K))
        mask[list(range(self.N)), self.Y] = 1
        return mask

    def forward(self, X, Y):
        """
        Computes negative log loss during the forward pass.

        Args:
            X: prediction probability matrix of shape (N, K)
            Y: target label matrix of shape (N) or (N, K)
        """
        # next two lines used for validating args. remove.
        self.prediction = X
        self.target = Y
        self.X = X
        self.Y = Y
        self.N, self.K = self.X.shape
        if not self._is_valid_args():
            raise ValueError("Mismatched sizes for prediction and target")
        if len(self.Y.shape) > 1:
            self.Y = np.argmax(self.Y, axis=1)
        # add eps for numerical stability in log computation
        self.X += self.eps
        self.tp_sum = self.X[list(range(self.N)), self.Y].reshape(self.N, 1)
        self.ltp_sum = np.log(self.tp_sum)
        loss = (-np.sum(self.ltp_sum) / self.N)
        return loss

    def backward(self):
        """
        Backprop through negative log loss function.
        """
        dltp_sum = -np.ones((self.N, 1))
        dtp_sum = dltp_sum * (1 / self.tp_sum)
        ohe_y = self._one_hot_encoded_target()
        dtp = dtp_sum * ohe_y
        dprobs = dtp * ohe_y
        return dprobs / self.N

